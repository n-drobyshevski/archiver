"""Anthropic client wrapper: one structured-output call, stop-reason checks, usage tally."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import anthropic
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError

from archiver.config import Settings
from archiver.errors import LlmRefused, LlmTruncated
from archiver.models import Usage

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# $ per million tokens: (input, output). Cache reads are billed at a tenth of the input price.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
CACHE_READ_FACTOR = 0.1


def _price_for(model: str) -> tuple[float, float] | None:
    if model in PRICES:
        return PRICES[model]
    # Dated snapshots such as "claude-sonnet-5-5-20261001" share the alias price.
    for name, price in PRICES.items():
        if model.startswith(name + "-"):
            return price
    return None


def _is_fallback_rejection(exc: anthropic.BadRequestError) -> bool:
    text = str(exc).lower()
    return "fallback" in text or "beta" in text


class LLM:
    def __init__(self, settings: Settings, client: AsyncAnthropic | None = None) -> None:
        self.settings = settings
        self._client = client
        self._fallbacks = settings.refusal_fallbacks
        self.usage: dict[str, Usage] = {}

    @property
    def client(self) -> AsyncAnthropic:
        # Created lazily so that building an LLM (e.g. in tests or preflight) needs no key.
        if self._client is None:
            self._client = AsyncAnthropic()
        return self._client

    async def parse(
        self,
        *,
        model: str,
        effort: str,
        system: str,
        user: str,
        output_format: type[T],
        max_tokens: int = 16000,
    ) -> T:
        """Run one call and return the validated object, or raise LlmRefused / LlmTruncated."""
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_format": output_format,
            "output_config": {"effort": effort},
        }
        try:
            response = await self._send(kwargs)
        except ValidationError as exc:
            # The SDK validates the JSON before we can look at stop_reason, so a reply cut off
            # by max_tokens (or a refusal that stopped mid-output) surfaces here.
            raise LlmTruncated(
                f"{model} returned incomplete structured output ({exc.error_count()} errors)"
            ) from exc

        self._record_usage(response, model)
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            raise LlmRefused(_refusal_message(response, model))
        if stop == "max_tokens":
            raise LlmTruncated(f"{model} hit max_tokens ({max_tokens}) before finishing")
        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise LlmTruncated(f"{model} returned no structured output (stop_reason={stop})")
        return parsed

    async def _send(self, kwargs: dict[str, Any]) -> Any:
        if not self._fallbacks:
            return await self.client.messages.parse(**kwargs)
        try:
            return await self.client.beta.messages.parse(
                **kwargs, betas=[FALLBACK_BETA], fallbacks="default"
            )
        except anthropic.BadRequestError as exc:
            if not _is_fallback_rejection(exc):
                raise
            log.warning("Refusal fallbacks rejected by the API; continuing without them: %s", exc)
            self._fallbacks = False
            return await self.client.messages.parse(**kwargs)

    def _record_usage(self, response: Any, requested_model: str) -> None:
        served = getattr(response, "model", None) or requested_model
        entry = self.usage.setdefault(served, Usage(model=served))
        entry.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        entry.input_tokens += getattr(usage, "input_tokens", 0) or 0
        entry.output_tokens += getattr(usage, "output_tokens", 0) or 0
        entry.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    def cost_usd(self) -> float | None:
        total = 0.0
        for model, u in self.usage.items():
            price = _price_for(model)
            if price is None:
                return None
            p_in, p_out = price
            total += (
                u.input_tokens * p_in
                + u.cache_read_tokens * p_in * CACHE_READ_FACTOR
                + u.output_tokens * p_out
            ) / 1_000_000
        return total


def _refusal_message(response: Any, model: str) -> str:
    details = getattr(response, "stop_details", None)
    category = getattr(details, "category", None) if details is not None else None
    served = getattr(response, "model", None) or model
    if category:
        return f"{served} declined the request (category: {category})"
    return f"{served} declined the request"
