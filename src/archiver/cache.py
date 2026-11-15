"""Per-video cache of paid stage results.

Each entry stores the fingerprint of what produced it (input, prompt, model). A lookup with a
different fingerprint is a miss, so changing a prompt or model never reuses stale output.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def fingerprint(*parts: object) -> str:
    """Stable short hash of the given parts. Order matters."""
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


class Cache:
    def __init__(self, root: Path, video_id: str, *, force: bool = False) -> None:
        self.dir = Path(root).expanduser() / video_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.force = force

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def load(self, name: str, fp: str, model: type[T]) -> T | None:
        """Return the cached object, or None on a miss, a fingerprint change, or --force."""
        if self.force:
            return None
        path = self.path(name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != fp:
                return None
            return model.model_validate(payload["data"])
        except (OSError, ValueError, KeyError, ValidationError):
            return None

    def save(self, name: str, fp: str, obj: BaseModel) -> None:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprint": fp, "data": obj.model_dump(mode="json")}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
