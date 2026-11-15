"""Checks that a cleaned chunk still says what the speaker said.

Both texts are reduced to content words (fillers dropped, stutters collapsed) and aligned with
difflib. Three numbers come out:

- recall: share of source content words found in the cleaned text (catches summarizing, cuts);
- novelty: share of cleaned words not found in the source (catches rewording, translation);
- longest_gap: longest run of consecutive source words that vanished (catches a dropped passage
  that recall alone would forgive in a long chunk; 40 words is roughly 20 seconds of speech).

The same alignment maps a position in the cleaned text back to the segment it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

RECALL_MIN = 0.75
NOVELTY_MAX = 0.20
LONGEST_GAP_MAX = 40

# fmt: off
FILLERS_RU = {
    "ну", "вот", "как бы", "то есть", "короче", "типа", "значит", "э", "эм", "ээ", "эээ", "м",
    "мм", "ммм", "а-а", "ага", "угу", "ой", "ах", "эх", "хм", "так сказать", "в общем",
    "в общем-то", "собственно", "вообще-то", "это самое", "в принципе", "блин", "слушай", "смотри",
    "понимаешь", "знаешь", "так вот", "просто", "кстати говоря", "скажем так", "допустим", "да",
    "ладно", "окей", "так", "вообще", "получается", "скажем",
}
FILLERS_EN = {
    "um", "uh", "uhm", "umm", "erm", "er", "ah", "eh", "hmm", "mm", "mhm", "uh-huh", "like",
    "you know", "i mean", "sort of", "kind of", "basically", "actually", "literally", "really",
    "so", "well", "okay", "ok", "right", "yeah", "just", "anyway", "you see", "kinda", "sorta",
    "i guess", "or something", "and stuff", "whatever", "oh", "honestly", "totally",
}
# fmt: on

_WORD = re.compile(r"\w+(?:[-']\w+)*", re.UNICODE)


def _normalize(text: str) -> str:
    return text.casefold().replace("ё", "е")


def _build_fillers() -> tuple[set[str], list[tuple[str, ...]]]:
    single: set[str] = set()
    multi: list[tuple[str, ...]] = []
    for phrase in FILLERS_RU | FILLERS_EN:
        words = tuple(_WORD.findall(_normalize(phrase)))
        if len(words) == 1:
            single.add(words[0])
        elif words:
            multi.append(words)
    multi.sort(key=len, reverse=True)  # match the longest phrase first
    return single, multi


_SINGLE_FILLERS, _MULTI_FILLERS = _build_fillers()


@dataclass(frozen=True)
class Token:
    text: str
    ref: int  # segment id for source tokens, paragraph index for cleaned tokens


def tokenize(text: str) -> list[str]:
    """Lowercased words with punctuation removed (`ё` folded to `е`)."""
    return _WORD.findall(_normalize(text))


def content_tokens(pieces: list[tuple[int, str]]) -> list[Token]:
    """Content words of (ref, text) pieces: fillers dropped, immediate repeats collapsed."""
    raw = [Token(word, ref) for ref, text in pieces for word in tokenize(text)]
    kept = _drop_fillers(raw)
    return _collapse_repeats(kept)


def _drop_fillers(tokens: list[Token]) -> list[Token]:
    out: list[Token] = []
    i = 0
    while i < len(tokens):
        skip = _filler_len(tokens, i)
        if skip:
            i += skip
            continue
        out.append(tokens[i])
        i += 1
    return out


def _filler_len(tokens: list[Token], i: int) -> int:
    for phrase in _MULTI_FILLERS:
        n = len(phrase)
        if tuple(t.text for t in tokens[i : i + n]) == phrase:
            return n
    return 1 if tokens[i].text in _SINGLE_FILLERS else 0


def _collapse_repeats(tokens: list[Token]) -> list[Token]:
    out: list[Token] = []
    for tok in tokens:
        if out and out[-1].text == tok.text:
            continue
        out.append(tok)
    return out


@dataclass
class FidelityReport:
    recall: float
    novelty: float
    longest_gap: int
    ok: bool
    reason: str = ""


@dataclass
class Alignment:
    source: list[Token]
    cleaned: list[Token]
    # For each cleaned token, the index of the source token it matched, or None.
    cleaned_to_source: list[int | None] = field(default_factory=list)
    source_matched: list[bool] = field(default_factory=list)

    @property
    def recall(self) -> float:
        if not self.source:
            return 1.0
        return sum(self.source_matched) / len(self.source)

    @property
    def novelty(self) -> float:
        if not self.cleaned:
            return 0.0
        unmatched = sum(1 for m in self.cleaned_to_source if m is None)
        return unmatched / len(self.cleaned)

    @property
    def longest_gap(self) -> int:
        best = run = 0
        for matched in self.source_matched:
            run = 0 if matched else run + 1
            best = max(best, run)
        return best

    def report(self) -> FidelityReport:
        recall, novelty, gap = self.recall, self.novelty, self.longest_gap
        problems: list[str] = []
        if recall < RECALL_MIN:
            problems.append(f"recall {recall:.2f} < {RECALL_MIN} (source words are missing)")
        if novelty > NOVELTY_MAX:
            problems.append(f"novelty {novelty:.2f} > {NOVELTY_MAX} (words not in the source)")
        if gap > LONGEST_GAP_MAX:
            problems.append(f"a run of {gap} source words vanished (limit {LONGEST_GAP_MAX})")
        return FidelityReport(recall, novelty, gap, ok=not problems, reason="; ".join(problems))

    def source_ref_for(self, cleaned_ref: int) -> int | None:
        """The source ref (segment id) where cleaned piece `cleaned_ref` begins.

        Uses the first matched token of that piece; if none of its words matched, the next
        matched token after it; failing that, the last matched token before it.
        """
        first = next((i for i, t in enumerate(self.cleaned) if t.ref == cleaned_ref), None)
        if first is None:
            return None
        for i in range(first, len(self.cleaned)):
            src = self.cleaned_to_source[i]
            if src is not None:
                return self.source[src].ref
        for i in range(first - 1, -1, -1):
            src = self.cleaned_to_source[i]
            if src is not None:
                return self.source[src].ref
        return None


def align(source: list[tuple[int, str]], cleaned: list[tuple[int, str]]) -> Alignment:
    """Align (segment id, text) source pieces to (paragraph index, text) cleaned pieces."""
    src = content_tokens(source)
    out = content_tokens(cleaned)
    matcher = SequenceMatcher(None, [t.text for t in src], [t.text for t in out], autojunk=False)
    cleaned_to_source: list[int | None] = [None] * len(out)
    source_matched = [False] * len(src)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            cleaned_to_source[block.b + k] = block.a + k
            source_matched[block.a + k] = True
    return Alignment(src, out, cleaned_to_source, source_matched)


def check(source: list[tuple[int, str]], cleaned: list[tuple[int, str]]) -> FidelityReport:
    return align(source, cleaned).report()
