"""Deterministic scorers shared by the tracks. All return plain floats/bools."""

import re
import unicodedata

from rapidfuzz.distance import Levenshtein

# Mkhedruli (modern Georgian) + Mtavruli (capitals)
GEORGIAN_RE = re.compile(r"[ა-ჿᲐ-Ჿ]")
LETTER_RE = re.compile(r"\w", re.UNICODE)
WRAPPER_RE = re.compile(r"^(output|corrected|გასწორებული)\s*[:：]\s*", re.IGNORECASE)
QUOTES = "\"'“”„«»`*"


def normalize(text: str) -> str:
    """Normalize for comparison: NFC, collapse whitespace, strip wrapping quotes."""
    text = unicodedata.normalize("NFC", text or "").strip()
    text = WRAPPER_RE.sub("", text).strip().strip(QUOTES).strip()
    return re.sub(r"\s+", " ", text)


def first_line(text: str) -> str:
    return next((line for line in (text or "").splitlines() if line.strip()), "")


def exact_match(output: str, expected: str) -> bool:
    return normalize(output) == normalize(expected)


def char_similarity(output: str, expected: str) -> float:
    """1 - normalized Levenshtein distance, in [0, 1]."""
    return Levenshtein.normalized_similarity(normalize(output), normalize(expected))


def edit_scores(source: str, output: str, expected: str) -> dict:
    """GEC-style view of a single correction.

    - over_correction: sentence was already correct but the model changed it
    - missed: sentence needed a fix but the model returned it unchanged
    - gain: how much closer to the gold the output is than the input
      (1 = perfect fix, 0 = no progress, negative = made it worse)
    """
    src, out, exp = normalize(source), normalize(output), normalize(expected)
    needs_fix = src != exp
    changed = out != src
    d_src = Levenshtein.distance(src, exp)
    d_out = Levenshtein.distance(out, exp)
    return {
        "needs_fix": needs_fix,
        "over_correction": (not needs_fix) and changed,
        "missed": needs_fix and not changed,
        "gain": (d_src - d_out) / d_src if needs_fix else (1.0 if not changed else -1.0),
    }


def format_ok(raw_output: str, source: str) -> bool:
    """Did the model return just the sentence (no explanation, labels, markdown)?"""
    raw = (raw_output or "").strip()
    if not raw or len(raw.splitlines()) > 1:
        return False
    if WRAPPER_RE.match(raw) or "**" in raw:
        return False
    return len(raw) <= 2 * len(source) + 10


def georgian_ratio(text: str) -> float:
    """Share of letters that are Georgian script (0 = none, 1 = all)."""
    letters = LETTER_RE.findall(text or "")
    letters = [c for c in letters if not c.isdigit() and c != "_"]
    if not letters:
        return 0.0
    return sum(bool(GEORGIAN_RE.match(c)) for c in letters) / len(letters)


def language_match(text: str, lang: str) -> bool:
    ratio = georgian_ratio(text)
    return ratio >= 0.5 if lang == "ka" else ratio < 0.05


def parse_choice(text: str, n_options: int = 4) -> int | None:
    """Extract a multiple-choice answer number (1..n) from a model reply."""
    m = re.search(rf"\b([1-{n_options}])\b", normalize(text))
    return int(m.group(1)) if m else None
