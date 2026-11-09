"""Belebele (FLORES-based reading comprehension), joined across Georgian and English.

Both languages contain the same 900 questions over the same 488 passages, which
makes it ideal for apples-to-apples Georgian vs English comparisons.
"""

import random
from functools import cache

LANG_CONFIG = {"ka": "kat_Geor", "en": "eng_Latn"}


@cache
def load_pairs() -> list[dict]:
    """Rows keyed by (link, question_number), with a `ka` and `en` version each."""
    from datasets import load_dataset

    by_lang = {
        lang: {(r["link"], r["question_number"]): r
               for r in load_dataset("facebook/belebele", cfg, split="test")}
        for lang, cfg in LANG_CONFIG.items()
    }
    keys = sorted(by_lang["ka"].keys() & by_lang["en"].keys())
    return [{"id": f"belebele-{i:03d}", "ka": by_lang["ka"][k], "en": by_lang["en"][k]}
            for i, k in enumerate(keys)]


def sample(n: int, seed: int = 0) -> list[dict]:
    """A seeded shuffle, then the first n: smaller samples are subsets of larger ones."""
    pairs = list(load_pairs())
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


def passages(lang_pairs: list[dict] | None = None) -> list[tuple[str, str]]:
    """Unique (ka, en) parallel passages."""
    seen, out = set(), []
    for p in lang_pairs or load_pairs():
        if p["en"]["flores_passage"] not in seen:
            seen.add(p["en"]["flores_passage"])
            out.append((p["ka"]["flores_passage"], p["en"]["flores_passage"]))
    return out
