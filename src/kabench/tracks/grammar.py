"""Track A - grammatical error correction (GEC) in Georgian and English."""

import json
import random

from .. import scorers
from ..client import chat
from ..config import DATA_DIR, PROMPTS_DIR, ModelSpec
from ..runner import run

VARIANTS = {
    "ka": ["zero_shot", "few_shot", "zero_shot_en_instructions"],
    "en": ["zero_shot", "few_shot"],
}


def load_items(lang: str, prompt: str, limit: int | None = None,
               sample: int | None = None) -> tuple[list[dict], int]:
    """Load the dataset, dropping items that appear verbatim in the prompt.

    Few-shot prompts that reuse test sentences as examples would otherwise
    inflate the score (test-set leakage). Returns (items, n_leaked).
    """
    rows = [json.loads(line) for line in (DATA_DIR / f"grammar_{lang}.jsonl").read_text().splitlines()]
    items, leaked = [], 0
    for r in rows:
        if f"Input: {r['input']}" in prompt:
            leaked += 1
            continue
        items.append({
            "id": r["id"],
            "input": r["input"],
            "expected_output": r["expected"],
            "metadata": {"category": r["category"], "lang": lang, "needs_review": bool(r.get("needs_review"))},
        })
    if sample:  # seeded shuffle prefix: nested across sizes, identical across models
        order = list(range(len(items)))
        random.Random(0).shuffle(order)
        keep = set(order[:sample])
        items = [it for i, it in enumerate(items) if i in keep]
    return items[:limit], leaked


def score(*, input, output, expected_output, **_) -> dict:
    corrected = scorers.normalize(scorers.first_line(output["text"]))
    s = scorers.edit_scores(input, corrected, expected_output)
    return {
        "exact_match": scorers.exact_match(corrected, expected_output),
        "char_similarity": scorers.char_similarity(corrected, expected_output),
        "gain": s["gain"],
        "over_correction": s["over_correction"] if not s["needs_fix"] else None,
        "missed": s["missed"] if s["needs_fix"] else None,
        "format_ok": scorers.format_ok(output["text"], input),
    }


def run_grammar(spec: ModelSpec, lang: str, variant: str, limit: int | None = None,
                sample: int | None = None) -> list[dict]:
    template = (PROMPTS_DIR / "grammar" / lang / f"{variant}.md").read_text(encoding="utf-8").strip()
    items, leaked = load_items(lang, template, limit, sample)
    if leaked:
        print(f"  ({leaked} items excluded: they appear as examples in the '{variant}' prompt)")

    def task(item) -> dict:
        text = item["input"] if isinstance(item, dict) else item.input
        return chat(spec, [{"role": "user", "content": template.replace("{text}", text)}],
                    name=f"grammar-{lang}").to_output()

    return run(track="grammar", spec=spec, variant=variant, lang=lang,
               dataset=f"kabench-grammar-{lang}", items=items, task=task, scorer=score)
