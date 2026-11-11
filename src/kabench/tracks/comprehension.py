"""Track B - reading comprehension on Belebele (same questions in Georgian and English)."""

from .. import scorers
from ..belebele import sample
from ..client import chat
from ..config import ModelSpec
from ..runner import run

# Instructions are in the language being tested.
PROMPTS = {
    "en": ("Read the passage and answer the question. "
           "Reply with only the number of the correct answer (1, 2, 3 or 4).\n\n"
           "Passage: {passage}\n\nQuestion: {question}\n\n{options}\n\nAnswer:"),
    "ka": ("წაიკითხე ტექსტი და უპასუხე კითხვას. "
           "უპასუხე მხოლოდ სწორი პასუხის ნომრით (1, 2, 3 ან 4).\n\n"
           "ტექსტი: {passage}\n\nკითხვა: {question}\n\n{options}\n\nპასუხი:"),
}


# Modes for the English-core experiments (.ideas/NEXT-STEPS.md). Same items and scoring as `direct`.
MODES = ("direct", "no-copy", "translate")
NO_COPY = ("Think in English. Do not copy, quote or translate the passage, the question or the options in your "
           "reasoning: refer to them directly. Reply exactly as the user asks.")
TRANSLATE = ("Translate the following text into English. Keep the numbered options and the layout. "
             "Do not answer the question. Reply with the translation only.\n\n{text}")


def build_items(lang: str, n: int) -> list[dict]:
    items = []
    for pair in sample(n):
        r = pair[lang]
        options = "\n".join(f"{i}) {r[f'mc_answer{i}']}" for i in range(1, 5))
        items.append({
            "id": pair["id"],
            "input": PROMPTS[lang].format(passage=r["flores_passage"], question=r["question"], options=options),
            "expected_output": str(r["correct_answer_num"]),
            "metadata": {"lang": lang, "link": r["link"], "question_number": r["question_number"]},
        })
    return items


def score(*, output, expected_output, **_) -> dict:
    choice = scorers.parse_choice(output["text"])
    return {"accuracy": choice == int(expected_output), "parsed": choice is not None}


def _translate_then_answer(spec: ModelSpec, prompt: str, lang: str) -> dict:
    """Translate once with thinking off, then answer the English version with the model's usual settings."""
    no_think = {**(spec.extra_body or {}), "reasoning_effort": "none"}
    tr = chat(spec, [{"role": "user", "content": TRANSLATE.format(text=prompt)}], name=f"belebele-{lang}-translate",
              extra_body=no_think, max_tokens=2048)
    if tr.error:
        return tr.to_output()
    ans = chat(spec, [{"role": "user", "content": tr.text}], name=f"belebele-{lang}-answer")
    # Totals cover both calls; the translation's share is kept so the answer's own thinking can be separated
    out = ans.to_output(translation=tr.text, translation_prompt_tokens=tr.prompt_tokens,
                        translation_completion_tokens=tr.completion_tokens)
    out.update(prompt_tokens=tr.prompt_tokens + ans.prompt_tokens,
               completion_tokens=tr.completion_tokens + ans.completion_tokens,
               latency_s=round(tr.latency_s + ans.latency_s, 3))
    return out


def run_comprehension(spec: ModelSpec, lang: str, n: int = 100, mode: str = "direct") -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    def task(item) -> dict:
        prompt = item["input"] if isinstance(item, dict) else item.input
        if mode == "translate" and lang != "en":
            return _translate_then_answer(spec, prompt, lang)
        messages = [{"role": "user", "content": prompt}]
        if mode == "no-copy":
            messages.insert(0, {"role": "system", "content": NO_COPY})
        return chat(spec, messages, name=f"belebele-{lang}").to_output()

    variant = "belebele" if mode == "direct" else f"belebele-{mode}"
    return run(track="comprehension", spec=spec, variant=variant, lang=lang,
               dataset=f"kabench-belebele-{lang}", items=build_items(lang, n), task=task, scorer=score)
