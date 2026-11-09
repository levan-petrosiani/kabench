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


def run_comprehension(spec: ModelSpec, lang: str, n: int = 100) -> list[dict]:
    def task(item) -> dict:
        prompt = item["input"] if isinstance(item, dict) else item.input
        return chat(spec, [{"role": "user", "content": prompt}], name=f"belebele-{lang}").to_output()

    return run(track="comprehension", spec=spec, variant="belebele", lang=lang,
               dataset=f"kabench-belebele-{lang}", items=build_items(lang, n), task=task, scorer=score)
