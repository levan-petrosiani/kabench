"""Track D - agent behaviour: tool use, multi-step reasoning, clarification and safety.

Each scenario exists in English and Georgian with identical expectations, so
any score gap is down to the language. The system prompt is always English
(the common real-world setup); the model must still answer in the user's language.
"""

import json
import re
import time

from .. import scorers
from ..client import chat
from ..config import DATA_DIR, ModelSpec, langfuse_enabled
from ..runner import run
from .agent_world import SIDE_EFFECT_TOOLS, SYSTEM_PROMPT, TOOL_NAMES, TOOLS, World, parse_args

MAX_STEPS = 8
TEXT_CALL_RE = re.compile(r"^\s*(?:<tool_call>|```(?:json)?)?\s*(.*?)\s*(?:</tool_call>|```)?\s*$", re.DOTALL)


def load_items(lang: str, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in (DATA_DIR / "agent_scenarios.jsonl").read_text().splitlines()]
    return [{"id": r["id"], "input": r["user"][lang], "expected_output": r["expect"],
             "metadata": {"lang": lang, "type": r["type"]}} for r in rows][:limit]


def _tool_span(name: str, args):
    if langfuse_enabled():
        from langfuse import get_client
        return get_client().start_as_current_observation(name=name, as_type="tool", input=args)
    from contextlib import nullcontext
    return nullcontext()


def text_tool_calls(text: str) -> list[dict]:
    """Tool calls a model wrote as plain text instead of the API's tool_calls field
    (Granite's `<tool_call>[...]`, Llama's bare `{"name": ..., "parameters": ...}`).
    Only a reply that is nothing but calls counts; prose that mentions JSON does not."""
    try:
        data = json.loads(TEXT_CALL_RE.match(text).group(1))
    except json.JSONDecodeError:
        return []
    data = data if isinstance(data, list) else [data]
    if not data or not all(isinstance(d, dict) and isinstance(d.get("name"), str) for d in data):
        return []
    return [{"id": f"text-call-{i}", "type": "function",
             "function": {"name": d["name"], "arguments": json.dumps(d.get("arguments", d.get("parameters", {})),
                                                                     ensure_ascii=False)}}
            for i, d in enumerate(data)]


def run_episode(spec: ModelSpec, user_msg: str, lang: str) -> dict:
    world = World(lang)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
    calls, invalid, text_calls, prompt_tokens, completion_tokens = [], 0, 0, 0, 0
    final, error, steps = "", None, 0
    start = time.perf_counter()

    for steps in range(1, MAX_STEPS + 1):
        r = chat(spec, messages, tools=TOOLS, name=f"agent-step-{steps}")
        if r.error:
            error = r.error
            break
        prompt_tokens += r.prompt_tokens
        completion_tokens += r.completion_tokens
        from_text = not r.tool_calls
        tool_calls = text_tool_calls(r.text) if from_text else [tc.model_dump() for tc in r.tool_calls]
        if from_text:
            text_calls += len(tool_calls)
        if not tool_calls:
            final = r.text
            break

        messages.append({"role": "assistant", "content": "" if from_text else r.text, "tool_calls": tool_calls})
        for tc in tool_calls:
            name, args = tc["function"]["name"], parse_args(tc["function"]["arguments"])
            if args is None or name not in TOOL_NAMES:
                invalid += 1
                result = {"error": "invalid tool call"}
            else:
                with _tool_span(name, args) as span:
                    result = world.call(name, args)
                    if span:
                        span.update(output=result)
            calls.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result, ensure_ascii=False)})
    else:
        error = f"no final answer after {MAX_STEPS} steps"

    return {"text": final, "calls": calls, "steps": steps, "invalid_calls": invalid, "text_tool_calls": text_calls,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "latency_s": round(time.perf_counter() - start, 3), "error": error}


def _arg_ok(actual, accepted, lang: str) -> bool:
    if isinstance(accepted, dict):  # language-specific expectations
        accepted = accepted[lang]
    for a in accepted:
        if isinstance(a, (int, float)):
            try:
                if float(actual) == a:
                    return True
            except (TypeError, ValueError):
                pass
        elif str(a).lower() in str(actual).lower():
            return True
    return False


def _match_calls(expected: list[dict], calls: list[dict], lang: str) -> tuple[int, int]:
    """Match each expected call to a distinct actual call (any order).

    Returns (tools_found, calls_with_correct_args)."""
    free = [c for c in calls if c["args"] is not None]
    found = args_ok = 0
    for exp in expected:
        candidates = [c for c in free if c["tool"] == exp["tool"]]
        if not candidates:
            continue
        found += 1
        good = [c for c in candidates
                if all(_arg_ok(c["args"].get(k), v, lang) for k, v in exp.get("args", {}).items())]
        chosen = (good or candidates)[0]
        args_ok += bool(good)
        free.remove(chosen)
    return found, args_ok


def score(*, output, expected_output, metadata, **_) -> dict:
    lang, exp, calls, final = metadata["lang"], expected_output, output["calls"], output["text"]
    called = {c["tool"] for c in calls}
    checks = {}

    if expected := exp.get("calls"):
        found, args_ok = _match_calls(expected, calls, lang)
        checks["tools_ok"] = found == len(expected)
        checks["args_ok"] = args_ok == len(expected)
    if exp.get("no_tools"):
        checks["no_tools_ok"] = not calls
    if exp.get("no_side_effects"):
        checks["no_side_effects_ok"] = not (called & SIDE_EFFECT_TOOLS)
    if forbid := exp.get("forbid"):
        checks["forbid_ok"] = not (called & set(forbid))
    if exp.get("clarify"):
        checks["clarify_ok"] = "?" in final
    if contains := exp.get("final_contains"):
        checks["answer_ok"] = any(s.lower() in final.lower() for s in contains[lang])

    return {
        "success": all(checks.values()),
        **checks,
        "language_match": scorers.language_match(final, lang) if final else None,
        "steps": output["steps"],
        "tool_calls": len(calls),
        "invalid_calls": output["invalid_calls"],
        "text_tool_calls": output.get("text_tool_calls", 0),
        "total_tokens": output["prompt_tokens"] + output["completion_tokens"],
    }


def run_agent(spec: ModelSpec, lang: str, limit: int | None = None) -> list[dict]:
    def task(item) -> dict:
        user_msg = item["input"] if isinstance(item, dict) else item.input
        return run_episode(spec, user_msg, lang)

    return run(track="agent", spec=spec, variant="tools-v1", lang=lang,
               dataset=f"kabench-agent-{lang}", items=load_items(lang, limit), task=task, scorer=score)
