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
from .agent_world import SIDE_EFFECT_TOOLS, SYSTEM_PROMPT, TOOLS, TOOLS_LONG, World, WorldLong, parse_args

MAX_STEPS = 8  # per user turn
# A clarifying reply: a question mark, or a request phrased without one ("გთხოვთ, დამიდასტურეთ" = "please confirm")
ASKS = {"en": re.compile(r"confirm|which (one|of)|could you (please )?(specify|clarify|tell)|please (specify|clarify|provide)", re.I),
        "ka": re.compile(r"დაადასტურ|დამიდასტურ|დადასტურ|რომელ|დააზუსტ|მიუთითე")}


def asks_user(reply: str, lang: str) -> bool:
    return "?" in reply or bool(ASKS[lang].search(reply))
# v1: 15 short scenarios (1-3 steps). long: multi-turn scenarios whose tools return Georgian-heavy content.
SUITES = {"v1": ("agent_scenarios.jsonl", TOOLS, World, "tools-v1", "kabench-agent"),
          "long": ("agent_long_scenarios.jsonl", TOOLS_LONG, WorldLong, "tools-long-v1", "kabench-agent-long")}
TEXT_CALL_RE = re.compile(r"^\s*(?:<tool_call>|```(?:json)?)?\s*(.*?)\s*(?:</tool_call>|```)?\s*$", re.DOTALL)


def load_items(lang: str, limit: int | None = None, suite: str = "v1") -> list[dict]:
    rows = [json.loads(line) for line in (DATA_DIR / SUITES[suite][0]).read_text().splitlines()]
    # A scenario is one user message ("user") or a scripted conversation ("turns")
    return [{"id": r["id"], "input": r["user"][lang] if "user" in r else r["turns"][lang], "expected_output": r["expect"],
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


def run_episode(spec: ModelSpec, user_msgs: str | list[str], lang: str, suite: str = "v1") -> dict:
    """Run one scenario. Multi-turn scenarios send the scripted user messages one after another; each turn runs
    until the model replies without a tool call. Prompt tokens add up over every step, so they measure how much
    context the conversation re-reads."""
    _, tools, world_cls, _, _ = SUITES[suite]
    world = world_cls(lang)
    turns = [user_msgs] if isinstance(user_msgs, str) else user_msgs
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    calls, replies, invalid, text_calls, prompt_tokens, completion_tokens = [], [], 0, 0, 0, 0
    error, stuck, steps = None, None, 0
    start = time.perf_counter()

    for turn, user_msg in enumerate(turns):
        messages.append({"role": "user", "content": user_msg})
        for _ in range(MAX_STEPS):
            steps += 1
            r = chat(spec, messages, tools=tools, name=f"agent-step-{steps}")
            if r.error and "error parsing tool call" in r.error:  # the server rejected a malformed tool call
                invalid += 1
                stuck = r.error
                break
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
                replies.append(r.text)
                messages.append({"role": "assistant", "content": r.text})
                break

            messages.append({"role": "assistant", "content": "" if from_text else r.text, "tool_calls": tool_calls})
            for tc in tool_calls:
                name, args = tc["function"]["name"], parse_args(tc["function"]["arguments"])
                if args is None or name not in world.tool_names:
                    invalid += 1
                    result = {"error": "invalid tool call"}
                else:
                    with _tool_span(name, args) as span:
                        result = world.call(name, args)
                        if span:
                            span.update(output=result)
                calls.append({"tool": name, "args": args, "result": result, "turn": turn})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(result, ensure_ascii=False)})
        else:
            stuck = f"no reply after {MAX_STEPS} steps in turn {turn + 1}"
        if error or stuck:
            break

    out = {"text": replies[-1] if len(replies) == len(turns) else "", "calls": calls, "steps": steps,
           "invalid_calls": invalid, "text_tool_calls": text_calls, "prompt_tokens": prompt_tokens,
           "completion_tokens": completion_tokens, "latency_s": round(time.perf_counter() - start, 3), "error": error}
    if stuck:  # model behaviour, not an infrastructure error: scored as a failed scenario
        out["stuck"] = stuck
    if len(turns) > 1:
        out["replies"] = replies
    return out


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
        # "turn": the call must happen while answering that user message (multi-turn scenarios)
        candidates = [c for c in free if c["tool"] == exp["tool"] and c.get("turn", 0) == exp.get("turn", c.get("turn", 0))]
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
        checks["clarify_ok"] = asks_user(final, lang)
    replies = output.get("replies") or [final]
    if (k := exp.get("clarify_turn")) is not None:  # must ask (and not act) after user message k
        checks["clarify_ok"] = len(replies) > k and asks_user(replies[k], lang)
    if (k := exp.get("no_side_effects_turn")) is not None:
        checks["no_side_effects_ok"] = not any(c["tool"] in SIDE_EFFECT_TOOLS for c in calls if c.get("turn", 0) == k)
    if not_calls := exp.get("not_calls"):  # specific calls that must never happen, e.g. replying to the wrong person
        checks["not_calls_ok"] = not any(
            c["tool"] == n["tool"] and c["args"] is not None
            and all(_arg_ok(c["args"].get(key), v, lang) for key, v in n.get("args", {}).items())
            for n in not_calls for c in calls)
    if contains := exp.get("final_contains"):
        checks["answer_ok"] = any(s.lower() in final.lower() for s in contains[lang])

    if output.get("stuck"):
        checks["replied_ok"] = False
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


def run_agent(spec: ModelSpec, lang: str, limit: int | None = None, suite: str = "v1") -> list[dict]:
    _, _, _, variant, dataset = SUITES[suite]

    def task(item) -> dict:
        user_msgs = item["input"] if isinstance(item, dict) else item.input
        return run_episode(spec, user_msgs, lang, suite)

    return run(track="agent", spec=spec, variant=variant, lang=lang,
               dataset=f"{dataset}-{lang}", items=load_items(lang, limit, suite), task=task, scorer=score)
