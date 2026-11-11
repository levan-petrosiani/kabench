"""One chat client for Ollama, LM Studio and vLLM (all OpenAI-compatible).

When Langfuse keys are set, the Langfuse drop-in OpenAI wrapper is used, so every
call is traced as a generation with token usage and latency.
"""

import os
import re
import time
from dataclasses import dataclass, field
from functools import cache

from .config import ModelSpec, langfuse_enabled
from .scorers import georgian_ratio

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class ChatResult:
    text: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int | None = None  # only some servers report it
    latency_s: float = 0.0
    tool_calls: list = field(default_factory=list)
    message: dict | None = None  # raw assistant message, for agent loops
    error: str | None = None

    @property
    def tokens_per_s(self) -> float:
        return self.completion_tokens / self.latency_s if self.latency_s else 0.0

    def to_output(self, **extra) -> dict:
        """JSON-serializable task output, as stored in Langfuse and results/runs."""
        # The reasoning text and its language show whether a model thinks in Georgian or English
        return {"text": self.text, "reasoning": self.reasoning or None, "reasoning_chars": len(self.reasoning),
                "reasoning_ka_ratio": round(georgian_ratio(self.reasoning), 3) if self.reasoning else None,
                "reasoning_tokens": self.reasoning_tokens,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "latency_s": round(self.latency_s, 3), "tokens_per_s": round(self.tokens_per_s, 1),
                "error": self.error, **extra}


@cache
def _client(base_url: str, api_key: str):
    if langfuse_enabled():
        from langfuse.openai import OpenAI
    else:
        from openai import OpenAI
    # Slow thinking models (27B at ~10 tok/s, 4k-token budget) need KABENCH_TIMEOUT raised
    return OpenAI(base_url=base_url, api_key=api_key, timeout=float(os.getenv("KABENCH_TIMEOUT", 300)), max_retries=0)


def chat(spec: ModelSpec, messages: list[dict], tools: list[dict] | None = None,
         name: str = "chat", **overrides) -> ChatResult:
    kwargs = dict(
        model=spec.model_id,
        messages=messages,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        extra_body=spec.extra_body or None,
    )
    if tools:
        kwargs["tools"] = tools
    if langfuse_enabled():
        kwargs["name"] = name
    kwargs.update(overrides)

    start = time.perf_counter()
    try:
        resp = _client(spec.base_url, spec.api_key).chat.completions.create(**kwargs)
    except Exception as e:  # server down, model not loaded, tools unsupported...
        return ChatResult(text="", error=f"{type(e).__name__}: {e}", latency_s=time.perf_counter() - start)
    latency = time.perf_counter() - start

    msg = resp.choices[0].message
    text = msg.content or ""
    extra = msg.model_extra or {}
    reasoning = extra.get("reasoning") or extra.get("reasoning_content") or ""
    # Some servers inline the reasoning instead of returning it separately
    if m := THINK_RE.search(text):
        reasoning = reasoning or m.group(1).strip()
        text = THINK_RE.sub("", text)

    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return ChatResult(
        text=text.strip(),
        reasoning=reasoning.strip(),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
        latency_s=latency,
        tool_calls=list(msg.tool_calls or []),
        message=msg.model_dump(exclude_none=True),
    )


def warm_up(spec: ModelSpec) -> str | None:
    """Load the model into memory so load time doesn't skew the first latency."""
    r = chat(spec, [{"role": "user", "content": "hi"}], name="warm-up", max_tokens=1)
    return r.error
