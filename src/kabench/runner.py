"""Runs a track over a dataset for one model and persists per-item rows.

With Langfuse keys set, items are synced to a Langfuse Dataset and executed via
`Langfuse.run_experiment`, so each model/variant shows up as a comparable
dataset run with item-level scores. Without keys the same task + scorer run in
a plain local loop. Either way rows land in results/runs/<track>/<run>.jsonl.
"""

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from rich.progress import track as progress

from .config import RUNS_DIR, ModelSpec, langfuse_enabled

Item = dict  # {"id", "input", "expected_output", "metadata"}
Scorer = Callable[..., dict[str, float | bool]]


class TooSlow(Exception):
    """Raised when a model is clearly not running properly (e.g. spilled to CPU/swap)."""


class SpeedGuard:
    """Trips after `strikes` consecutive slow items: decoding under `min_tps`, or a
    short answer (< 20 tokens) taking longer than `max_latency_s`. Long reasoning
    answers are judged by decode speed only. Once tripped, remaining items are skipped."""

    def __init__(self, min_tps: float = 3.0, max_latency_s: float = 300.0, strikes: int = 5):
        self.min_tps, self.max_latency_s, self.limit = min_tps, max_latency_s, strikes
        self.strikes, self.tripped = 0, False

    def check(self, output: dict) -> None:
        if err := output.get("error"):
            # A dead or hung server shows up as timeouts/connection errors: count them,
            # but not model-behaviour errors (e.g. an agent that never gives a final answer)
            if any(k in err for k in ("Timeout", "Connection", "Error code: 5")):
                self.strikes += 1
                self.tripped = self.strikes >= self.limit
            return
        latency, tokens = output.get("latency_s") or 0, output.get("completion_tokens") or 0
        if tokens >= 20:
            slow = tokens / latency < self.min_tps
        else:
            slow = latency > self.max_latency_s
        self.strikes = self.strikes + 1 if slow else 0
        self.tripped = self.strikes >= self.limit


def _get(item: Any, key: str):
    return item.get(key) if isinstance(item, dict) else getattr(item, key)


def sync_dataset(name: str, items: list[Item], description: str = "") -> None:
    """Upsert items into a Langfuse dataset (item ids are stable, so reruns are idempotent)."""
    from langfuse import get_client

    lf = get_client()
    try:
        lf.get_dataset(name)
    except Exception:
        lf.create_dataset(name=name, description=description)
    for it in items:
        lf.create_dataset_item(dataset_name=name, id=f"{name}/{it['id']}", input=it["input"],
                               expected_output=it["expected_output"], metadata=it["metadata"])


def run(*, track: str, spec: ModelSpec, variant: str, lang: str, dataset: str,
        items: list[Item], task: Callable[[Item], dict], scorer: Scorer) -> list[dict]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"{spec.name}:{lang}:{variant}:{stamp}"
    meta = {"model": spec.name, "model_id": spec.model_id, "provider": spec.provider,
            "variant": variant, "lang": lang, "track": track}

    guard = SpeedGuard()

    def guarded_task(item) -> dict:
        if guard.tripped:
            return {"error": "skipped: model too slow"}
        output = task(item)
        guard.check(output)
        return output

    if langfuse_enabled():
        results = _run_langfuse(dataset, items, guarded_task, scorer, run_name, meta)
    else:
        results = _run_local(items, guarded_task, scorer, run_name)
    if guard.tripped:
        raise TooSlow(f"{spec.name} decoded < {guard.min_tps} tok/s (or > {guard.max_latency_s:.0f}s/item) "
                      f"{guard.limit} times in a row; partial run discarded")

    rows = [{"run": run_name, **meta, "item_id": item_id, "metadata": metadata,
             "input": inp, "expected": expected, "output": output, "scores": scores}
            for item_id, inp, expected, metadata, output, scores in results]
    out = RUNS_DIR / track / f"{run_name.replace(':', '__')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {len(rows)} rows written to {out.relative_to(RUNS_DIR.parent.parent)}")
    return rows


def _score(scorer: Scorer, item, output) -> dict:
    if output.get("error"):
        return {}
    return scorer(input=_get(item, "input"), output=output,
                  expected_output=_get(item, "expected_output"), metadata=_get(item, "metadata"))


def _run_local(items, task, scorer, run_name):
    out = []
    for item in progress(items, description=run_name):
        output = task(item)
        out.append((item["id"], item["input"], item["expected_output"], item["metadata"],
                    output, _score(scorer, item, output)))
    return out


def _run_langfuse(dataset, items, task, scorer, run_name, meta):
    from langfuse import Evaluation, get_client

    lf = get_client()
    sync_dataset(dataset, items)
    wanted = {f"{dataset}/{it['id']}" for it in items}
    ds_items = [it for it in lf.get_dataset(dataset).items if it.id in wanted]

    def evaluator(*, input, output, expected_output, metadata, **_):
        if output.get("error"):
            return []
        scores = scorer(input=input, output=output, expected_output=expected_output, metadata=metadata)
        return [Evaluation(name=k, value=float(v)) for k, v in scores.items() if v is not None]

    result = lf.run_experiment(
        name=dataset, run_name=run_name, data=ds_items,
        task=lambda *, item, **_: task(item),
        evaluators=[evaluator], max_concurrency=1, metadata={k: str(v) for k, v in meta.items()},
    )
    lf.flush()
    if result.dataset_run_url:
        print(f"  -> Langfuse: {result.dataset_run_url}")

    out = []
    for r in result.item_results:
        it = r.item
        output = r.output if isinstance(r.output, dict) else {"error": str(r.output)}
        scores = {e.name: e.value for e in r.evaluations}
        out.append((it.id.split("/")[-1], it.input, it.expected_output, it.metadata, output, scores))
    return out


DATASET_PREFIX = {"grammar": "kabench-grammar", "comprehension": "kabench-belebele", "agent": "kabench-agent"}


def upload(path) -> str:
    """Replay a saved run file into Langfuse as a dataset run, for runs made with Langfuse off
    (e.g. big models run with the Docker stack stopped to free RAM). Outputs and scores are the
    stored ones; trace timings are not real, the measured latency stays in each output."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    first = rows[0]
    items = [{"id": r["item_id"], "input": r["input"], "expected_output": r["expected"], "metadata": r["metadata"]}
             for r in rows]
    stored = {json.dumps(r["input"], ensure_ascii=False): r for r in rows}
    meta = {k: first[k] for k in ("model", "model_id", "provider", "variant", "lang", "track")} | {"replayed": True}

    def task(item) -> dict:
        return stored[json.dumps(_get(item, "input"), ensure_ascii=False)]["output"]

    def scorer(*, input, **_) -> dict:
        return stored[json.dumps(input, ensure_ascii=False)]["scores"]

    prefix = DATASET_PREFIX[first["track"]]
    if first["variant"].startswith("tools-long"):  # the long agent suite has its own dataset, as in run_agent
        prefix = "kabench-agent-long"
    _run_langfuse(f"{prefix}-{first['lang']}", items, task, scorer, first["run"], meta)
    return first["run"]
