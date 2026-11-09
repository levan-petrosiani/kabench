from typing import Annotated

import typer

from .client import warm_up
from .config import get_models, langfuse_enabled, load_models

app = typer.Typer(help="Georgian vs English benchmark for local LLMs.", no_args_is_help=True)

Models = Annotated[str, typer.Option("--models", "-m", help="Comma-separated names from configs/models.yaml")]
Langs = Annotated[str, typer.Option("--lang", "-l", help="Comma-separated: ka,en")]
Limit = Annotated[int | None, typer.Option(help="Only run the first N items (smoke tests)")]


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _prepare(models: str):
    print(f"Langfuse tracing: {'on' if langfuse_enabled() else 'off (set LANGFUSE_* in .env)'}")
    specs = get_models(_split(models))
    for spec in specs:
        if err := warm_up(spec):
            raise SystemExit(f"Cannot reach {spec.name} at {spec.base_url}: {err}")
    return specs


@app.command("models")
def list_models():
    """List configured models."""
    for name, m in load_models().items():
        print(f"{name:24} {m.provider:9} {m.model_id:28} {m.base_url}")


@app.command()
def discover(urls: Annotated[str, typer.Option(help="Comma-separated OpenAI-compatible base URLs")] =
             "http://localhost:1234/v1,http://localhost:11434/v1"):
    """List models on running servers and print YAML entries for the ones not configured yet."""
    import re

    from openai import OpenAI

    configured = {(m.base_url, m.model_id) for m in load_models().values()}
    providers = {"1234": "lmstudio", "11434": "ollama", "8000": "vllm"}
    for url in _split(urls):
        try:
            ids = sorted(m.id for m in OpenAI(base_url=url, api_key="x", timeout=5).models.list())
        except Exception as e:
            print(f"# {url}: not reachable ({type(e).__name__})")
            continue
        provider = next((p for port, p in providers.items() if f":{port}" in url), "openai")
        skip = ("embed", "rerank", "bge")
        new = [i for i in ids if (url, i) not in configured and not any(s in i.lower() for s in skip)]
        print(f"# {url} ({provider}): {len(ids)} models, {len(new)} not in configs/models.yaml")
        for model_id in new:
            name = re.sub(r"[^a-z0-9.]+", "-", model_id.split("/")[-1].lower()).strip("-")
            print(f"  {name}-{provider}:\n    provider: {provider}\n    base_url: {url}\n    model_id: {model_id}\n")


@app.command()
def grammar(models: Models, lang: Langs = "ka,en", variants: Annotated[str, typer.Option(
        help="Prompt variants (prompts/grammar/<lang>/<variant>.md); 'all' = every variant")] = "all",
        limit: Limit = None,
        sample: Annotated[int | None, typer.Option(help="Fixed random subset of N items (same for every model)")] = None):
    """Track A: grammatical error correction."""
    from .tracks.grammar import VARIANTS, run_grammar

    for spec in _prepare(models):
        for lg in _split(lang):
            for variant in VARIANTS[lg] if variants == "all" else _split(variants):
                print(f"[grammar] {spec.name} / {lg} / {variant}")
                run_grammar(spec, lg, variant, limit, sample)


@app.command()
def comprehension(models: Models, lang: Langs = "ka,en",
                  n: Annotated[int, typer.Option(help="Belebele questions per language")] = 100):
    """Track B: Belebele reading comprehension (same questions in both languages)."""
    from .tracks.comprehension import run_comprehension

    for spec in _prepare(models):
        for lg in _split(lang):
            print(f"[comprehension] {spec.name} / {lg}")
            run_comprehension(spec, lg, n)


@app.command()
def tokens(models: Annotated[str, typer.Option("--models", "-m", help="Also measure live via these served models")] = "",
           n: Annotated[int, typer.Option(help="Passages for the live measurement")] = 60,
           live_only: Annotated[bool, typer.Option(help="Skip the offline tokenizer pass")] = False):
    """Track C: Georgian vs English token cost (offline HF tokenizers, optionally live)."""
    from .config import load_tokenizers
    from .tracks.tokens import run_offline, run_live

    if not live_only:
        print("[tokens] offline tokenizers")
        run_offline(load_tokenizers())
    if models:
        print("[tokens] live, via serving backends")
        run_live(_prepare(models), n)


@app.command()
def agent(models: Models, lang: Langs = "ka,en", limit: Limit = None):
    """Track D: agent behaviour with mock tools."""
    from .tracks.agent import run_agent

    for spec in _prepare(models):
        for lg in _split(lang):
            print(f"[agent] {spec.name} / {lg}")
            run_agent(spec, lg, limit)


@app.command("all")
def run_all(models: Models, n: Annotated[int, typer.Option(help="Belebele questions per language")] = 100):
    """Run every track for the given models, then build the report."""
    grammar(models)
    comprehension(models, n=n)
    agent(models)
    tokens(models)
    report()


@app.command()
def upload(models: Models):
    """Push saved runs of these models (made with Langfuse off) into Langfuse as dataset runs."""
    from .config import RUNS_DIR
    from .runner import upload as upload_run

    if not langfuse_enabled():
        raise SystemExit("Langfuse keys are not set (.env)")
    for name in _split(models):
        files = sorted(RUNS_DIR.glob(f"*/{name}__*.jsonl"))
        if not files:
            print(f"[upload] {name}: no saved runs")
        for f in files:
            print(f"[upload] {f.relative_to(RUNS_DIR)} -> {upload_run(f)}")


@app.command()
def review():
    """Write results/review.html: every test item with its expected answer and all model answers."""
    from .review import build_review

    print(f"Review page written to {build_review()}")


@app.command()
def report():
    """Aggregate results/runs into leaderboard tables and charts."""
    from .report import build_report

    build_report()


def main():
    """Entry point; exit code 3 tells scripts/run_queue.sh to skip the rest of a too-slow model."""
    from .runner import TooSlow

    try:
        app()
    except TooSlow as e:
        print(f"ABORTED: {e}")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
