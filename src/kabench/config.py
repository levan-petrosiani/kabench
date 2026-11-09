import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"

load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))


class ModelSpec(BaseModel):
    name: str
    provider: str
    base_url: str
    model_id: str
    hf_tokenizer: str | None = None
    api_key: str = "not-needed"
    temperature: float = 0.0
    max_tokens: int = 1024
    extra_body: dict = {}


def _load_yaml() -> dict:
    return yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())


def load_models() -> dict[str, ModelSpec]:
    return {name: ModelSpec(name=name, **cfg) for name, cfg in _load_yaml()["models"].items()}


def load_tokenizers() -> dict[str, str]:
    return _load_yaml().get("tokenizers", {})


def get_models(names: list[str]) -> list[ModelSpec]:
    models = load_models()
    missing = [n for n in names if n not in models]
    if missing:
        raise SystemExit(f"Unknown model(s) {missing}. Known: {list(models)}")
    return [models[n] for n in names]


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
