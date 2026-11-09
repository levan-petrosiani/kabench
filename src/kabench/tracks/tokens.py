"""Track C - tokenization cost of Georgian vs English ("the Georgian tax").

Two measurements over the same parallel FLORES passages:
  * offline: count tokens with each model's HF tokenizer (no server needed)
  * live:    read `usage.prompt_tokens` from the serving backend, which also
             covers models whose tokenizer isn't on HF or is gated
"""

import pandas as pd
from rich.progress import track as progress

from ..belebele import passages
from ..client import chat
from ..config import RESULTS_DIR, ModelSpec


def _stats(name: str, count, pairs: list[tuple[str, str]]) -> dict:
    row = {"tokenizer": name}
    for lang, idx in (("ka", 0), ("en", 1)):
        texts = [p[idx] for p in pairs]
        tokens = sum(count(t) for t in texts)
        words = sum(len(t.split()) for t in texts)
        chars = sum(len(t) for t in texts)
        row |= {f"{lang}_tokens": tokens,
                f"{lang}_tokens_per_word": tokens / words,
                f"{lang}_chars_per_token": chars / tokens}
    row["ka_en_ratio"] = row["ka_tokens"] / row["en_tokens"]
    return row


def run_offline(tokenizers: dict[str, str]) -> pd.DataFrame:
    from tokenizers import Tokenizer

    pairs = passages()
    rows = []
    for name, repo in tokenizers.items():
        try:
            tok = Tokenizer.from_pretrained(repo)
        except Exception as e:
            print(f"  skip {name} ({repo}): {type(e).__name__} - gated or missing tokenizer.json")
            continue
        rows.append(_stats(name, lambda t: len(tok.encode(t, add_special_tokens=False).ids), pairs) | {"repo": repo})
        print(f"  {name:14} ka/en = {rows[-1]['ka_en_ratio']:.2f}x")
    df = pd.DataFrame(rows).sort_values("ka_en_ratio")
    df.to_csv(RESULTS_DIR / "tokens_offline.csv", index=False, float_format="%.3f")
    return df


def run_live(specs: list[ModelSpec], n: int = 60) -> pd.DataFrame:
    pairs = passages()[:n]
    rows = []
    for spec in specs:
        # Prompt tokens include the chat template; subtract an empty message to isolate the text.
        # Some templates reject empty content: fall back to "." and count it as one token.
        base = chat(spec, [{"role": "user", "content": ""}], name="tokens-baseline", max_tokens=1)
        offset = 0
        if base.error:
            base = chat(spec, [{"role": "user", "content": "."}], name="tokens-baseline", max_tokens=1)
            offset = 1
        if base.error:
            print(f"  skip {spec.name}: {base.error}")
            continue

        def count(text, spec=spec, base=base.prompt_tokens - offset):
            return chat(spec, [{"role": "user", "content": text}], name="tokens", max_tokens=1).prompt_tokens - base

        cached = {}
        for ka, en in progress(pairs, description=spec.name):
            cached[ka], cached[en] = count(ka), count(en)
        rows.append(_stats(spec.name, cached.__getitem__, pairs) | {"model_id": spec.model_id})
        print(f"  {spec.name:14} ka/en = {rows[-1]['ka_en_ratio']:.2f}x")
    if not rows:
        return pd.DataFrame()
    # Merge with earlier runs: models are measured one at a time as they get loaded
    path = RESULTS_DIR / "tokens_live.csv"
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old["tokenizer"].isin(df["tokenizer"])], df], ignore_index=True)
    df.sort_values("ka_en_ratio").to_csv(path, index=False, float_format="%.3f")
    return df
