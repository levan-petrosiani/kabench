"""Aggregate results/runs/* into leaderboard tables (Markdown) and charts (PNG)."""

import json
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from .config import RESULTS_DIR, ROOT, RUNS_DIR

CHARTS_DIR = RESULTS_DIR / "charts"
LANG_COLORS = {"en": "#2a78d6", "ka": "#eb6834"}  # categorical slots 1-2
LANG_LABELS = {"en": "English", "ka": "Georgian"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs() -> pd.DataFrame:
    rows = [json.loads(line) for f in RUNS_DIR.glob("*/*.jsonl") for line in f.read_text().splitlines()]
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, sep=".")
    # Keep only the latest run per (track, model, lang, variant): reruns supersede smoke tests
    latest = df.groupby(["track", "model", "lang", "variant"])["run"].transform("max")
    return df[df["run"] == latest]


def _mean(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(dtype=float)


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (well-behaved at 0%, 100% and small n)."""
    if not n:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(df: pd.DataFrame, track: str, metrics: list[str], ci: str | None = None) -> pd.DataFrame:
    """Mean of each metric per model/variant/lang; `ci` (a 0/1 metric) also gets a 95% interval."""
    d = df[df["track"] == track].copy()
    if d.empty:
        return d
    for m in metrics + ["output.prompt_tokens", "output.completion_tokens", "output.latency_s"]:
        d[m] = _mean(d, m)
    d["errors"] = d["output.error"].notna() if "output.error" in d else False
    agg = d.groupby(["model", "variant", "lang"]).agg(
        n=("item_id", "count"), errors=("errors", "sum"),
        **{m.split(".")[-1]: (m, "mean") for m in metrics},
        prompt_tokens=("output.prompt_tokens", "mean"),
        completion_tokens=("output.completion_tokens", "mean"),
        latency_s=("output.latency_s", "mean"),
    ).reset_index()
    if ci:
        name = ci.split(".")[-1]
        counts = d.groupby(["model", "variant", "lang"])[ci].agg(["sum", "count"]).reset_index()
        bounds = [wilson(k, n) for k, n in zip(counts["sum"], counts["count"])]
        counts["ci_lo"], counts["ci_hi"] = zip(*bounds) if bounds else ([], [])
        agg = agg.merge(counts[["model", "variant", "lang", "ci_lo", "ci_hi"]], on=["model", "variant", "lang"])
        agg.insert(agg.columns.get_loc(name) + 1, "95% CI",
                   [f"{lo:.0%}–{hi:.0%}" for lo, hi in zip(agg["ci_lo"], agg["ci_hi"])])
    return agg


def _style(ax, title: str, xlabel: str):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def _save(fig, name: str):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / name, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def chart_token_tax(tok: pd.DataFrame):
    tok = tok.sort_values("ka_en_ratio")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(tok) + 1.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.barh(tok["tokenizer"], tok["ka_en_ratio"], color=LANG_COLORS["ka"], height=0.6)
    ax.axvline(1, color=MUTED, linewidth=1, linestyle="--")
    ax.bar_label(bars, labels=[f"{v:.1f}×" for v in tok["ka_en_ratio"]], padding=4, color=INK, fontsize=9)
    _style(ax, "Georgian token tax: tokens for the same text, Georgian ÷ English", "ratio (1× = parity)")
    _save(fig, "token_tax.png")


def chart_lang_pair(agg: pd.DataFrame, metric: str, title: str, name: str, label_fn=str):
    """Grouped horizontal bars: one row per model (+variant), English vs Georgian."""
    if agg.empty:
        return
    suffix = " · " + agg["variant"] if agg["variant"].nunique() > 1 else ""
    agg = agg.assign(row=agg["model"] + suffix)
    piv = agg.pivot_table(index="row", columns="lang", values=metric).sort_index(ascending=False)
    has_ci = "ci_lo" in agg
    if has_ci:
        lo = agg.pivot_table(index="row", columns="lang", values="ci_lo").reindex(piv.index)
        hi = agg.pivot_table(index="row", columns="lang", values="ci_hi").reindex(piv.index)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(piv) + 1.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    h = 0.36
    for i, lang in enumerate([lg for lg in ("en", "ka") if lg in piv]):
        y = [k + (i - 0.5) * (h + 0.04) for k in range(len(piv))]
        bars = ax.barh(y, piv[lang], height=h, color=LANG_COLORS[lang], label=LANG_LABELS[lang])
        ends = piv[lang]
        if has_ci:
            err = [(piv[lang] - lo[lang]).clip(lower=0), (hi[lang] - piv[lang]).clip(lower=0)]
            ax.errorbar(piv[lang], y, xerr=err, fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=0)
            ends = hi[lang]
        # Value labels sit just past the bar, or past its interval, so they never collide
        for yy, v, end in zip(y, piv[lang], ends):
            if pd.notna(v):
                ax.text(end + 0.012, yy, label_fn(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, -0.04 * 20 / max(len(piv), 1)), ncol=2,
              labelcolor=INK)
    _style(ax, title, "")
    if has_ci:
        ax.set_xlabel("thin line = 95% confidence interval", color=MUTED, fontsize=8, loc="left")
    _save(fig, name)


def overview_table(g: pd.DataFrame, c: pd.DataFrame, a: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """One row per model that ran all three scored tracks: headline metric per track and language."""
    parts = [(g[g["variant"] == "zero_shot"], "exact_match", "grammar"), (c, "accuracy", "belebele"), (a, "success", "agent")]
    cols = [d.pivot_table(index="model", columns="lang", values=m).add_prefix(f"{name} ") for d, m, name in parts]
    t = pd.concat(cols, axis=1).dropna()
    for lg in ("ka", "en"):
        t[f"avg {lg}"] = t[[f"{name} {lg}" for _, _, name in parts]].mean(axis=1)
    t["gap"] = t["avg en"] - t["avg ka"]
    # Models that ran a reduced subset (reasoning group) are marked
    n = runs[runs["track"] == "comprehension"].groupby("model")["item_id"].count()
    t.index = [f"{m} ✱" if n.get(m, 0) < n.max() else m for m in t.index]
    return t.sort_values("avg ka", ascending=False).rename_axis("model").reset_index()


def chart_overview(t: pd.DataFrame):
    """Dumbbell: English vs Georgian average per model; the gap is the story."""
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(t) + 1.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = range(len(t))
    ax.hlines(y, t["avg ka"], t["avg en"], color=GRID, linewidth=2.5, zorder=1)
    for lang in ("en", "ka"):
        ax.scatter(t[f"avg {lang}"], y, s=46, color=LANG_COLORS[lang], edgecolors="#fcfcfb", linewidths=1.5,
                   zorder=2, label=LANG_LABELS[lang])
    for k, (_, r) in enumerate(t.iterrows()):
        ax.text(r["avg ka"] - 0.015, k, f"{r['avg ka']:.0%}", ha="right", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y), t["model"])
    ax.set_xlim(0, 1.04)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, -0.035), ncol=2, labelcolor=INK)
    _style(ax, "English vs Georgian: average score per model", "")
    ax.set_xlabel("mean of grammar exact match, Belebele accuracy and agent success", color=MUTED,
                  fontsize=8, loc="left")
    _save(fig, "overview.png")


def quant_table(grammar: pd.DataFrame, comp: pd.DataFrame, agent: pd.DataFrame) -> pd.DataFrame:
    """Pair every `<name>-q8` with `<name>` and show per-language score deltas."""
    metrics = [("grammar", grammar, "exact_match", "zero_shot"), ("belebele", comp, "accuracy", None),
               ("agent", agent, "success", None)]
    rows = []
    for track, df, metric, variant in metrics:
        if df.empty:
            continue
        d = df[df["variant"] == variant] if variant else df
        val = d.set_index(["model", "lang"])[metric]
        for (model, lang), q8 in val.items():
            base = model.removesuffix("-q8")
            if model.endswith("-q8") and (base, lang) in val.index:
                q4 = val[(base, lang)]
                rows.append({"model": base, "track": track, "lang": lang, "q4": q4, "q8": q8, "delta": q8 - q4})
    return pd.DataFrame(rows)


def subset_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Score every model on the items the reasoning group ran (they are nested subsets)."""
    sel = runs[((runs["track"] == "grammar") & (runs["variant"] == "zero_shot")) | (runs["track"] == "comprehension")]
    counts = sel.groupby(["model", "track", "lang"])["item_id"].transform("count")
    full = counts.groupby([sel["track"], sel["lang"]]).transform("max")
    reduced = sel[counts < full]
    if reduced.empty:
        return pd.DataFrame()
    keys = set(zip(reduced["track"], reduced["lang"], reduced["item_id"]))
    on_subset = sel[[k in keys for k in zip(sel["track"], sel["lang"], sel["item_id"])]].copy()
    on_subset["score"] = _mean(on_subset, "scores.exact_match").fillna(_mean(on_subset, "scores.accuracy"))
    on_subset["reasoning"] = on_subset["model"].isin(set(reduced["model"]))
    out = on_subset.pivot_table(index=["model", "reasoning"], columns=["track", "lang"], values="score")
    out.columns = [f"{'grammar EM' if t == 'grammar' else 'belebele acc'} ({lg})" for t, lg in out.columns]
    return out.reset_index().sort_values(["reasoning", "model"])


def _md(df: pd.DataFrame) -> str:
    df = df.drop(columns=["ci_lo", "ci_hi"], errors="ignore")
    return df.to_markdown(index=False, floatfmt=".2f") if not df.empty else "_no runs yet_"


def build_report():
    runs = load_runs()
    sections = ["# kabench leaderboard\n", "_Generated by `kabench report`. Latest run per model/lang/variant._\n"]
    readme = []  # the README only gets the overview; the full tables live in LEADERBOARD.md

    tok_path = RESULTS_DIR / "tokens_offline.csv"
    if tok_path.exists():
        tok = pd.read_csv(tok_path)
        chart_token_tax(tok)
        cols = ["tokenizer", "ka_en_ratio", "ka_chars_per_token", "en_chars_per_token", "ka_tokens_per_word", "en_tokens_per_word"]
        sections += ["## Tokenization cost (offline, 488 parallel FLORES passages)\n",
                     "![token tax](charts/token_tax.png)\n", _md(tok[cols].sort_values("ka_en_ratio"))]
    live = RESULTS_DIR / "tokens_live.csv"
    if live.exists():
        sections += ["\n### Tokenization measured on the serving backend\n",
                     _md(pd.read_csv(live)[["tokenizer", "model_id", "ka_en_ratio", "ka_chars_per_token", "en_chars_per_token"]])]

    if not runs.empty:
        g = summarize(runs, "grammar", ["scores.exact_match", "scores.char_similarity", "scores.gain",
                                        "scores.over_correction", "scores.missed", "scores.format_ok"],
                      ci="scores.exact_match")
        chart_lang_pair(g[g["variant"] == "zero_shot"] if not g.empty else g, "exact_match",
                        "Grammar correction (zero-shot prompt): exact match", "grammar.png", lambda v: f"{v:.0%}")
        c = summarize(runs, "comprehension", ["scores.accuracy", "scores.parsed"], ci="scores.accuracy")
        chart_lang_pair(c, "accuracy", "Belebele reading comprehension: accuracy", "comprehension.png",
                        lambda v: f"{v:.0%}")
        a = summarize(runs, "agent", ["scores.success", "scores.language_match", "scores.steps",
                                      "scores.invalid_calls", "scores.total_tokens"], ci="scores.success")
        chart_lang_pair(a, "success", "Agent scenarios: task success", "agent.png", lambda v: f"{v:.0%}")

        if not (ov := overview_table(g, c, a, runs)).empty:
            chart_overview(ov)
            sections += ["\n## Overview\n", "![overview](charts/overview.png)\n",
                         "Headline metric per track: grammar exact match (zero-shot prompt), Belebele accuracy, "
                         "agent task success. ✱ = reasoning group, scored on a fixed subset.\n", _md(ov)]
            readme = ["_Generated by `kabench report`._ Headline metric per track: grammar exact match (zero-shot "
                      "prompt), Belebele accuracy, agent task success; `gap` = English avg − Georgian avg. "
                      "✱ = reasoning group, scored on a fixed subset.\n", _md(ov),
                      "\nPer-track tables with 95% confidence intervals, over-correction rates, the Q4 vs Q8 "
                      "comparison, the reasoning subset, agent results by scenario type and token cost on each "
                      "backend: **[results/LEADERBOARD.md](results/LEADERBOARD.md)**. To read every answer, "
                      "run `uv run kabench review`."]
        sections += ["\n## How big a difference is real?\n",
                     "Every score comes with a 95% Wilson interval in the tables below. With 15 agent scenarios "
                     "one scenario is ~7 points, and a 60% score has an interval of roughly 36–80%; with 100 "
                     "Belebele questions it narrows to about ±10 points. Treat gaps inside the intervals as ties.\n"]
        sections += ["\n## Grammar correction\n", "![grammar](charts/grammar.png)\n",
                     "`over_correction` = share of already-correct sentences the model changed; "
                     "`missed` = share of wrong sentences returned unchanged; "
                     "`gain` = edit-distance progress towards the gold sentence.\n", _md(g),
                     "\n## Reading comprehension (Belebele)\n", "![comprehension](charts/comprehension.png)\n", _md(c),
                     "\n## Agent behaviour\n", "![agent](charts/agent.png)\n", _md(a)]
        if not (q := quant_table(g, c, a)).empty:
            sections += ["\n## Quantization: Q4 vs Q8\n",
                         "Same model, 4-bit vs 8-bit weights. Δ = Q8 − Q4: does quantization hurt Georgian more than English?\n",
                         _md(q)]
        if not (r := subset_table(runs)).empty:
            sections += ["\n## Reasoning group vs everyone, on the same subset\n",
                         "Reasoning models run on a fixed subset (30 grammar zero-shot + 30 Belebele items per language). "
                         "Here every model is scored on exactly those items.\n", _md(r)]
        if not (ag := runs[runs["track"] == "agent"]).empty:
            ag = ag.assign(success=_mean(ag, "scores.success"))
            by_type = ag.pivot_table(index="metadata.type", columns=["model", "lang"], values="success")
            by_type.columns = [f"{m} ({lg})" for m, lg in by_type.columns]
            sections += ["\n### Agent success by scenario type\n", by_type.reset_index().rename(
                columns={"metadata.type": "type"}).to_markdown(index=False, floatfmt=".2f")]

    text = "\n".join(sections) + "\n"
    (RESULTS_DIR / "LEADERBOARD.md").write_text(text)
    if readme:
        _update_readme("\n".join(readme) + "\n")
    print(f"Report written to {RESULTS_DIR.relative_to(ROOT)}/LEADERBOARD.md and charts/")


def _update_readme(body: str):
    readme = ROOT / "README.md"
    start, end = "<!-- leaderboard:start -->", "<!-- leaderboard:end -->"
    text = readme.read_text()
    if start not in text:
        return
    readme.write_text(re.sub(f"{start}.*?{end}", lambda _: f"{start}\n{body}\n{end}", text, flags=re.DOTALL))
