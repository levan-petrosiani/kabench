"""One self-contained HTML page to check every test item: input, expected answer and
every model's answer side by side, with per-item review notes kept in the browser."""

import json

from .config import RESULTS_DIR, RUNS_DIR
from .report import load_runs

PRIMARY = {"grammar": "exact_match", "comprehension": "accuracy", "agent": "success"}
VARIANT_ORDER = ["zero_shot", "few_shot", "zero_shot_en_instructions", "belebele", "tools-v1"]
TRACK_LABELS = {"grammar": "Grammar", "comprehension": "Belebele", "agent": "Agent"}


def _answer(row: dict) -> dict:
    out, scores = row["output"], row["scores"] or {}
    ok = scores.get(PRIMARY[row["track"]])
    answer = {"model": row["model"], "variant": row["variant"], "ok": None if ok is None else bool(ok),
              "text": out.get("text") or "", "error": out.get("error")}
    if row["track"] == "agent":
        answer["calls"] = [{"tool": c["tool"], "args": c["args"], "result": c["result"]} for c in out.get("calls", [])]
        answer["failed"] = [k for k, v in scores.items() if k.endswith("_ok") and v is False]
    if row["track"] == "grammar" and scores.get("over_correction"):
        answer["note"] = "changed a correct sentence"
    return answer


def collect() -> list[dict]:
    """Group the latest run of every model by test item."""
    latest = set(load_runs()["run"])
    groups: dict[tuple, dict] = {}
    for f in sorted(RUNS_DIR.glob("*/*.jsonl")):
        for line in f.read_text().splitlines():
            row = json.loads(line)
            if row["run"] not in latest:
                continue
            # One card per test item: the expected answer is shared by every prompt variant
            key = (row["track"], row["lang"], row["item_id"])
            meta = row.get("metadata") or {}
            g = groups.setdefault(key, {
                "id": row["item_id"], "track": row["track"], "lang": row["lang"],
                "category": meta.get("category") or meta.get("type") or "", "input": row["input"],
                "expected": row["expected"], "answers": []})
            g["answers"].append(_answer(row))
    rank = {v: i for i, v in enumerate(VARIANT_ORDER)}
    items = list(groups.values())
    for g in items:
        g["answers"].sort(key=lambda a: (rank.get(a["variant"], 99), a["model"]))
        g["variants"] = list(dict.fromkeys(a["variant"] for a in g["answers"]))
        g["pass"] = {}
        for v in g["variants"]:
            scored = [a["ok"] for a in g["answers"] if a["variant"] == v and a["ok"] is not None]
            g["pass"][v] = sum(scored) / len(scored) if scored else None
    # Georgian first; within a track, items in id order
    items.sort(key=lambda g: (list(PRIMARY).index(g["track"]), g["lang"] != "ka", g["id"]))
    return items


def build_review() -> str:
    items = collect()
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    out = RESULTS_DIR / "review.html"
    out.write_text(PAGE.replace("/*DATA*/", data).replace("/*LABELS*/", json.dumps(TRACK_LABELS)))
    return f"{out.relative_to(RESULTS_DIR.parent)} ({len(items)} items, {sum(len(g['answers']) for g in items)} answers)"


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kabench answer review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Georgian:wght@400;600&family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #fcfcfb; --surface: #ffffff; --ink: #0b0b0b; --muted: #52514e; --faint: #8a8984; --line: #e4e3df;
  --good: #1f7a3d; --good-bg: #e8f4ec; --bad: #b3261e; --bad-bg: #fbeaea; --accent: #2a78d6; --flag: #b26a00;
  --code: #f4f3f0;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141413; --surface: #1c1c1a; --ink: #f2f1ed; --muted: #b5b3ad; --faint: #85837d; --line: #33322f;
    --good: #6cc78a; --good-bg: #1d2e22; --bad: #f08c84; --bad-bg: #3a1f1d; --accent: #7fb0ec; --flag: #e8a948;
    --code: #24241f;
  }
}
:root[data-theme="dark"] {
  --bg: #141413; --surface: #1c1c1a; --ink: #f2f1ed; --muted: #b5b3ad; --faint: #85837d; --line: #33322f;
  --good: #6cc78a; --good-bg: #1d2e22; --bad: #f08c84; --bad-bg: #3a1f1d; --accent: #7fb0ec; --flag: #e8a948;
  --code: #24241f;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 Inter, "Noto Sans Georgian", system-ui, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line); }
.wrap { max-width: 1100px; margin: 0 auto; padding: 0 16px; }
h1 { font-size: 18px; margin: 14px 0 2px; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 10px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; padding-bottom: 12px; align-items: center; }
.controls > * { max-width: 100%; min-width: 0; }
html, body { overflow-x: clip; }
select, input[type=search], button { font: inherit; font-size: 14px; color: var(--ink); background: var(--surface);
  border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; }
input[type=search] { flex: 1 1 180px; min-width: 0; }
button { cursor: pointer; }
button:hover { border-color: var(--muted); }
.count { color: var(--muted); font-size: 13px; margin-left: auto; }
main { padding: 16px 0 60px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; margin-bottom: 16px; overflow: hidden; }
.card.problem { border-color: var(--bad); }
.card.okd { border-color: var(--good); }
.head { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: baseline; padding: 12px 16px 0; }
.id { font-weight: 600; }
.tag { font-size: 12px; color: var(--muted); border: 1px solid var(--line); border-radius: 99px; padding: 1px 8px; }
.rate { margin-left: auto; font-size: 13px; color: var(--muted); display: flex; align-items: center; gap: 8px; }
.bar { width: 80px; height: 6px; background: var(--bad-bg); border-radius: 3px; overflow: hidden; }
.bar > i { display: block; height: 100%; background: var(--good); }
.block { padding: 10px 16px; }
.label { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--faint); margin-bottom: 2px; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; }
.input.long { max-height: 9em; overflow: auto; }
.expected { background: var(--code); border-radius: 6px; padding: 8px 10px; white-space: pre-wrap; overflow-wrap: anywhere;
  font-size: 14px; }
.review { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 4px 16px 12px; }
.review textarea { flex: 1 1 260px; min-height: 34px; font: inherit; font-size: 14px; color: var(--ink);
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; resize: vertical; }
.review button.on.okb { background: var(--good-bg); border-color: var(--good); color: var(--good); }
.review button.on.badb { background: var(--bad-bg); border-color: var(--bad); color: var(--bad); }
details.answers { border-top: 1px solid var(--line); }
details.answers > summary { padding: 10px 16px; cursor: pointer; color: var(--muted); font-size: 14px; }
.row { display: grid; grid-template-columns: 190px 70px 1fr; gap: 10px; padding: 8px 16px; border-top: 1px solid var(--line); }
.row .m { font-size: 13px; color: var(--muted); overflow-wrap: anywhere; }
.verdict { font-size: 13px; font-weight: 600; }
.verdict.y { color: var(--good); } .verdict.n { color: var(--bad); } .verdict.x { color: var(--faint); }
.ans { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 14px; }
.ans .err { color: var(--bad); }
.ans .why { color: var(--flag); font-size: 13px; }
.calls { font-size: 12.5px; color: var(--muted); margin: 4px 0 0; padding-left: 18px; }
.calls li { overflow-wrap: anywhere; }
.vhead { padding: 8px 16px; border-top: 1px solid var(--line); background: var(--code); font-size: 12px;
  text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.more { text-align: center; }
.empty { color: var(--muted); text-align: center; padding: 40px 0; }
@media (max-width: 640px) {
  .controls select { flex: 1 1 45%; }
  .count { margin-left: 0; flex-basis: 100%; }
  .sub { font-size: 12px; }
  .row { grid-template-columns: 1fr auto; }
  .row .ans { grid-column: 1 / -1; }
  .rate { margin-left: 0; }
}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>kabench answer review</h1>
    <p class="sub">Every test item with its expected answer and every model's answer. Mark items as correct or as a
      problem; marks and notes stay in this browser. Use <b>Export notes</b> to save them.</p>
    <div class="controls">
      <select id="track"></select>
      <select id="lang"><option value="">Both languages</option><option value="ka">Georgian</option><option value="en">English</option></select>
      <select id="variant"></select>
      <select id="show">
        <option value="">All items</option>
        <option value="mostly">Most models failed (&lt; 50% passed)</option>
        <option value="none">No model passed</option>
        <option value="unreviewed">Not reviewed yet</option>
        <option value="problem">Marked as problem</option>
      </select>
      <input type="search" id="q" placeholder="Search id, input, expected or answer">
      <button id="export">Export notes</button>
      <span class="count" id="count"></span>
    </div>
  </div>
</header>
<main class="wrap" id="list"></main>
<script>
const ITEMS = /*DATA*/;
const TRACKS = /*LABELS*/;
const PAGE = 40;
const KEY = "kabench-review-v1";
let notes = {};
try { notes = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(notes)); } catch (e) {} };
// Notes belong to the test item, not the prompt: the expected answer is the same for every grammar prompt
const uid = it => `${it.track}/${it.lang}/${it.id}`;
const VNAMES = { zero_shot: "zero-shot prompt", few_shot: "few-shot prompt",
  zero_shot_en_instructions: "zero-shot, English instructions", belebele: "Belebele", "tools-v1": "agent" };
// The pass rate shown for an item: the selected prompt, else its first (zero-shot) prompt
const variantOf = it => $("variant").value && it.variants.includes($("variant").value) ? $("variant").value : it.variants[0];
const passOf = it => it.pass[variantOf(it)];
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text; return e; };

const tracks = [...new Set(ITEMS.map(i => i.track))];
$("track").innerHTML = '<option value="">All tracks</option>' +
  tracks.map(t => `<option value="${t}">${TRACKS[t] || t}</option>`).join("");
function fillVariants() {
  const t = $("track").value;
  const vs = [...new Set(ITEMS.filter(i => !t || i.track === t).flatMap(i => i.variants))];
  $("variant").innerHTML = '<option value="">All prompts</option>' +
    vs.map(v => `<option value="${v}">${VNAMES[v] || v}</option>`).join("");
}
fillVariants();

function fmtExpected(it) {
  if (it.track === "comprehension") return `Correct option: ${it.expected}`;
  if (it.track === "agent") return JSON.stringify(it.expected, null, 2);
  return it.expected + (it.expected === it.input ? "\n(already correct: the model should return it unchanged)" : "");
}
function haystack(it) {
  return (it.id + " " + it.category + " " + JSON.stringify(it.input) + " " + JSON.stringify(it.expected) + " " +
    it.answers.map(a => a.text).join(" ")).toLowerCase();
}
ITEMS.forEach(it => it._hay = haystack(it));

function filtered() {
  const t = $("track").value, l = $("lang").value, v = $("variant").value, s = $("show").value;
  const q = $("q").value.trim().toLowerCase();
  return ITEMS.filter(it => (!t || it.track === t) && (!l || it.lang === l) && (!v || it.variants.includes(v)) &&
    (!q || it._hay.includes(q)) &&
    (!s || (s === "mostly" && passOf(it) != null && passOf(it) < 0.5) || (s === "none" && passOf(it) === 0) ||
      (s === "unreviewed" && !notes[uid(it)]?.mark) || (s === "problem" && notes[uid(it)]?.mark === "problem")));
}

function answerRow(a) {
  const row = el("div", "row");
  row.append(el("div", "m", a.model));
  const v = a.ok === null ? ["x", "— error"] : a.ok ? ["y", "✓ pass"] : ["n", "✗ fail"];
  row.append(el("div", "verdict " + v[0], v[1]));
  const ans = el("div", "ans");
  ans.append(el("span", "", a.text || (a.error ? "" : "(empty answer)")));
  if (a.error) ans.append(el("div", "err", a.error));
  if (a.note) ans.append(el("div", "why", a.note));
  if (a.failed && a.failed.length) ans.append(el("div", "why", "failed checks: " + a.failed.join(", ")));
  if (a.calls && a.calls.length) {
    const ul = el("ol", "calls");
    a.calls.forEach(c => ul.append(el("li", "", `${c.tool}(${JSON.stringify(c.args)}) → ${JSON.stringify(c.result)}`)));
    ans.append(ul);
  } else if (a.calls) ans.append(el("div", "why", "no tool calls"));
  row.append(ans);
  return row;
}

function card(it) {
  const key = uid(it), n = notes[key] || {};
  const c = el("article", "card" + (n.mark === "problem" ? " problem" : n.mark === "ok" ? " okd" : ""));
  const head = el("div", "head");
  head.append(el("span", "id", it.id), el("span", "tag", TRACKS[it.track] || it.track),
    el("span", "tag", it.lang === "ka" ? "Georgian" : "English"));
  if (it.category) head.append(el("span", "tag", it.category));
  const rate = el("span", "rate"), v = variantOf(it), p = it.pass[v];
  if (p != null) {
    const n = it.answers.filter(a => a.variant === v).length;
    const bar = el("span", "bar"), fill = el("i"); fill.style.width = (p * 100) + "%"; bar.append(fill);
    rate.append(bar, el("span", "", `${Math.round(p * 100)}% of ${n} models passed` +
      (it.variants.length > 1 ? ` (${VNAMES[v]})` : "")));
  }
  head.append(rate);
  c.append(head);

  const inp = el("div", "block");
  const inputText = typeof it.input === "string" ? it.input : JSON.stringify(it.input, null, 2);
  inp.append(el("div", "label", "Input"), el("div", "text input" + (inputText.length > 400 ? " long" : ""), inputText));
  const exp = el("div", "block");
  exp.append(el("div", "label", "Expected"), el("div", "expected", fmtExpected(it)));
  c.append(inp, exp);

  const rev = el("div", "review");
  const okb = el("button", "okb" + (n.mark === "ok" ? " on" : ""), "✓ Expected answer is right");
  const badb = el("button", "badb" + (n.mark === "problem" ? " on" : ""), "⚑ Problem");
  const ta = el("textarea"); ta.placeholder = "Note (what is wrong, what the expected answer should be)";
  ta.value = n.note || "";
  const set = patch => { notes[key] = { ...notes[key], ...patch }; save(); };
  okb.onclick = () => { set({ mark: n.mark === "ok" ? "" : "ok" }); render(true); };
  badb.onclick = () => { set({ mark: n.mark === "problem" ? "" : "problem" }); render(true); };
  ta.onchange = () => set({ note: ta.value });
  rev.append(okb, badb, ta);
  c.append(rev);

  const det = el("details", "answers");
  const sel = $("variant").value;
  const answers = it.answers.filter(a => !sel || a.variant === sel);
  const passed = answers.filter(a => a.ok).length;
  det.append(el("summary", "", `Model answers (${passed} of ${answers.length} passed` +
    (it.variants.length > 1 && !sel ? `, across ${it.variants.length} prompts)` : ")")));
  det.addEventListener("toggle", () => {
    if (!det.open || det.children.length > 1) return;
    let last = null;
    answers.forEach(a => {
      if (it.variants.length > 1 && a.variant !== last) { det.append(el("div", "vhead", VNAMES[a.variant] || a.variant)); last = a.variant; }
      det.append(answerRow(a));
    });
  });
  c.append(det);
  return c;
}

let shown = PAGE;
function render(keepScroll) {
  const items = filtered(), list = $("list"), y = window.scrollY;
  const all = new Set(ITEMS.map(uid)), reviewed = [...all].filter(k => notes[k]?.mark).length;
  $("count").textContent = `${items.length} shown · ${reviewed} of ${all.size} test items reviewed`;
  list.replaceChildren();
  if (!items.length) { list.append(el("p", "empty", "No items match these filters.")); return; }
  items.slice(0, shown).forEach(it => list.append(card(it)));
  if (items.length > shown) {
    const more = el("p", "more"), b = el("button", "", `Show ${Math.min(PAGE, items.length - shown)} more`);
    b.onclick = () => { shown += PAGE; render(true); };
    more.append(b); list.append(more);
  }
  if (keepScroll) window.scrollTo(0, y);
}
["track", "lang", "variant", "show"].forEach(id => $(id).addEventListener("change", () => {
  if (id === "track") fillVariants(); shown = PAGE; render(); }));
$("q").addEventListener("input", () => { shown = PAGE; render(); });
$("export").onclick = () => {
  const rows = Object.entries(notes).filter(([, n]) => n.mark || n.note)
    .map(([k, n]) => { const [track, lang, id] = k.split("/"); return { id, track, lang, ...n }; });
  const blob = new Blob([JSON.stringify(rows, null, 2)], { type: "application/json" });
  const a = el("a"); a.href = URL.createObjectURL(blob); a.download = "kabench-review-notes.json"; a.click();
};
render();
</script>
</body>
</html>
"""
