#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the human review page for the QA sheets.

    python paper/analysis/qa_gallery.py        # -> paper/results/qa/index.html

Open the file in a browser (no server needed). It lists every case of every
run that has QA sheets (bench/run_cohort.py --qa), with filters for the
cases worth a look: refusals, escalations, attacker hits on the face, a
region close to the brain. Each sheet can be marked "looks right" or
"problem" with a note; marks are kept in the browser and exported with
"Copy review CSV". Commit that CSV as results/qa_review.csv; the analysis
reports how many sheets were reviewed and how many were flagged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from common import RESULTS, read_jsonl  # noqa: E402

QA = RESULTS / "qa"


def cases() -> list[dict]:
    out = []
    for run_dir in sorted(p for p in QA.iterdir() if p.is_dir()):
        recs = {r["id"]: r for r in read_jsonl(RESULTS / "runs" / f"{run_dir.name}.jsonl")}
        for img in sorted(run_dir.glob("*.jpg")):
            r = recs.get(img.stem, {"id": img.stem})
            rep, br, at = r.get("report", {}), r.get("brain", {}), r.get("attack") or {}
            msg = r.get("stderr") or r.get("error") or ""
            dist = br.get("min_dist_mm")
            if dist is None and (rep.get("brain") or {}).get("margin_mm") is not None:
                dist = round(rep["brain"]["margin_mm"], 2)
            out.append({
                "run": run_dir.name, "id": img.stem, "img": f"{run_dir.name}/{img.name}",
                "status": r.get("status", "?"),
                "escalated": bool(rep.get("escalated_to_block")),
                "rounds": rep.get("verify_rounds"),
                "dist": dist,
                "touched": br.get("brain_overlap_voxels", (rep.get("brain") or {}).get("voxels_modified")),
                "hits": {k: v["output_residual_views"] for k, v in at.items()},
                "reason": msg.rsplit(".nii.gz: ", 1)[-1][:200] if r.get("status") != "ok" else "",
            })
    return out


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RadVeil QA review</title>
<style>
:root { --bg:#f6f7f8; --panel:#fff; --fg:#1d2226; --muted:#5f6b74; --rule:#d8dde1; --accent:#b45309;
        --ok:#15803d; --bad:#b91c1c; }
@media (prefers-color-scheme: dark) { :root { --bg:#121518; --panel:#1b2025; --fg:#e4e8eb; --muted:#9aa5ad;
        --rule:#2d353b; --accent:#f59e0b; --ok:#4ade80; --bad:#f87171; color-scheme:dark } }
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif }
header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--rule);
         padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center }
header h1 { font-size:15px; margin:0 8px 0 0 }
label { display:inline-flex; gap:5px; align-items:center; color:var(--muted); font-size:13px }
select, input[type=text], input[type=number] { font:inherit; padding:3px 6px; border:1px solid var(--rule);
         border-radius:4px; background:var(--bg); color:var(--fg) }
input[type=number] { width:4.5em }
button { font:inherit; padding:4px 10px; border:1px solid var(--rule); border-radius:4px; background:var(--bg);
         color:var(--fg); cursor:pointer }
button:focus-visible, a:focus-visible { outline:2px solid var(--accent); outline-offset:2px }
#count { color:var(--muted); font-variant-numeric:tabular-nums }
main { padding:14px 16px; display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:12px }
.card { background:var(--panel); border:1px solid var(--rule); border-radius:6px; overflow:hidden; display:flex;
        flex-direction:column }
.card img { width:100%; display:block; cursor:zoom-in; background:#000 }
.meta { padding:7px 9px; display:flex; flex-direction:column; gap:3px; font-size:12.5px }
.meta b { font-size:13px }
.tags { display:flex; flex-wrap:wrap; gap:4px }
.tag { font-size:11px; padding:1px 6px; border-radius:999px; border:1px solid var(--rule); color:var(--muted) }
.tag.warn { border-color:var(--accent); color:var(--accent) }
.tag.ok { border-color:var(--ok); color:var(--ok) } .tag.bad { border-color:var(--bad); color:var(--bad) }
.verdict { display:flex; gap:6px; padding:0 9px 8px }
.verdict button.on.ok { border-color:var(--ok); color:var(--ok); font-weight:600 }
.verdict button.on.bad { border-color:var(--bad); color:var(--bad); font-weight:600 }
.reason { color:var(--muted) }
#view[hidden] { display:none }
#view { position:fixed; inset:0; background:rgba(0,0,0,.88); z-index:5; display:flex; flex-direction:column;
        align-items:center; justify-content:center; gap:10px; padding:16px }
#view img { max-width:100%; max-height:82vh; background:#fff }
#view .bar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; color:#ddd; justify-content:center }
#view textarea { width:min(700px, 90vw); height:3.2em; font:inherit; border-radius:4px; padding:4px 6px }
</style>
<header>
  <h1>RadVeil QA review</h1>
  <label for="run">run <select id="run"></select></label>
  <label for="status">status <select id="status"><option value="">all</option><option>ok</option><option>refused</option><option>error</option></select></label>
  <label><input type="checkbox" id="esc"> escalated</label>
  <label><input type="checkbox" id="hit"> attacker hit on face</label>
  <label for="near">brain within <input type="number" id="near" min="0" step="0.5" placeholder="mm"> mm</label>
  <label for="rev">review <select id="rev"><option value="">all</option><option value="todo">not reviewed</option><option value="ok">looks right</option><option value="bad">problem</option></select></label>
  <label for="q">id <input type="text" id="q" placeholder="e.g. ixi_t1"></label>
  <span id="count"></span>
  <button id="csv">Copy review CSV</button>
</header>
<main id="grid"></main>
<div id="view" hidden>
  <img id="big" alt="">
  <div class="bar"><span id="bigtitle"></span></div>
  <div class="bar">
    <button id="prev">← previous</button>
    <button id="vok">Looks right (o)</button><button id="vbad">Problem (p)</button>
    <button id="next">next →</button><button id="close">close (esc)</button>
  </div>
  <textarea id="note" placeholder="note for this case (saved with the verdict)"></textarea>
</div>
<script>
const CASES = __DATA__;
const KEY = "radveil-qa-review";
let rev = {};
try { rev = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(rev)); } catch (e) {} };
const k = c => c.run + "/" + c.id;
const $ = id => document.getElementById(id);
const runs = [...new Set(CASES.map(c => c.run))];
$("run").innerHTML = '<option value="">all</option>' + runs.map(r => `<option>${r}</option>`).join("");
let shown = [], cur = -1;

function hitSum(c) { return Object.values(c.hits).reduce((a, b) => a + b, 0); }
function tags(c) {
  const t = [];
  t.push(`<span class="tag ${c.status === "ok" ? "" : "warn"}">${c.status}</span>`);
  if (c.escalated) t.push('<span class="tag warn">escalated</span>');
  if (c.rounds != null) t.push(`<span class="tag">verify rounds ${c.rounds}</span>`);
  if (c.dist != null) t.push(`<span class="tag ${c.dist < 3 ? "warn" : ""}">brain ${c.dist} mm</span>`);
  if (c.touched) t.push(`<span class="tag bad">${c.touched} brain vox</span>`);
  for (const [d, n] of Object.entries(c.hits)) t.push(`<span class="tag ${n ? "warn" : ""}">${d} ${n} on-face</span>`);
  const r = rev[k(c)];
  if (r) t.push(`<span class="tag ${r.v}">${r.v === "ok" ? "looks right" : "problem"}</span>`);
  return t.join("");
}
function render() {
  const run = $("run").value, st = $("status").value, q = $("q").value.trim(), rv = $("rev").value;
  const near = parseFloat($("near").value);
  shown = CASES.filter(c => (!run || c.run === run) && (!st || c.status === st)
    && (!$("esc").checked || c.escalated) && (!$("hit").checked || hitSum(c) > 0)
    && (isNaN(near) || (c.dist != null && c.dist <= near)) && (!q || c.id.includes(q))
    && (!rv || (rv === "todo" ? !rev[k(c)] : rev[k(c)] && rev[k(c)].v === rv)));
  const done = CASES.filter(c => rev[k(c)]).length, bad = CASES.filter(c => rev[k(c)] && rev[k(c)].v === "bad").length;
  $("count").textContent = `${shown.length} shown · ${done}/${CASES.length} reviewed · ${bad} flagged`;
  $("grid").innerHTML = shown.map((c, i) => `<div class="card">
    <img loading="lazy" src="${c.img}" alt="QA sheet ${c.id}" data-i="${i}">
    <div class="meta"><b>${c.id}</b><span class="reason">${c.run}</span><div class="tags">${tags(c)}</div>
    ${c.reason ? `<span class="reason">${c.reason}</span>` : ""}</div>
    <div class="verdict"><button class="ok ${rev[k(c)]?.v === "ok" ? "on" : ""}" data-v="ok" data-i="${i}">Looks right</button>
    <button class="bad ${rev[k(c)]?.v === "bad" ? "on" : ""}" data-v="bad" data-i="${i}">Problem</button></div></div>`).join("");
}
function mark(i, v) {
  const c = shown[i]; const old = rev[k(c)] || {};
  rev[k(c)] = { v, note: old.note || "", t: new Date().toISOString() }; save();
}
function open(i) {
  if (i < 0 || i >= shown.length) return;
  cur = i; const c = shown[i];
  $("big").src = c.img; $("big").alt = "QA sheet " + c.id;
  $("bigtitle").textContent = `${i + 1}/${shown.length} · ${c.run} · ${c.id}` + (rev[k(c)] ? ` · marked ${rev[k(c)].v === "ok" ? "looks right" : "problem"}` : "");
  $("note").value = rev[k(c)]?.note || "";
  $("view").hidden = false;
}
function closeView() { $("view").hidden = true; cur = -1; render(); }
$("grid").addEventListener("click", e => {
  const i = +e.target.dataset.i;
  if (e.target.tagName === "IMG") open(i);
  else if (e.target.dataset.v) { mark(i, e.target.dataset.v); render(); }
});
$("note").addEventListener("input", () => {
  const c = shown[cur]; if (!c) return;
  rev[k(c)] = { ...(rev[k(c)] || { v: "bad", t: new Date().toISOString() }), note: $("note").value }; save();
});
$("prev").onclick = () => open(cur - 1); $("next").onclick = () => open(cur + 1); $("close").onclick = closeView;
$("vok").onclick = () => { mark(cur, "ok"); open(cur + 1); };
$("vbad").onclick = () => { mark(cur, "bad"); open(cur); $("note").focus(); };
document.addEventListener("keydown", e => {
  if ($("view").hidden || e.target === $("note")) return;
  if (e.key === "ArrowRight") open(cur + 1); else if (e.key === "ArrowLeft") open(cur - 1);
  else if (e.key === "Escape") closeView(); else if (e.key === "o") $("vok").click(); else if (e.key === "p") $("vbad").click();
});
for (const id of ["run", "status", "esc", "hit", "near", "rev", "q"]) $(id).addEventListener("input", render);
$("csv").onclick = async () => {
  const esc = s => '"' + String(s).replace(/"/g, '""') + '"';
  const lines = ["run,id,verdict,note,reviewed_at"].concat(Object.entries(rev).map(([key, r]) => {
    const [run, id] = key.split("/"); return [run, id, r.v, esc(r.note || ""), r.t].join(",");
  }));
  const text = lines.join("\n") + "\n";
  try { await navigator.clipboard.writeText(text); $("csv").textContent = "Copied"; }
  catch (e) { const w = window.open(); if (w) { w.document.body.innerText = text; } }
  setTimeout(() => $("csv").textContent = "Copy review CSV", 1500);
};
render();
</script>
"""


def main() -> None:
    if not QA.exists():
        sys.exit("no QA sheets yet: run bench/run_cohort.py with --qa")
    data = cases()
    (QA / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(data)))
    print(f"{len(data)} sheets -> {QA / 'index.html'}")


if __name__ == "__main__":
    main()
