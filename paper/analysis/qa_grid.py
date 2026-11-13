#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Fast grid review of every released output: one row per scan, one column
per configuration.

    python paper/analysis/qa_grid.py        # -> paper/results/qa/grid.html

Each accepted output is shown as two crops of its QA sheet, the output
rendered from the front (is the face gone?) and the output mid-sagittal slice
with the brain outline (is the brain intact?); refusals and errors are shown
as a grey cell with the reason. The first column is the input. Reviewing:
click a cell to flag it as a problem, press Enter to mark every other
accepted cell on the page as looking right and go to the next page; the
full QA sheet opens with the corner button or by hovering a cell and
pressing Space. Verdicts are shared with results/qa/index.html (same browser
storage, same CSV), so "Download review CSV" gives results/qa_review.csv.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_gallery import QA, cases  # noqa: E402

THUMBS = QA / "thumbs"
# panel boxes (left, top, right, bottom) in the QA sheets of bench/qa.py:
# 880 px wide, accepted outputs 764 px tall (input / output / checks rows),
# refusals and errors 291 px tall (input row only)
OUT_FRONT, OUT_SAG = (228, 302, 436, 509), (663, 302, 871, 509)
IN_FRONT = {764: (228, 52, 436, 258), 291: (228, 64, 436, 279)}


def _thumb(job: tuple[str, str, str]) -> None:
    from PIL import Image

    kind, src, dest = job
    d = Path(dest)
    if d.exists() and d.stat().st_mtime >= Path(src).stat().st_mtime:
        return
    im = Image.open(src).convert("RGB")
    if kind == "out":
        if im.height != 764:
            return
        t = Image.new("RGB", (416, 207), "white")
        t.paste(im.crop(OUT_FRONT), (0, 0))
        t.paste(im.crop(OUT_SAG), (208, 0))
    else:
        box = IN_FRONT.get(im.height)
        if box is None:
            return
        t = im.crop(box)
    d.parent.mkdir(parents=True, exist_ok=True)
    t.save(d, quality=80)


def main() -> None:
    if not QA.exists():
        sys.exit("no QA sheets yet: run bench/run_cohort.py with --qa")
    data = cases()
    jobs, rows, runs = [], {}, {}
    for c in data:
        ds = c["run"].split("__", 1)[0]
        runs.setdefault(ds, [])
        if c["run"] not in runs[ds]:
            runs[ds].append(c["run"])
        sheet = QA / c["img"]
        row = rows.setdefault((ds, c["id"]), {"id": c["id"], "ds": ds, "in": None, "c": {}})
        if row["in"] is None or c["run"].endswith("__adaptive"):
            row["in"] = f"thumbs/input/{ds}/{c['id']}.jpg"
            jobs.append(("in", str(sheet), str(QA / row["in"])))
        cell = {k: c[k] for k in ("status", "escalated", "rounds", "dist", "touched", "hits", "reason")}
        cell["sheet"] = c["img"]
        if c["status"] == "ok":
            cell["t"] = f"thumbs/{c['run']}/{c['id']}.jpg"
            jobs.append(("out", str(sheet), str(QA / cell["t"])))
        row["c"][c["run"]] = cell
    # the input crop comes from the adaptive sheet when there is one; drop
    # crops queued from other sheets for the same destination
    jobs = list({j[2]: j for j in jobs}.values())
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(min(8, os.cpu_count() or 4)) as ex:
        for i, _ in enumerate(ex.map(_thumb, jobs, chunksize=64), 1):
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)} crops", flush=True)
    order = ["adaptive", "adaptive_noverify", "blur", "cut", "adaptive_ears", "adaptive_guard",
             "adaptive_robust", "pydeface", "mri_deface", "quickshear"]
    for ds in runs:
        runs[ds].sort(key=lambda r: order.index(r.split("__", 1)[1]) if r.split("__", 1)[1] in order else 99)
    payload = {"runs": runs, "rows": sorted(rows.values(), key=lambda r: (r["ds"], r["id"]))}
    page = QA / "grid.html"
    page.write_text(PAGE.replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
    n_ok = sum(c["status"] == "ok" for c in data)
    print(f"{len(rows)} scans, {n_ok} accepted outputs -> {page}")


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RadVeil QA grid</title>
<style>
:root { --bg:#f6f7f8; --panel:#fff; --fg:#1d2226; --muted:#5f6b74; --rule:#d8dde1; --accent:#b45309;
        --ok:#15803d; --bad:#b91c1c; --off:#e9ecee }
@media (prefers-color-scheme: dark) { :root { --bg:#121518; --panel:#1b2025; --fg:#e4e8eb; --muted:#9aa5ad;
        --rule:#2d353b; --accent:#f59e0b; --ok:#4ade80; --bad:#f87171; --off:#20262b; color-scheme:dark } }
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif }
header { position:sticky; top:0; z-index:3; background:var(--panel); border-bottom:1px solid var(--rule);
         padding:8px 12px; display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center }
header h1 { font-size:14px; margin:0 6px 0 0 }
label { display:inline-flex; gap:4px; align-items:center; color:var(--muted) }
select, input[type=text], input[type=number], button, textarea { font:inherit; color:var(--fg); background:var(--bg);
         border:1px solid var(--rule); border-radius:4px; padding:3px 8px }
button { cursor:pointer } button.primary { border-color:var(--accent); color:var(--accent); font-weight:600 }
button:focus-visible { outline:2px solid var(--accent); outline-offset:2px }
#cols { display:flex; flex-wrap:wrap; gap:4px 10px; padding:6px 12px; border-bottom:1px solid var(--rule);
        background:var(--panel) }
#count { color:var(--muted); font-variant-numeric:tabular-nums }
.help { padding:6px 12px; color:var(--muted) }
.wrap { overflow-x:auto; padding:0 12px 40px }
table { border-collapse:separate; border-spacing:4px }
th { position:sticky; top:0; font-weight:600; font-size:12px; text-align:left; color:var(--muted); padding:2px 4px;
     white-space:nowrap }
td { vertical-align:top; padding:0 }
td.id { font-size:12px; white-space:nowrap; padding:4px 6px 0 0; color:var(--muted) }
.cell { position:relative; width:208px; height:104px; border:2px solid transparent; border-radius:4px;
        overflow:hidden; background:var(--off); cursor:pointer }
.cell img { width:100%; height:100%; object-fit:cover; display:block }
.cell.inp { width:104px; cursor:default }
.cell.ok { border-color:var(--ok) } .cell.bad { border-color:var(--bad); box-shadow:0 0 0 2px var(--bad) }
.cell.na { cursor:default; color:var(--muted); font-size:11px; padding:6px; display:flex; flex-direction:column; gap:2px }
.cell.na b { color:var(--fg); font-size:11.5px }
.badges { position:absolute; left:3px; bottom:3px; display:flex; gap:3px }
.badge { font-size:10px; line-height:1; padding:2px 4px; border-radius:3px; background:rgba(0,0,0,.72); color:#fde68a }
.badge.red { color:#fca5a5 }
.zoom { position:absolute; right:3px; top:3px; font-size:11px; line-height:1; padding:3px 5px; border-radius:3px;
        border:0; background:rgba(0,0,0,.6); color:#fff }
.mark { position:absolute; right:3px; bottom:3px; font-size:11px; font-weight:700; padding:1px 5px; border-radius:3px;
        background:var(--bad); color:#fff }
.pager { display:flex; gap:8px; align-items:center; padding:8px 12px }
#view[hidden] { display:none }
#view { position:fixed; inset:0; background:rgba(0,0,0,.9); z-index:5; display:flex; flex-direction:column;
        align-items:center; justify-content:center; gap:8px; padding:12px }
#view img { max-width:100%; max-height:84vh; background:#fff }
#view .bar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; color:#ddd; justify-content:center }
#view textarea { width:min(700px, 90vw); height:2.8em }
</style>
<header>
  <h1>RadVeil QA grid</h1>
  <label for="ds">dataset <select id="ds"></select></label>
  <label for="show">rows <select id="show">
    <option value="all">all</option><option value="todo">with unreviewed outputs</option>
    <option value="risk">with a warning badge</option><option value="bad">with a flagged output</option>
    <option value="some">accepted by at least one tool</option></select></label>
  <label for="q">id <input type="text" id="q" size="10" placeholder="e.g. ixi_t1"></label>
  <label for="per">per page <input type="number" id="per" min="5" max="200" value="25"></label>
  <span id="count"></span>
  <button id="dl">Download review CSV</button>
</header>
<div id="cols"></div>
<div class="help">Each output cell: rendered from the front (left) and mid-sagittal slice with the brain outline in blue
(right). Check that the face is gone and the brain is untouched. <b>Click</b> a cell to flag a problem (click again to
unflag), <b>Enter</b> or "Page looks right" marks every other output on the page as fine and goes on,
<b>←/→</b> change page, hover + <b>Space</b> or ⤢ opens the full sheet. Badges: <b>esc</b> escalated,
<b>Nmm</b> brain closer than 3 mm, <b>brain</b> brain voxels changed, <b>hit</b> a face detector still fires.</div>
<div class="pager"><button id="pp">← previous</button><span id="pg"></span><button id="np">next →</button>
  <button class="primary" id="pageok">Page looks right ⏎</button></div>
<div class="wrap"><table id="t"></table></div>
<div id="view" hidden>
  <img id="big" alt="">
  <div class="bar"><span id="bigtitle"></span></div>
  <div class="bar"><button id="vok">Looks right (o)</button><button id="vbad">Problem (p)</button>
    <button id="vclose">close (esc)</button></div>
  <textarea id="note" placeholder="note for this output (saved with the verdict)"></textarea>
</div>
<script>
const D = __DATA__;
const KEY = "radveil-qa-review", COLKEY = "radveil-qa-grid-cols";
let rev = {}, hidden = {};
try { rev = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
try { hidden = JSON.parse(localStorage.getItem(COLKEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(rev)); } catch (e) {} };
const $ = id => document.getElementById(id);
const short = r => r.split("__")[1];
let page = 0, rowsShown = [], hover = null, open = null;
$("ds").innerHTML = Object.keys(D.runs).map(d => `<option>${d}</option>`).join("");

function risk(c) {
  const b = [];
  if (c.escalated) b.push(["esc", ""]);
  if (c.dist != null && c.dist < 3) b.push([c.dist + "mm", ""]);
  if (c.touched) b.push(["brain", "red"]);
  if (Object.values(c.hits || {}).some(n => n > 0)) b.push(["hit", "red"]);
  return b;
}
function cols() { return D.runs[$("ds").value].filter(r => !hidden[r]); }
function drawCols() {
  $("cols").innerHTML = D.runs[$("ds").value].map(r =>
    `<label><input type="checkbox" data-r="${r}" ${hidden[r] ? "" : "checked"}> ${short(r)}</label>`).join("");
}
function filtered() {
  const ds = $("ds").value, show = $("show").value, q = $("q").value.trim(), cs = cols();
  return D.rows.filter(r => r.ds === ds && (!q || r.id.includes(q)) && (() => {
    const cells = cs.map(k => [k, r.c[k]]).filter(([, c]) => c);
    if (show === "todo") return cells.some(([k, c]) => c.status === "ok" && !rev[k + "/" + r.id]);
    if (show === "risk") return cells.some(([, c]) => c.status === "ok" && risk(c).length);
    if (show === "bad") return cells.some(([k]) => rev[k + "/" + r.id]?.v === "bad");
    if (show === "some") return cells.some(([, c]) => c.status === "ok");
    return true;
  })());
}
function cellHTML(k, r) {
  const c = r.c[k];
  if (!c) return `<div class="cell na">not run</div>`;
  if (c.status !== "ok") return `<div class="cell na"><b>${c.status}</b>${(c.reason || "").slice(0, 90)}
    <button class="zoom" data-z="${k}|${r.id}" title="open full sheet">⤢</button></div>`;
  const v = rev[k + "/" + r.id]?.v;
  return `<div class="cell ${v || ""}" data-k="${k}|${r.id}"><img loading="lazy" src="${c.t}" alt="${short(k)} ${r.id}">
    <div class="badges">${risk(c).map(([t, cl]) => `<span class="badge ${cl}">${t}</span>`).join("")}</div>
    <button class="zoom" data-z="${k}|${r.id}" title="open full sheet">⤢</button>${v === "bad" ? '<span class="mark">!</span>' : ""}</div>`;
}
function render() {
  rowsShown = filtered();
  const per = Math.max(5, +$("per").value || 25), pages = Math.max(1, Math.ceil(rowsShown.length / per));
  page = Math.min(page, pages - 1);
  const cs = cols(), slice = rowsShown.slice(page * per, page * per + per);
  $("t").innerHTML = `<tr><th>scan</th><th>input</th>${cs.map(k => `<th>${short(k)}</th>`).join("")}</tr>` +
    slice.map(r => `<tr><td class="id">${r.id}</td><td><div class="cell inp"><img loading="lazy" src="${r.in}" alt="input ${r.id}"></div></td>` +
      cs.map(k => `<td>${cellHTML(k, r)}</td>`).join("") + "</tr>").join("");
  $("pg").textContent = `page ${page + 1} / ${pages} · ${rowsShown.length} scans`;
  const all = D.rows.flatMap(r => Object.entries(r.c).filter(([, c]) => c.status === "ok").map(([k]) => k + "/" + r.id));
  const done = all.filter(x => rev[x]).length, bad = all.filter(x => rev[x]?.v === "bad").length;
  $("count").textContent = `${done}/${all.length} outputs reviewed · ${bad} flagged`;
}
function set(key, v) { rev[key] = { v, note: rev[key]?.note || "", t: new Date().toISOString() }; save(); }
function pageOK() {
  const per = Math.max(5, +$("per").value || 25), cs = cols();
  for (const r of rowsShown.slice(page * per, page * per + per))
    for (const k of cs) { const key = k + "/" + r.id; if (r.c[k]?.status === "ok" && !rev[key]) set(key, "ok"); }
  page++; render(); window.scrollTo(0, 0);
}
function zoom(z) {
  const [k, id] = z.split("|"); const r = D.rows.find(x => x.ds === k.split("__")[0] && x.id === id);
  open = k + "/" + id; $("big").src = r.c[k].sheet; $("big").alt = "QA sheet " + open;
  $("bigtitle").textContent = open + (rev[open] ? " · marked " + (rev[open].v === "ok" ? "looks right" : "problem") : "");
  $("note").value = rev[open]?.note || ""; $("view").hidden = false;
}
function close() { $("view").hidden = true; open = null; render(); }
$("t").addEventListener("click", e => {
  const z = e.target.closest("[data-z]"); if (z) { zoom(z.dataset.z); return; }
  const c = e.target.closest("[data-k]"); if (!c) return;
  const [k, id] = c.dataset.k.split("|"), key = k + "/" + id;
  if (rev[key]?.v === "bad") delete rev[key], save(); else set(key, "bad");
  render();
});
$("t").addEventListener("mouseover", e => { const c = e.target.closest("[data-k],[data-z]"); hover = c ? (c.dataset.k || c.dataset.z) : null; });
$("cols").addEventListener("change", e => { const r = e.target.dataset.r; if (e.target.checked) delete hidden[r]; else hidden[r] = 1;
  try { localStorage.setItem(COLKEY, JSON.stringify(hidden)); } catch (x) {} render(); });
$("ds").addEventListener("change", () => { page = 0; drawCols(); render(); });
for (const id of ["show", "q", "per"]) $(id).addEventListener("input", () => { page = 0; render(); });
$("pp").onclick = () => { page = Math.max(0, page - 1); render(); };
$("np").onclick = () => { page++; render(); };
$("pageok").onclick = pageOK;
$("vok").onclick = () => { set(open, "ok"); close(); };
$("vbad").onclick = () => { set(open, "bad"); $("note").focus(); };
$("vclose").onclick = close;
$("note").addEventListener("input", () => { if (!open) return;
  rev[open] = { ...(rev[open] || { v: "bad", t: new Date().toISOString() }), note: $("note").value }; save(); });
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("view").hidden) { e.preventDefault(); $("note").blur(); close(); return; }
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT" && e.target.type !== "checkbox") return;
  if (!$("view").hidden) {
    if (e.key === "Escape" || e.key === " ") { e.preventDefault(); close(); }
    else if (e.key === "o") $("vok").click(); else if (e.key === "p") $("vbad").click();
    return;
  }
  if (e.key === "Enter") { e.preventDefault(); pageOK(); }
  else if (e.key === "ArrowRight") $("np").click(); else if (e.key === "ArrowLeft") $("pp").click();
  else if (e.key === " " && hover) { e.preventDefault(); zoom(hover); }
});
$("dl").onclick = () => {
  const esc = s => '"' + String(s).replace(/"/g, '""') + '"';
  const lines = ["run,id,verdict,note,reviewed_at"].concat(Object.entries(rev).map(([key, r]) => {
    const [run, id] = key.split("/"); return [run, id, r.v, esc(r.note || ""), r.t].join(",");
  }));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" }));
  a.download = "qa_review.csv"; a.click();
};
drawCols(); render();
</script>
"""


if __name__ == "__main__":
    main()
