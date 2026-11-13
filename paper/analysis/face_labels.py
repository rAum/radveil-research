#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Blinded human ground truth for refusals: is there a face in the input?

    python paper/analysis/face_labels.py --dataset synthstrip --data-dir $DATA_DIR
    python paper/analysis/face_labels.py --dataset ixi --data-dir $DATA_DIR
    # -> paper/results/qa/faces/<dataset>.html  (open in a browser, no server)

RadVeil refuses a volume when it finds no face it can anonymize. Whether a
refusal was right is a question about the input alone, so the reviewer sees
the input only: surface renders at five yaws, two more iso-surface
thresholds, a view from below, and two slices, with no RadVeil status, in an
order shuffled per dataset. Each scan gets one label:

  face     full face: eyes, nose, mouth and chin all in the scan
  partial  only part of the face, e.g. a thin slab around the eyes or a
           field-of-view cut below the nose
  none     no facial features (EPI/DWI/angiography, cropped, skull-stripped,
           already defaced)
  unsure   cannot tell from these views

Only a refusal on a full face counts as a false refusal: on a partial face
there may be too little to fit and verify a defacing, so refusing is a
defensible decision, not a failure; the refused scan still shows part of a
face and must be withheld or handled by the pipeline. Partial faces are
reported apart.

Which scans are shown: every SynthStrip scan (582); for IXI, where every
volume is a full head, every scan the adaptive run refused or failed on plus
as many randomly chosen accepted scans, so the set does not give away the
decision. "Download CSV" saves the labels; put them in
results/face_labels.csv (several reviewers: concatenate, the reviewer column
keeps them apart). analysis/make_assets.py crosses them with the run records.
The sheets are git-ignored (results/qa/); the labels hold no image data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from common import RESULTS, read_jsonl  # noqa: E402
from qa import _canonical, _show, _slices  # noqa: E402

OUT = RESULTS / "qa" / "faces"
RENDER = 300
SEED = "radveil-face-labels-v1"
# (title, yaw, pitch, threshold multiple)
VIEWS = [("yaw -70", -70, 0, 1), ("yaw -35", -35, 0, 1), ("front", 0, 0, 1), ("yaw +35", 35, 0, 1),
         ("yaw +70", 70, 0, 1), ("front, threshold x0.5", 0, 0, 0.5), ("front, threshold x2", 0, 0, 2),
         ("from below, pitch -30", 0, -30, 1)]


def select(dataset: str) -> list[dict]:
    rows = list(csv.DictReader((RESULTS / "manifests" / f"{dataset}.csv").open()))
    if dataset != "ixi":
        return rows
    recs = {r["id"]: r for r in read_jsonl(RESULTS / "runs" / "ixi__adaptive.jsonl")}
    if not recs:
        sys.exit("no IXI adaptive run yet (results/runs/ixi__adaptive.jsonl)")
    bad = [r for r in rows if r["id"] in recs and recs[r["id"]]["status"] != "ok"]
    ok = [r for r in rows if r["id"] in recs and recs[r["id"]]["status"] == "ok"]
    return bad + random.Random(SEED).sample(ok, min(len(ok), len(bad)))


def sheet(job: tuple[str, str]) -> str:
    src, dest = job
    if Path(dest).exists():
        return dest
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import radveil

    vol = radveil.Volume(src)
    fig, axs = plt.subplots(2, 5, figsize=(10, 4.6))
    for ax in axs.flat:
        ax.set_axis_off()
    base = vol.render(yaw_deg=0.0, size=RENDER)["threshold"]
    for ax, (title, yaw, pitch, mult) in zip(axs.flat, VIEWS):
        ax.imshow(vol.render(yaw_deg=float(yaw), pitch_deg=float(pitch), size=RENDER,
                             threshold=base * mult)["rgb"])
        ax.set_title(title, fontsize=7)
    d, sp = _canonical(src)
    sag, axi, _ = _slices(d, tuple(s // 2 for s in d.shape))
    _show(axs[1, 3], sag, (sp[2], sp[1]), title="mid-sagittal")
    _show(axs[1, 4], axi, (sp[1], sp[0]), title="mid-axial")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94, wspace=0.04, hspace=0.18)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    tmp = dest + ".tmp.jpg"
    fig.savefig(tmp, dpi=100, pil_kwargs={"quality": 85})
    plt.close(fig)
    os.replace(tmp, dest)
    return dest


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face presence labels</title>
<style>
:root { --bg:#f6f7f8; --panel:#fff; --fg:#1d2226; --muted:#5f6b74; --rule:#d8dde1; --accent:#b45309 }
@media (prefers-color-scheme: dark) { :root { --bg:#121518; --panel:#1b2025; --fg:#e4e8eb; --muted:#9aa5ad;
        --rule:#2d353b; --accent:#f59e0b; color-scheme:dark } }
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif }
header { background:var(--panel); border-bottom:1px solid var(--rule); padding:10px 16px; display:flex;
         flex-wrap:wrap; gap:8px 14px; align-items:center }
header h1 { font-size:15px; margin:0 6px 0 0 }
#count { color:var(--muted); font-variant-numeric:tabular-nums }
input, button, textarea { font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--rule);
         border-radius:4px; padding:4px 9px }
button { cursor:pointer }
button:focus-visible { outline:2px solid var(--accent); outline-offset:2px }
details { padding:8px 16px; color:var(--muted); font-size:13px; max-width:900px }
main { padding:10px 16px; display:flex; flex-direction:column; align-items:center; gap:10px }
#img { max-width:100%; max-height:70vh; background:#fff; border:1px solid var(--rule) }
.bar { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; align-items:center }
.lab.on { border-color:var(--accent); color:var(--accent); font-weight:600 }
textarea { width:min(760px, 100%); height:2.6em }
</style>
<header>
  <h1>Face presence · __DATASET__</h1>
  <label>reviewer <input id="who" size="10" placeholder="initials"></label>
  <span id="count"></span>
  <label><input type="checkbox" id="todo"> unlabelled only</label>
  <button id="dl">Download CSV</button>
</header>
<details open><summary>What to label: how much face is in this original scan? (whether RadVeil accepted it is hidden on purpose)</summary>
<p><b>1 full face</b>: eyes, nose, mouth and chin are all in the scan.
<b>2 partial face</b>: only part of it, e.g. a thin slab around the eyes, or the scan stops below the nose.
<b>3 no face</b>: no facial features (EPI, DWI, angiography, cropped, skull-stripped or already defaced).
<b>4 can't tell</b>: these views do not let you decide.
Keys: 1-4 label and go to the next, arrows move, n focuses the note.</p></details>
<main>
  <div class="bar"><button id="prev">← previous</button><span id="pos"></span><button id="next">next →</button></div>
  <img id="img" alt="">
  <div class="bar">
    <button class="lab" data-l="face">1 full face</button><button class="lab" data-l="partial">2 partial face</button>
    <button class="lab" data-l="none">3 no face</button><button class="lab" data-l="unsure">4 can't tell</button>
  </div>
  <textarea id="note" placeholder="optional note for this scan"></textarea>
</main>
<script>
const DATASET = "__DATASET__", CASES = __DATA__, KEY = "radveil-face-labels-" + DATASET;
const LABELS = ["face", "partial", "none", "unsure"];
let lab = {}, who = "";
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); who = localStorage.getItem(KEY + "-who") || ""; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(lab)); localStorage.setItem(KEY + "-who", who); } catch (e) {} };
const $ = id => document.getElementById(id);
$("who").value = who;
$("who").oninput = () => { who = $("who").value.trim(); save(); };
let order = CASES, cur = 0;
function list() { return $("todo").checked ? CASES.filter(c => !lab[c.id] || c.id === order[cur]?.id) : CASES; }
function show(i) {
  order = list();
  if (!order.length) { $("pos").textContent = "all labelled"; $("img").removeAttribute("src"); return; }
  cur = Math.max(0, Math.min(i, order.length - 1));
  const c = order[cur], l = lab[c.id];
  $("img").src = c.img; $("img").alt = "input views of scan " + c.n;
  $("pos").textContent = `scan ${c.n} · ${cur + 1}/${order.length}`;
  for (const b of document.querySelectorAll(".lab")) b.classList.toggle("on", !!l && l.label === b.dataset.l);
  $("note").value = l?.note || "";
  const done = CASES.filter(c => lab[c.id]).length;
  $("count").textContent = `${done}/${CASES.length} labelled`;
}
function set(label) {
  const c = order[cur]; if (!c) return;
  lab[c.id] = { label, note: $("note").value, who, t: new Date().toISOString() }; save();
  const next = $("todo").checked ? cur : cur + 1;
  show(next);
}
for (const b of document.querySelectorAll(".lab")) b.onclick = () => set(b.dataset.l);
$("prev").onclick = () => show(cur - 1); $("next").onclick = () => show(cur + 1);
$("todo").onchange = () => show(0);
$("note").oninput = () => { const c = order[cur]; if (c && lab[c.id]) { lab[c.id].note = $("note").value; save(); } };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") { if (e.key === "Escape") e.target.blur(); return; }
  if (e.key >= "1" && e.key <= "4") set(LABELS[+e.key - 1]);
  else if (e.key === "ArrowRight") show(cur + 1); else if (e.key === "ArrowLeft") show(cur - 1);
  else if (e.key === "n") { e.preventDefault(); $("note").focus(); }
});
$("dl").onclick = () => {
  const q = s => '"' + String(s).replace(/"/g, '""') + '"';
  const rows = ["dataset,id,label,note,reviewer,labelled_at"].concat(CASES.filter(c => lab[c.id]).map(c => {
    const l = lab[c.id]; return [DATASET, c.id, l.label, q(l.note || ""), q(l.who || who), l.t].join(",");
  }));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join("\n") + "\n"], { type: "text/csv" }));
  a.download = `face_labels_${DATASET}${who ? "_" + who : ""}.csv`; a.click();
};
const first = CASES.findIndex(c => !lab[c.id]);
show(first < 0 ? 0 : first);
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=["synthstrip", "ixi"])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    rows = select(a.dataset)
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    jobs = [(str(a.data_dir / r["image"]), str(OUT / a.dataset / f"{r['id']}.jpg")) for r in rows]
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, _ in enumerate(ex.map(sheet, jobs), 1):
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} sheets", flush=True)
    # shuffled per dataset so neighbouring scans do not share a subset or a decision
    rows.sort(key=lambda r: hashlib.sha256(f"{SEED}/{r['id']}".encode()).hexdigest())
    data = [{"id": r["id"], "n": i + 1, "img": f"{a.dataset}/{r['id']}.jpg"} for i, r in enumerate(rows)]
    page = OUT / f"{a.dataset}.html"
    page.write_text(PAGE.replace("__DATASET__", a.dataset).replace("__DATA__", json.dumps(data)))
    print(f"{len(data)} scans -> {page}")


if __name__ == "__main__":
    main()
