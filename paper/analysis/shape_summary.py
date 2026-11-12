#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""3D shape linkage attack on IXI: summary for the paper.

    python paper/analysis/shape_summary.py      # -> results/shape_summary.json

Gallery: one clean T1 descriptor per subject. Probes: the subjects' T2 and
PD scans. Conditions (rank-1 identification, subject-cluster bootstrap 95%
intervals, chance = 1/gallery):

  control        clean T2/PD, whole head and outside the face sector;
  anonymized     each configuration's accepted outputs, the same two
                 variants; "release risk" counts refused scans clean;
  deeper         adaptive outputs and clean controls at the 2x surface,
                 against a 2x gallery;
  pose           IXI's contrasts come from one session and share head pose.
                 To show the match is anatomy, not pose, a fixed random sample
                 of probes is moved by a random rigid motion (10 deg, 10 mm)
                 and matched by ICP against every gallery candidate, clean and
                 anonymized;
  paired         every configuration on the subjects all were run on.

AUC is not reported: impostor scores exist only for the ICP-refined
candidates, which would bias it.

Matching: analysis/shape_reid.py (radial-map ranking, ICP re-ranking of the
top K candidates). Needs the git-ignored descriptors in results/shape/.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, read_jsonl, read_manifest  # noqa: E402

K = 20
POSE = (10.0, 10.0)  # degrees, mm
POSE_PROBES = 150
CONFIGS = ("adaptive", "adaptive_noverify", "cut", "adaptive_ears", "pydeface", "mri_deface", "quickshear")
_G = None


def _load_gallery(dataset: str) -> dict:
    z = np.load(RESULTS / "shape" / f"{dataset}__gallery.npz")
    out: dict = {}
    for key in z.files:
        i, m, f = key.split("__")
        out.setdefault(i, {}).setdefault(m, {})[f] = z[key]
    return out


def _load_out(run: str, i: str) -> dict:
    z = np.load(RESULTS / "shape" / run / f"{i}.npz")
    out: dict = {}
    for key in z.files:
        m, f = key.split("_", 1)
        out.setdefault("x" + m[3:], {})[f] = z[key]
    return out


def _init(g):
    global _G
    _G = g


def _eval(args):
    import shape_reid

    probes, subj, exclude, pose = args
    return shape_reid.evaluate(_G, probes, subj, exclude_face=exclude, k=K, pose=pose)


def _evaluate(gallery: dict, probes: dict, subj: dict, exclude: bool, jobs: int, pose=None) -> dict:
    ids = sorted(probes)
    chunks = [ids[j::jobs] for j in range(jobs)]
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(gallery,)) as ex:
        parts = list(ex.map(_eval, [({i: probes[i] for i in c}, subj, exclude, pose) for c in chunks if c]))
    ranks, order = [], []
    for c, p in zip([c for c in chunks if c], parts):
        kept = [i for i in sorted(c) if subj[i] in {subj[g] for g in gallery}]
        ranks += p["ranks"]
        order += kept
    gen = np.concatenate([p["genuine"] for p in parts]) if parts else np.array([])
    imp = np.concatenate([p["impostor"] for p in parts]) if parts else np.array([])
    return {"ids": order, "ranks": np.array(ranks), "genuine": gen, "impostor": imp}


def _row(label: str, ev: dict, subj: dict, gsize: int, rng) -> dict:
    import reid

    hit = (np.asarray(ev["ranks"]) == 1).astype(float)
    s = np.array([subj[i] for i in ev["ids"]])
    return {"label": label, "n": int(len(hit)), "subjects": int(len(np.unique(s))), "rank1": float(hit.mean()),
            "rank1_ci": reid._ci(hit, s, rng), "rank10": float((np.asarray(ev["ranks"]) <= 10).mean()),
            "chance": 1 / gsize, "p_vs_chance": reid._p_chance(hit, s, gsize)}


def main() -> None:
    jobs = os.cpu_count() or 4
    rng = np.random.default_rng(0)
    man = {r["id"]: r for r in read_manifest(RESULTS / "manifests" / "ixi.csv")}
    G = _load_gallery("ixi")
    subj = {i: man[i]["subject"] for i in man}
    t1 = {i: d for i, d in G.items() if man[i]["contrast"] == "T1"}
    probes_all = {i: d for i, d in G.items() if man[i]["contrast"] in ("T2", "PD")
                  and man[i]["subject"] in {subj[g] for g in t1}}
    gal = {m: {i: d[m] for i, d in t1.items()} for m in ("x1", "x2")}
    rows = {}
    out = {"gallery": len(t1), "k": K, "rows": rows}

    for exclude, tag, lab in ((False, "Whole", "whole head"), (True, "Outside", "outside the face")):
        ev = _evaluate(gal["x1"], {i: d["x1"] for i, d in probes_all.items()}, subj, exclude, jobs)
        rows[f"control{tag}"] = _row(f"clean T2/PD, {lab}", ev, subj, len(t1), rng)
        ctl_rank = dict(zip(ev["ids"], ev["ranks"]))
        for cfg in CONFIGS:
            runs = {r["id"]: r for r in read_jsonl(RESULTS / "runs" / f"ixi__{cfg}.jsonl")}
            rdir = RESULTS / "shape" / f"ixi__{cfg}"
            if not runs or not rdir.exists():
                continue
            acc = {i: _load_out(f"ixi__{cfg}", i)["x1"] for i in probes_all
                   if runs.get(i, {}).get("status") == "ok" and (rdir / f"{i}.npz").exists()}
            if not acc:
                continue
            ev_a = _evaluate(gal["x1"], acc, subj, exclude, jobs)
            r = _row(f"{cfg}, accepted, {lab}", ev_a, subj, len(t1), rng)
            r["control_rank1_same_probes"] = float(np.mean([ctl_rank[i] == 1 for i in ev_a["ids"]]))
            rows[f"{cfg}{tag}"] = r
            # release risk: refused scans keep their clean rank
            arank = dict(zip(ev_a["ids"], ev_a["ranks"]))
            ids = sorted(i for i in ctl_rank if i in runs)  # probes this configuration saw
            rr = {"ids": ids, "ranks": np.array([arank.get(i, ctl_rank[i]) for i in ids]),
                  "genuine": np.array([]), "impostor": np.array([])}
            rows[f"{cfg}{tag}Risk"] = _row(f"{cfg}, release risk, {lab}", rr, subj, len(t1), rng)
    # deeper surface, adaptive only
    runs = {r["id"]: r for r in read_jsonl(RESULTS / "runs" / "ixi__adaptive.jsonl")}
    rdir = RESULTS / "shape" / "ixi__adaptive"
    if runs and rdir.exists():
        acc = {i: _load_out("ixi__adaptive", i)["x2"] for i in probes_all
               if runs.get(i, {}).get("status") == "ok" and (rdir / f"{i}.npz").exists()}
        ev = _evaluate(gal["x2"], {i: probes_all[i]["x2"] for i in acc}, subj, False, jobs)
        rows["controlDeep"] = _row("clean T2/PD (same probes), 2x surface", ev, subj, len(t1), rng)
        ev = _evaluate(gal["x2"], acc, subj, False, jobs)
        rows["adaptiveDeep"] = _row("adaptive, accepted, 2x surface", ev, subj, len(t1), rng)
    # pose-perturbed control: same random probe sample for clean and anonymized
    pool = sorted(probes_all)
    pick = sorted(np.random.default_rng(0).choice(pool, min(POSE_PROBES, len(pool)), replace=False).tolist())
    ev = _evaluate(gal["x1"], {i: probes_all[i]["x1"] for i in pick}, subj, False, jobs, POSE)
    rows["controlPose"] = _row(f"clean T2/PD, moved {POSE[0]:g} deg/{POSE[1]:g} mm, whole head", ev, subj, len(t1), rng)
    runs = {r["id"]: r for r in read_jsonl(RESULTS / "runs" / "ixi__adaptive.jsonl")}
    rdir = RESULTS / "shape" / "ixi__adaptive"
    if runs and rdir.exists():
        acc = {i: _load_out("ixi__adaptive", i)["x1"] for i in pick
               if runs.get(i, {}).get("status") == "ok" and (rdir / f"{i}.npz").exists()}
        if acc:
            ev = _evaluate(gal["x1"], {i: probes_all[i]["x1"] for i in acc}, subj, False, jobs, POSE)
            rows["controlPoseSame"] = _row("clean, same probes as below, moved", ev, subj, len(t1), rng)
            ev = _evaluate(gal["x1"], acc, subj, False, jobs, POSE)
            rows["adaptivePose"] = _row("adaptive, accepted, moved, whole head", ev, subj, len(t1), rng)
    # paired: subjects every configuration was run on
    seen = []
    for cfg in CONFIGS:
        r_ = {r["id"] for r in read_jsonl(RESULTS / "runs" / f"ixi__{cfg}.jsonl")}
        if r_:
            seen.append(r_)
    if seen:
        common = sorted(i for i in probes_all if all(i in x for x in seen))
        out["paired_probes"] = len(common)
        ev = _evaluate(gal["x1"], {i: probes_all[i]["x1"] for i in common}, subj, False, jobs)
        rows["pairedControl"] = _row("paired sample, clean", ev, subj, len(t1), rng)
        ctl = dict(zip(ev["ids"], ev["ranks"]))
        for cfg in CONFIGS:
            runs = {r["id"]: r for r in read_jsonl(RESULTS / "runs" / f"ixi__{cfg}.jsonl")}
            rdir = RESULTS / "shape" / f"ixi__{cfg}"
            if not runs or not rdir.exists():
                continue
            acc = {i: _load_out(f"ixi__{cfg}", i)["x1"] for i in common
                   if runs.get(i, {}).get("status") == "ok" and (rdir / f"{i}.npz").exists()}
            evc = _evaluate(gal["x1"], acc, subj, False, jobs) if acc else {"ids": [], "ranks": []}
            rk = dict(zip(evc["ids"], evc["ranks"]))
            rr = {"ids": common, "ranks": np.array([rk.get(i, ctl[i]) for i in common])}
            rows[f"paired{cfg.title().replace('_', '')}"] = _row(f"paired sample, {cfg} (release)", rr, subj, len(t1), rng)
    p = RESULTS / "shape_summary.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", p, {k: round(v["rank1"], 3) for k, v in rows.items()})


if __name__ == "__main__":
    main()
