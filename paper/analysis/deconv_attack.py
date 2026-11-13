#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Blur-inversion attack on RadVeil's blur-family output.

    python paper/analysis/deconv_attack.py --data-dir $DATA_DIR [--subjects 60]

The threat model gives the attacker the code, so the blur kernel is known:
RadVeil reports the Gaussian sigma it used (and the adaptive mode's heavy
kernel is that sigma times its adaptive factor). The attacker deconvolves the
anonymized region with Richardson-Lucy (15 iterations) under each of the two
kernels and re-runs face recognition.

Sample: held-out IXI subjects (not used in RadVeil's development) whose T2
the adaptive run accepted; one probe per subject (T2), so probes are
independent. Each scan is anonymized again in memory with the Python API
(the same code as the cohort run), deconvolved, rendered and embedded with
bench/attack.py; it is matched against the full clean-T1 gallery with the
fused (max over depth) attacker of analysis/reid.py.

Writes results/deconv_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, read_jsonl, read_manifest  # noqa: E402

ITERS = 15
_A = None


def _init(a):
    global _A
    _A = a
    os.environ["RAYON_NUM_THREADS"] = "2"


def richardson_lucy(obs: np.ndarray, sigma_vox, mask: np.ndarray, iters: int = ITERS) -> np.ndarray:
    """Richardson-Lucy deconvolution under a Gaussian PSF; voxels outside
    `mask` keep their observed values (they were never blurred)."""
    from scipy.ndimage import gaussian_filter

    lo = float(obs.min())
    y = obs - lo + 1e-3  # RL needs positive data
    est = y.copy()
    for _ in range(iters):
        conv = gaussian_filter(est, sigma_vox) + 1e-6
        est = est * gaussian_filter(y / conv, sigma_vox)
        est[~mask] = y[~mask]
    return est + lo - 1e-3


def case(row: dict) -> dict:
    import attack
    import radveil

    path = str(Path(_A.data_dir) / row["image"])
    vin = radveil.Volume(path)
    emb_clean, ref = attack.embed_volume(vin, _A.models)
    v = radveil.Volume(path)
    rep = radveil.deface_volume(v, mode="adaptive", return_mask=True)
    mask = np.asarray(rep["mask"]).astype(bool)
    data = np.asarray(v.data(), np.float32)
    affine = np.asarray(v.affine)
    spacing = np.asarray(v.spacing, float)
    sigma = float(rep.get("final_blur_sigma_mm") or rep["blur_sigma_mm"])
    # the adaptive mode's heavy kernel: sigma x adaptive_factor (2.5, docs/parameters.md)
    kernels = {"plain": None, "sigma": sigma, "sigma_x_adaptive": sigma * 2.5}
    out = {"id": row["id"], "escalated": bool(rep.get("escalated_to_block")), "sigma_mm": sigma,
           "emb": {f"{r}_clean": e for r, e in emb_clean.items()}}
    for name, s_mm in kernels.items():
        vol = v if s_mm is None else radveil.Volume.from_array(
            richardson_lucy(data, s_mm / spacing, mask).astype(np.float32), affine)
        e = attack.embed_volume(vol, _A.models, ref)
        for r in attack.RECOGNIZERS:
            out["emb"][f"{r}_{name}"] = np.nanmax(np.stack([e[f"{r}_oracle"], e[f"{r}_near"]]), 0) \
                if np.isfinite(e[f"{r}_near"]).any() else e[f"{r}_oracle"]
    return out


def main() -> None:
    import reid

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--subjects", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    a.models = str(Path(a.data_dir) / "models")
    man = {r["id"]: r for r in read_manifest(RESULTS / "manifests" / "ixi.csv")}
    runs = {r["id"]: r for r in read_jsonl(RESULTS / "runs" / "ixi__adaptive.jsonl")}
    pool = sorted(i for i, r in runs.items() if r.get("status") == "ok" and man[i]["contrast"] == "T2"
                  and man[i]["development"] == "0")
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(pool, min(a.subjects, len(pool)), replace=False).tolist())
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a,)) as ex:
        cases = list(ex.map(case, [man[i] for i in pick]))

    g = np.load(RESULTS / "embeddings" / "ixi__gallery.npz")
    ids = g["ids"].tolist()
    out = {"n": len(cases), "iterations": ITERS, "escalated": sum(c["escalated"] for c in cases), "rows": {}}
    for rec in reid.RECOGNIZERS:
        clean = dict(zip(ids, reid._templates(g[rec])))
        t1 = {man[i]["subject"]: i for i in ids if i in man and man[i]["contrast"] == "T1"}
        gsub = sorted(s for s, i in t1.items() if np.isfinite(clean[i][reid.MULTS.index(1.0), 0]))
        G = np.stack([clean[t1[s]] for s in gsub])
        use = [c for c in cases if man[c["id"]]["subject"] in gsub]
        truth = np.array([gsub.index(man[c["id"]]["subject"]) for c in use])
        subj = np.array([man[c["id"]]["subject"] for c in use])
        for name in ("clean", "plain", "sigma", "sigma_x_adaptive"):
            P = np.stack([reid._templates(c["emb"][f"{rec}_{name}"][None])[0] for c in use])
            m = reid._metrics(reid._fuse(reid._sim(P, G)), truth)
            s = reid._summary(m, subj, np.random.default_rng(0), len(gsub))
            out["rows"][f"{rec}_{name}"] = s
    p = RESULTS / "deconv_summary.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", p, {k: round(v["rank1"], 3) for k, v in out["rows"].items()})


if __name__ == "__main__":
    main()
