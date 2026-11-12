#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualitative figure: renders and a sagittal slice before/after, per mode.

    python paper/analysis/make_figures.py --data-dir $DATA_DIR [--case ID]

Needs the data (the SynthStrip set), so its output, tex/figures/fig_modes.png,
is committed. The case is processed fresh in memory with the Python API at
the paper's settings. Without --case it is the first IXI T1 scan that the
adaptive cohort run accepted without escalating to the opaque fill, i.e. a
typical outcome rather than a hand-picked one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from common import RESULTS, read_jsonl, read_manifest  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "tex" / "figures"
MODES = ("adaptive", "blur", "cut")


def sagittal(vol, mask=None):
    """Mid-sagittal slice in display orientation (anterior right, superior up).
    Also returns the row/column spacing ratio so imshow keeps anatomy
    proportions on anisotropic voxels."""
    a = np.asarray(vol.data() if callable(vol.data) else vol.data, np.float32)
    aff = np.asarray(vol.affine)
    sp = np.asarray(vol.spacing, float)
    # voxel axis that runs left-right in world space
    lr = int(np.argmax(np.abs(aff[:3, :3])[0]))
    sl = [slice(None)] * 3
    sl[lr] = a.shape[lr] // 2
    s = a[tuple(sl)]
    m = None if mask is None else np.asarray(mask)[tuple(sl)]
    rest = [i for i in range(3) if i != lr]
    ap_ax = 0 if abs(aff[1, rest[0]]) > abs(aff[1, rest[1]]) else 1
    if ap_ax == 1:
        s = s.T
        m = None if m is None else m.T
    # now rows = A-P, cols = S-I; make rows S-I (superior up), cols A-P (anterior right)
    si = rest[1 - ap_ax]
    s = s.T
    m = None if m is None else m.T
    if aff[2, si] > 0:
        s, m = s[::-1], (None if m is None else m[::-1])
    if aff[1, rest[ap_ax]] < 0:
        s, m = s[:, ::-1], (None if m is None else m[:, ::-1])
    aspect = sp[si] / sp[rest[ap_ax]]  # mm per row / mm per column
    return s, m, aspect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--case", default=None)
    a = ap.parse_args()
    if a.case is None:
        ok = [r["id"] for r in read_jsonl(RESULTS / "runs" / "synthstrip__adaptive.jsonl")
              if r["status"] == "ok" and not r["report"]["escalated_to_block"]]
        a.case = next(i for pre in ("ixi_t1_", "fsm_t1_", "asl_t1_") for i in sorted(ok) if i.startswith(pre))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import radveil

    row = {r["id"]: r for r in read_manifest(RESULTS / "manifests" / "synthstrip.csv")}[a.case]
    path = str(Path(a.data_dir) / row["image"])
    vin = radveil.Volume(path)
    cols = [("input", vin, None)]
    for m in MODES:
        v = radveil.Volume(path)
        rep = radveil.deface_volume(v, mode=m, return_mask=True)
        cols.append((m + (" (escalated)" if rep.get("escalated_to_block") else ""), v, rep["mask"]))

    sp = vin.spacing
    fig, axs = plt.subplots(3, len(cols), figsize=(1.75 * len(cols), 5.4))
    for j, (name, v, mask) in enumerate(cols):
        for i, yaw in enumerate((0.0, 40.0)):
            axs[i, j].imshow(v.render(yaw_deg=yaw, size=384)["rgb"])
        s, m, aspect = sagittal(v, mask)
        lo, hi = np.percentile(s, (1, 99.5))
        axs[2, j].imshow(s, cmap="gray", vmin=lo, vmax=hi, aspect=aspect)
        if m is not None:
            axs[2, j].contour(m.astype(float), levels=[0.5], colors="#f59e0b", linewidths=1.2)
        axs[0, j].set_title(name, fontsize=8)
        for i in range(3):
            axs[i, j].set_axis_off()
    fig.subplots_adjust(wspace=0.02, hspace=0.04, left=0.045, right=0.99, top=0.95, bottom=0.01)
    for i, lab in enumerate(("surface render, frontal", "surface render, 40° yaw", "mid-sagittal slice")):
        box = axs[i, 0].get_position()
        fig.text(0.015, (box.y0 + box.y1) / 2, lab, rotation=90, va="center", ha="center", fontsize=7)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_modes.png", dpi=220)
    print("wrote", FIG / "fig_modes.png", "case", a.case, "spacing", sp)


if __name__ == "__main__":
    main()
