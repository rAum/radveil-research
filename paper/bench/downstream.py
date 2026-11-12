# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Downstream effect of anonymization, and validation of RadVeil's SynthStrip.

  compare(src, out, tmp)        run the *original* SynthStrip (Hoopes et al.
                                2022; freesurfer/synthstrip:1.6, pinned) on the
                                input and on the anonymized output and compare
                                the two brain masks: Dice, volume change, 95th
                                percentile Hausdorff distance. Defacing that
                                changes skull-stripping changes every analysis
                                built on it (de Sitter et al. 2020).
  validate_rust(src, tmp, cli)  Dice etc. between `radveil deskull` (RadVeil's
                                Rust reimplementation, which --brain-check and
                                --brain-guard rely on) and the original
                                SynthStrip on the same input.

  segment(src, out, tmp, contrast)
                                tissue segmentation downstream of skull
                                stripping: each volume is stripped with its own
                                original-SynthStrip mask and segmented with FSL
                                FAST (3 classes; native FSL conda build, pinned
                                by scripts/fetch_containers.sh); per-class Dice
                                and volume change between input and output.

SynthSeg/FreeSurfer segmentation is not included (FreeSurfer licence); FAST is
the segmentation-level check.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from baselines import synthstrip_mask


def synthstrip(path: Path, mask_out: Path) -> dict:
    """Original SynthStrip brain mask of `path` into `mask_out`; returns timing."""
    return synthstrip_mask(Path(path), Path(mask_out))


def _load_mask(p: Path):
    import nibabel as nib

    img = nib.load(str(p))
    return np.squeeze(np.asarray(img.dataobj)) > 0, tuple(float(z) for z in img.header.get_zooms()[:3])


def _same_grid(a: Path, b: Path) -> bool:
    import nibabel as nib

    x, y = nib.load(str(a)), nib.load(str(b))
    return x.shape[:3] == y.shape[:3] and bool(np.allclose(x.affine, y.affine, atol=1e-3))


def mask_agreement(a: np.ndarray, b: np.ndarray, spacing) -> dict:
    """Dice, volumes and symmetric 95th-percentile surface distance (mm)."""
    from scipy import ndimage

    vox_ml = float(np.prod(spacing)) / 1000.0
    va, vb = a.sum() * vox_ml, b.sum() * vox_ml
    inter = np.logical_and(a, b).sum()
    out = {"dice": round(2.0 * inter / max(1, a.sum() + b.sum()), 5),
           "vol_a_ml": round(va, 2), "vol_b_ml": round(vb, 2),
           "vol_diff_ml": round(vb - va, 3), "vol_diff_pct": round(100.0 * (vb - va) / va, 3) if va else None}
    if a.any() and b.any():
        sa = a & ~ndimage.binary_erosion(a)
        sb = b & ~ndimage.binary_erosion(b)
        da = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
        db = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
        out["hd95_mm"] = round(float(np.percentile(np.concatenate([da, db]), 95)), 3)
        out["mean_surface_mm"] = round(float(np.concatenate([da, db]).mean()), 3)
    else:
        out["hd95_mm"] = out["mean_surface_mm"] = None
    return out


def compare(src: Path, out: Path, tmp: Path) -> dict:
    """Does anonymization change the original SynthStrip's brain mask?
    Masks of input (a) and output (b) on the same grid."""
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    from baselines import input_brain_mask

    mb = tmp / "ss_out.nii.gz"
    ma, ra = input_brain_mask(Path(src))  # cached per input
    rb = synthstrip(out, mb)
    if ra["exit"] or rb["exit"] or not ma.exists() or not mb.exists():
        return {"error": (ra["stderr"] or rb["stderr"])[:300]}
    if not _same_grid(ma, mb):
        return {"error": "input and output masks are on different grids"}
    a, sp = _load_mask(ma)
    b, _ = _load_mask(mb)
    return {**mask_agreement(a, b, sp), "synthstrip_s": round(ra["wall_s"] + rb["wall_s"], 2)}


FSL = Path(os.environ.get("RADVEIL_FSL", Path(os.environ.get("DATA_DIR", ".")) / "containers" / "fsl" / "env"))
FAST_TYPE = {"T1": 1, "T2": 2, "PD": 3, "t1": 1, "t2": 2, "pd": 3}


def _fast(img_path: Path, mask_path: Path, work: Path, contrast: str) -> Path:
    """FSL FAST (3 classes) on img x mask; returns the label map path."""
    import nibabel as nib

    work.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(img_path))
    m = np.squeeze(np.asarray(nib.load(str(mask_path)).dataobj)) > 0
    nib.save(nib.Nifti1Image(np.squeeze(np.asarray(img.dataobj, np.float32)) * m, img.affine), str(work / "brain.nii.gz"))
    env = dict(os.environ, FSLDIR=str(FSL), FSLOUTPUTTYPE="NIFTI_GZ", OMP_NUM_THREADS="1")
    r = subprocess.run([str(FSL / "bin" / "fast"), "-t", str(FAST_TYPE.get(contrast, 1)), "-n", "3",
                        "-o", str(work / "fast"), str(work / "brain.nii.gz")], capture_output=True, text=True, env=env)
    seg = work / "fast_seg.nii.gz"
    if r.returncode or not seg.exists():
        raise RuntimeError(f"fast failed: {r.stderr[-200:]}")
    return seg


def segment(src: Path, out: Path, tmp: Path, contrast: str) -> dict:
    """Per-class agreement of FAST tissue segmentations of input and output,
    each stripped with its own original-SynthStrip mask. The input's
    segmentation is cached next to its mask."""
    import nibabel as nib
    from baselines import input_brain_mask

    tmp = Path(tmp)
    ma, ra = input_brain_mask(Path(src))
    mb = tmp / "ss_out.nii.gz"
    if not mb.exists():
        rb = synthstrip(out, mb)
        if rb["exit"]:
            return {"error": rb["stderr"][-300:]}
    if ra["exit"] or not _same_grid(ma, mb):
        return {"error": "missing mask or different grids"}
    t0 = time.perf_counter()
    cached = ma.with_name(ma.name.replace(".nii.gz", f".fast{FAST_TYPE.get(contrast, 1)}.nii.gz"))
    if not cached.exists():
        seg = _fast(Path(src), ma, tmp / "fast_in", contrast)
        shutil.copy(seg, cached.with_suffix(".tmp.gz"))
        os.replace(cached.with_suffix(".tmp.gz"), cached)
    sb = _fast(Path(out), mb, tmp / "fast_out", contrast)
    a = np.squeeze(np.asarray(nib.load(str(cached)).dataobj)).astype(int)
    b = np.squeeze(np.asarray(nib.load(str(sb)).dataobj)).astype(int)
    vox_ml = float(np.prod(nib.load(str(cached)).header.get_zooms()[:3])) / 1000.0
    res = {"fast_s": round(time.perf_counter() - t0, 1)}
    for k in (1, 2, 3):
        x, y = a == k, b == k
        res[f"class{k}_dice"] = round(2.0 * np.sum(x & y) / max(1, x.sum() + y.sum()), 5)
        res[f"class{k}_vol_diff_pct"] = round(100.0 * (y.sum() - x.sum()) / max(1, x.sum()), 3)
        res[f"class{k}_ml"] = round(float(x.sum()) * vox_ml, 1)
    return res


def validate_rust(src: Path, tmp: Path, cli: str) -> dict:
    """`radveil deskull` mask (b) vs the original SynthStrip (a) on one input."""
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    from baselines import input_brain_mask

    mb = tmp / "rv_deskull.nii.gz"
    ma, ra = input_brain_mask(Path(src))  # cached per input
    t0 = time.perf_counter()
    r = subprocess.run([cli, "deskull", str(src), "--mask-out", str(mb)], capture_output=True, text=True)
    t_rust = time.perf_counter() - t0
    if ra["exit"] or r.returncode or not ma.exists() or not mb.exists():
        return {"error": (ra["stderr"] or r.stderr)[-300:]}
    a, sp = _load_mask(ma)
    b, _ = _load_mask(mb)
    if a.shape != b.shape:
        return {"error": f"shape {a.shape} != {b.shape}"}
    return {**mask_agreement(a, b, sp), "synthstrip_s": ra["wall_s"], "radveil_deskull_s": round(t_rust, 2)}


if __name__ == "__main__":
    import argparse
    import json
    import tempfile

    ap = argparse.ArgumentParser(description="downstream skull-stripping comparison for one case")
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", help="anonymized volume (omit with --rust)")
    ap.add_argument("--rust", metavar="CLI", help="validate `CLI deskull` against SynthStrip instead")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as t:
        r = validate_rust(Path(a.input), Path(t), a.rust) if a.rust else compare(Path(a.input), Path(a.output), Path(t))
    print(json.dumps(r, indent=2))
