# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the paper benchmarks: configurations, environment
capture, timed CLI runs and the voxel-level metrics."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
RESULTS = PAPER / "results"

# Every configuration the paper reports, as `radveil auto` flags. The name is
# the run's identity: results/runs/<dataset>__<config>.jsonl.
CONFIGS: dict[str, list[str]] = {
    "adaptive": ["--mode", "adaptive"],
    "adaptive_noverify": ["--mode", "adaptive", "--no-verify"],
    "adaptive_ears": ["--mode", "adaptive", "--region", "face+ears", "--ear-mode", "cut"],
    "adaptive_guard": ["--mode", "adaptive", "--brain-guard"],
    "adaptive_robust": ["--mode", "adaptive", "--partial-face", "--auto-threshold"],
    "blur": ["--mode", "blur"],
    "cut": ["--mode", "cut"],
    "verify_fast": ["--mode", "adaptive", "--verify-effort", "fast"],
    "verify_thorough": ["--mode", "adaptive", "--verify-effort", "thorough"],
}

# Messages with which `radveil auto` refuses a volume on purpose. Any other
# non-zero exit (a panic, a signal, a missing file) is an error, never a
# refusal: errors are retried on resume and reported separately.
REFUSAL_MARKERS = {
    "no plausible face": "no face found",
    "did not survive validation": "candidates failed 3D consensus",
    "not seen through": "face cut by the field of view",
    "field of view": "face cut by the field of view",
}

# SynthSeg / FreeSurfer label groups (seg_labels.txt in the SynthStrip set).
# Labels 100-105 are synthetic non-brain classes and are not brain;
# "brain" for distances is every group except ventricles/CSF, the same
# definition the "touched" counts use.
LABEL_GROUPS: dict[str, tuple[int, ...]] = {
    "cerebral_cortex": (3, 42),
    "cerebral_wm": (2, 41),
    "cerebellum": (7, 8, 46, 47),
    "brainstem": (16,),
    "deep_gm": (10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60),
    "ventricles_csf": (4, 5, 14, 15, 24, 43, 44),
}


def cli_path() -> str:
    """The radveil CLI built from this checkout (`make env`), or $RADVEIL_CLI."""
    p = os.environ.get("RADVEIL_CLI") or str(REPO / "target" / "release" / "radveil")
    if not Path(p).exists():
        sys.exit(f"radveil CLI not found at {p}: run `make -C paper env` or set RADVEIL_CLI")
    return p


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


BRAIN_LABELS = tuple(i for g, ids in LABEL_GROUPS.items() if g != "ventricles_csf" for i in ids)


def classify_exit(code: int, stderr: str) -> tuple[str, str]:
    """(status, reason) for a failed `radveil auto` run."""
    for marker, reason in REFUSAL_MARKERS.items():
        if marker in (stderr or ""):
            return "refused", reason
    return "error", f"exit {code}"


def git_rev(paths: tuple[str, ...] = ("crates", "python", "Cargo.lock", "Cargo.toml", "pyproject.toml",
                                      "paper/bench", "paper/scripts")) -> str:
    """HEAD, suffixed -dirty when anything that affects results is modified."""
    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *paths],
                               capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def sha256_file(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def environment() -> dict:
    """Hardware and software the numbers were measured on."""
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "git": git_rev(),
        # the code itself, independent of commit history (survives squashes)
        "git_tree": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD^{tree}"],
                                   capture_output=True, text=True).stdout.strip(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if sys.platform == "darwin":
        q = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        env["cpu"] = q("machdep.cpu.brand_string")
        env["mem_gb"] = round(int(q("hw.memsize") or 0) / 2**30, 1)
        env["perf_cores"] = q("hw.perflevel0.physicalcpu")
    else:
        try:
            info = Path("/proc/cpuinfo").read_text()
            env["cpu"] = re.search(r"model name\s*:\s*(.*)", info).group(1)
            env["mem_gb"] = round(int(re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text()).group(1)) / 2**20, 1)
        except Exception:
            pass
    try:
        cli = cli_path()
        env["radveil_cli"] = subprocess.run([cli, "--version"], capture_output=True, text=True).stdout.strip()
        env["radveil_cli_sha256"] = sha256_file(cli)
    except SystemExit:
        pass
    try:
        import radveil._radveil as ext

        env["radveil_py_sha256"] = sha256_file(ext.__file__)
    except ImportError:
        pass
    try:
        env["rustc"] = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        pass
    return env


def _time_cmd() -> list[str]:
    t = shutil.which("gtime") or "/usr/bin/time"
    if not Path(t).exists():
        return []
    return [t, "-l"] if sys.platform == "darwin" and "gtime" not in t else [t, "-v"]


def run_timed(cmd: list[str], env: dict | None = None, timeout: int = 1800) -> dict:
    """Run a command; return exit code, wall seconds, peak RSS (MB) and stderr tail."""
    tc = _time_cmd()
    t0 = time.perf_counter()
    r = subprocess.run(tc + cmd, capture_output=True, text=True, env=env, timeout=timeout)
    wall = time.perf_counter() - t0
    rss = None
    m = re.search(r"(\d+)\s+maximum resident set size", r.stderr)  # macOS: bytes
    if m:
        rss = int(m.group(1)) / 2**20
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", r.stderr)  # GNU: KiB
    if m:
        rss = int(m.group(1)) / 2**10
    # drop the time(1) statistics (BSD: "  <number>  <label>", GNU: "\t<label>: <n>")
    tail = [ln for ln in r.stderr.strip().splitlines()
            if ln.strip() and not re.match(r"^\s*[\d.]+\s", ln) and not ln.startswith("\t")]
    return {"exit": r.returncode, "wall_s": round(wall, 3),
            "peak_rss_mb": None if rss is None else round(rss, 1),
            "stderr": "\n".join(tail[-8:])[-2000:]}


# ------------------------------------------------------------------ metrics


def load_raw(path):
    import nibabel as nib

    img = nib.load(str(path))
    return img, np.squeeze(np.asarray(img.dataobj.get_unscaled()))


def integrity(src, out, mask: np.ndarray, mask_affine=None) -> dict:
    """Bit-exactness outside the anonymization mask, header preservation, and
    how much *tissue* the region changed (the mask is mostly air in front of
    the face, so its volume alone overstates the edit).

    Tissue is the input above 10% of its 99th intensity percentile, a loose
    head mask; extent is the region's bounding box in mm."""
    a_img, a = load_raw(src)
    b_img, b = load_raw(out)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(f"expected 3D volumes, got {a.shape} and {b.shape}")
    if mask_affine is not None and not np.allclose(mask_affine, a_img.affine, atol=1e-3):
        raise ValueError("region mask is not on the input grid (affine differs)")
    same_shape = a.shape == b.shape == mask.shape
    changed = (a != b) & ~(np.isnan(a) & np.isnan(b)) if same_shape else None
    spacing = np.asarray(a_img.header.get_zooms()[:3], float)
    vox_ml = float(np.prod(spacing)) / 1000.0
    tissue = a > 0.1 * np.percentile(a, 99)
    ijk = np.argwhere(mask)
    return {
        "bit_exact_outside": bool(same_shape and not changed[~mask].any()),
        "dtype_preserved": a_img.get_data_dtype() == b_img.get_data_dtype(),
        "affine_preserved": bool(np.allclose(a_img.affine, b_img.affine)),
        "scaling_preserved": [float(x) if x is not None and np.isfinite(x) else None
                              for x in a_img.header.get_slope_inter()]
        == [float(x) if x is not None and np.isfinite(x) else None
            for x in b_img.header.get_slope_inter()],
        "changed_inside": int(changed[mask].sum()) if same_shape else None,
        "changed_tissue_ml": round(float((changed & tissue).sum()) * vox_ml, 2) if same_shape else None,
        "changed_ml": round(float(changed.sum()) * vox_ml, 2) if same_shape else None,
        "region_extent_mm": (((ijk.max(0) - ijk.min(0) + 1) * spacing).round(1).tolist() if len(ijk) else None),
        "input_has_scaling": a_img.header.get_slope_inter() not in ((None, None), (1.0, 0.0)),
    }


def brain_metrics(mask: np.ndarray, brain: np.ndarray, spacing, labels: np.ndarray | None = None) -> dict:
    """How close the anonymization region came to the brain.

    ``brain`` is a binary reference (expert mask, or SynthStrip's mask for
    IXI). Distance is from the nearest anonymized voxel to the nearest brain
    voxel, in mm (0 when they overlap). With a SynthSeg ``labels`` map, also
    the anonymized voxels per anatomical group and the distance to the
    nearest labelled brain structure."""
    from scipy import ndimage

    vox_ml = float(np.prod(spacing)) / 1000.0
    out = {"mask_voxels": int(mask.sum()), "mask_ml": round(mask.sum() * vox_ml, 2),
           "brain_ml": round(brain.sum() * vox_ml, 1)}
    ov = mask & brain
    out["brain_overlap_voxels"] = int(ov.sum())
    out["brain_overlap_ml"] = round(ov.sum() * vox_ml, 4)
    if not mask.any() or not brain.any():
        out["min_dist_mm"] = None
    elif ov.any():
        out["min_dist_mm"] = 0.0
    else:
        out["min_dist_mm"] = round(float(ndimage.distance_transform_edt(~brain, sampling=spacing)[mask].min()), 2)
    if labels is not None and labels.shape == mask.shape:
        lab_in = labels[mask]
        out["label_overlap_voxels"] = {g: int(np.isin(lab_in, ids).sum()) for g, ids in LABEL_GROUPS.items()}
        seg_brain = np.isin(labels, BRAIN_LABELS)
        if mask.any() and seg_brain.any():
            d = ndimage.distance_transform_edt(~seg_brain, sampling=spacing)[mask]
            out["seg_min_dist_mm"] = round(float(d.min()), 2)
            # per-structure closest approach, only the groups the paper discusses
            for g in ("cerebral_cortex", "cerebellum", "brainstem"):
                s = np.isin(labels, LABEL_GROUPS[g])
                if s.any():
                    out[f"seg_min_dist_{g}_mm"] = round(
                        float(ndimage.distance_transform_edt(~s, sampling=spacing)[mask].min()), 2)
    return out


def write_atomic(path: Path, text: str) -> None:
    """Replace a file so that a crash leaves either the old or the new one."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path, repair: bool = False) -> list[dict]:
    """Records of a run. A line cut short by a crash is skipped (and with
    ``repair`` removed from the file, so the next append starts clean); for a
    case recorded twice the last record wins."""
    if not Path(path).exists():
        return []
    good, recs = [], {}
    for ln in Path(path).read_text().splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        good.append(ln)
        recs[r.get("id", len(recs))] = r
    if repair:
        write_atomic(Path(path), "".join(x + "\n" for x in good))
    return list(recs.values())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
