# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Recognizer-free 3D shape linkage attack: head-shape descriptors.

An attacker holding a clean scan of each candidate links an anonymized scan
to its subject by the shape of the head, with no 2D face recognizer.

  1. Iso-surface: voxels above ``threshold_mult`` x the base threshold, the
     largest connected component (base = 0.14 x the 99th intensity
     percentile, RadVeil's own convention; ``threshold`` overrides it).
     ``threshold_mult=2`` gives the deeper surface below RadVeil's
     skin-deep anonymized layer.
  2. Frame: the scanner's RAS world frame (via the affine, so storage
     orientation, voxel size and slice thickness do not matter), with the
     origin at the centroid of the top 60 mm of the head surface: the
     vertex, which a face defacer never touches. Per-scan alignment to a
     template was tried and dropped: a head is close to a sphere, so ICP
     against a population template drifts by degrees.
  3. Descriptor:
       rmap    radius of the outermost surface crossing on a 4-degree ray
               grid (elevation -50..+90, all azimuths; 3,240 cells), refined
               to 0.05 mm by trilinear interpolation; NaN where the ray
               leaves the field of view or the surface touches the volume
               edge, so a cut field of view never fakes a surface;
       points  up to 2,000 surface points relative to the origin, for the
               pairwise rigid ICP that analysis/shape_reid.py runs to
               re-rank the best coarse candidates.

A descriptor is ~30 KB (radial map float32, points float16), stored per case
like the face embeddings.
"""

from __future__ import annotations

import time

import numpy as np

STEP_DEG = 4.0
ELEV = np.arange(-50.0, 90.0 + 1e-6, STEP_DEG)
AZIM = np.arange(0.0, 360.0, STEP_DEG)
R_MIN, R_MAX, R_STEP = 20.0, 200.0, 0.75
N_POINTS = 2000
TOP_MM = 60.0
BASE_FRAC = 0.14  # x p99, RadVeil's surface threshold for MRI


def directions() -> np.ndarray:
    """Unit ray directions (len(ELEV), len(AZIM), 3), RAS: +x right,
    +y anterior, +z superior; azimuth 0 = anterior, 90 = right."""
    el, az = np.meshgrid(np.radians(ELEV), np.radians(AZIM), indexing="ij")
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], -1)


def in_face(q: np.ndarray) -> np.ndarray:
    """Points (relative to the origin) in the anterior-inferior sector a
    face-only defacer rewrites: within 75 deg of anterior, below 10 deg of
    elevation. Checked against the cells RadVeil's adaptive mode changes."""
    el = np.degrees(np.arctan2(q[..., 2], np.hypot(q[..., 0], q[..., 1])))
    az = np.degrees(np.arctan2(q[..., 0], q[..., 1]))
    return (np.abs(az) <= 75.0) & (el <= 10.0)


def face_sector() -> np.ndarray:
    return in_face(directions())


def _load(path: str):
    import nibabel as nib

    img = nib.squeeze_image(nib.load(str(path)))
    data = np.asarray(img.dataobj, np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine


def base_threshold(data: np.ndarray) -> float:
    return BASE_FRAC * float(np.percentile(data, 99))


def _mask(data: np.ndarray, thr: float) -> np.ndarray:
    from scipy import ndimage

    m = data > thr
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == np.argmax(sizes)
    return m


def _surface(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    ijk = np.argwhere(mask & ~ndimage.binary_erosion(mask))
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def _radial(mask, data, thr, affine, origin) -> np.ndarray:
    from scipy import ndimage

    dirs = directions().reshape(-1, 3)
    radii = np.arange(R_MIN, R_MAX, R_STEP)
    inv = np.linalg.inv(affine)
    to_vox = lambda w: w @ inv[:3, :3].T + inv[:3, 3]
    ijk = to_vox(origin + dirs[:, None, :] * radii[None, :, None])
    idx = np.rint(ijk).astype(np.int64)
    shp = np.array(mask.shape)
    inside = np.all((idx >= 0) & (idx < shp), -1)
    cl = np.clip(idx, 0, shp - 1)
    hit = mask[cl[..., 0], cl[..., 1], cl[..., 2]] & inside
    last = len(radii) - 1 - np.argmax(hit[:, ::-1], axis=1)
    out = np.where(hit.any(1), radii[last], np.nan)
    rows = np.arange(len(dirs))
    nxt = np.minimum(last + 1, len(radii) - 1)
    saw_air = inside[rows, nxt] & (last + 1 < len(radii))
    sv = cl[rows, last]
    edge = np.any((sv < 2) | (sv > shp - 3), -1)
    out[~saw_air | edge] = np.nan

    # refine to the trilinear threshold crossing at 0.05 mm
    ok = np.isfinite(out)
    fine = np.arange(-R_STEP, R_STEP + 1e-6, 0.05)
    rr = out[ok][:, None] + fine[None, :]
    vk = to_vox(origin + dirs[ok][:, None, :] * rr[..., None])
    val = ndimage.map_coordinates(data, vk.reshape(-1, 3).T, order=1, mode="nearest").reshape(rr.shape)
    above = val > thr
    k = above.shape[1] - 1 - np.argmax(above[:, ::-1], axis=1)
    has = above.any(1) & (k < above.shape[1] - 1)
    i = np.arange(len(k))
    v0, v1 = val[i, k], val[i, np.minimum(k + 1, above.shape[1] - 1)]
    frac = np.clip((v0 - thr) / np.where(v0 == v1, 1, v0 - v1), 0, 1)
    sub = out[ok]
    sub[has] = (rr[i, k] + frac * 0.05)[has]
    out[ok] = sub
    return out.reshape(len(ELEV), len(AZIM))


def descriptor(path: str, threshold_mult: float = 1.0, threshold: float | None = None) -> dict:
    """Shape descriptor of one volume. ``threshold`` is the base iso-value
    (default 0.14 x p99); the surface is taken at ``threshold_mult`` x base.
    Describe an anonymized output with its clean input's base threshold,
    as the render probe does."""
    t0 = time.perf_counter()
    with np.errstate(all="ignore"):  # spurious Accelerate matmul warnings on macOS
        data, affine = _load(path)
        base = base_threshold(data) if threshold is None else float(threshold)
        thr = base * threshold_mult
        mask = _mask(data, thr)
        surf = _surface(mask, affine)
        origin = surf[surf[:, 2] > surf[:, 2].max() - TOP_MM].mean(0)
        rmap = _radial(mask, data, thr, affine, origin)
        # deterministic, spatially even subsample: one point per 4 mm cell
        rel = surf - origin
        _, first = np.unique(np.floor(rel / 4.0).astype(np.int32), axis=0, return_index=True)
        pts = rel[np.sort(first)]
        if len(pts) > N_POINTS:
            pts = pts[np.linspace(0, len(pts) - 1, N_POINTS).astype(int)]
    return {
        "rmap": rmap.astype(np.float32), "points": pts.astype(np.float16),
        "origin": origin.astype(np.float32), "threshold": np.float32(thr),
        "threshold_mult": np.float32(threshold_mult),
        "valid_frac": np.float32(np.isfinite(rmap).mean()),
        "seconds": np.float32(time.perf_counter() - t0),
    }


if __name__ == "__main__":
    import sys

    d = descriptor(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
    print({k: (v if np.ndim(v) == 0 else v.shape) for k, v in d.items()})
