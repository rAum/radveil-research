# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Matching for the 3D shape linkage attack (bench/shape.py descriptors).

Two stages, as a practical attacker would run them:

  coarse   radial maps compared over the cells valid in both: negative
           median absolute radius difference (mm) after removing the median
           offset (iso-surfaces of different contrasts sit a fixed distance
           apart). Ranks the whole gallery cheaply.
  refine   for the top ``k`` coarse candidates, trimmed rigid ICP of the
           probe's surface points onto the candidate's (both in their RAS
           frame, origins at the vertex); score = negative trimmed RMS
           residual (mm). Re-ranks those candidates; the rest keep their
           coarse rank below them.

``exclude_face=True`` drops the anterior-inferior sector a face-only
defacer rewrites (bench/shape.py:in_face) from both stages, so the match
uses only the head the defacer left: forehead, scalp, ears, back of head.

evaluate() mirrors analysis/reid.py: closed-set rank-1 over the gallery,
genuine and impostor scores (refined where computed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from shape import face_sector, in_face  # noqa: E402

MIN_CELLS = 300
_FACE = face_sector()


def similarity(a: dict, b: dict, exclude_face: bool = False) -> float:
    """Coarse radial-map similarity (higher = more alike), -inf if fewer than
    MIN_CELLS cells are comparable."""
    ra = np.asarray(a["rmap"], np.float32)
    rb = np.asarray(b["rmap"], np.float32)
    ok = np.isfinite(ra) & np.isfinite(rb)
    if exclude_face:
        ok &= ~_FACE
    if ok.sum() < MIN_CELLS:
        return -np.inf
    d = ra[ok] - rb[ok]
    return -float(np.median(np.abs(d - np.median(d))))


def _kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


_TREES: dict = {}


def refine(a: dict, b: dict, exclude_face: bool = False, iters: int = 30, keep: float = 0.8,
           bkey: str | None = None) -> float:
    """Negative trimmed RMS (mm) of probe a's surface after rigid ICP onto b's.
    ``bkey`` names the gallery entry for the KD-tree cache (a stable id, so a
    rebuilt gallery can never reuse another scan's tree)."""
    from scipy.spatial import cKDTree

    pa = np.asarray(a["points"], np.float64)
    pb = np.asarray(b["points"], np.float64)
    if exclude_face:
        pa, pb = pa[~in_face(pa)], pb[~in_face(pb)]
    key = (bkey if bkey is not None else id(b), exclude_face)
    if key not in _TREES:
        _TREES[key] = cKDTree(pb)
    tree = _TREES[key]
    r, t = np.eye(3), np.zeros(3)
    rms = np.inf
    with np.errstate(all="ignore"):
        for _ in range(iters):
            q = pa @ r.T + t
            d, j = tree.query(q)
            sel = d <= np.quantile(d, keep)
            new = float(np.sqrt(np.mean(d[sel] ** 2)))
            r, t = _kabsch(pa[sel], pb[j[sel]])
            if abs(rms - new) < 1e-3:
                break
            rms = new
    return -rms


def perturb(d: dict, deg: float, mm: float, seed: str) -> dict:
    """The probe's surface points under a random rigid motion (rotation of
    ``deg`` degrees about a random axis, translation of ``mm``), seeded by the
    probe id. Used to remove same-session head pose from the comparison: the
    radial map, which is pose-dependent, is then not used for matching."""
    import hashlib

    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    th = np.radians(deg)
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    r = np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)
    t = rng.normal(size=3)
    t *= mm / np.linalg.norm(t)
    with np.errstate(all="ignore"):  # spurious Accelerate matmul warnings on macOS
        return {**d, "points": (np.asarray(d["points"], np.float64) @ r.T + t)}


def evaluate(gallery: dict, probes: dict, subj: dict, exclude_face: bool = False, k: int | None = 10,
             pose: tuple[float, float] | None = None) -> dict:
    """gallery: id -> descriptor (one per subject); probes: id -> descriptor;
    subj: id -> subject key. ``k`` coarse candidates are refined by ICP
    (None = all). Probes whose subject is not in the gallery are skipped.
    With ``pose`` = (degrees, mm) each probe is first moved by a random rigid
    motion and every candidate is refined (the coarse stage only orders)."""
    if pose is not None:
        k = None
    gids = sorted(gallery)
    gsub = [subj[g] for g in gids]
    hits, gen, imp, n, ranks = 0, [], [], 0, []
    for pid in sorted(probes):
        if subj[pid] not in gsub:
            continue
        p = probes[pid] if pose is None else perturb(probes[pid], pose[0], pose[1], pid)
        coarse = np.array([similarity(probes[pid], gallery[g], exclude_face) for g in gids])
        order = np.argsort(-coarse)
        top = order if k is None else order[:k]
        fine = {int(i): refine(p, gallery[gids[i]], exclude_face, bkey=gids[i]) for i in top}
        # refined candidates first (by ICP), then the rest by coarse score
        ranked = sorted(fine, key=lambda i: -fine[i]) + [int(i) for i in order if int(i) not in fine]
        g = gsub.index(subj[pid])
        n += 1
        rank = ranked.index(g) + 1
        ranks.append(rank)
        hits += int(rank == 1)
        if g in fine:
            gen.append(fine[g])
        imp.extend(v for i, v in fine.items() if i != g)
    return {"n": n, "gallery": len(gids), "rank1": hits, "ranks": ranks, "genuine": gen, "impostor": imp}
