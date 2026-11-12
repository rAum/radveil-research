#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Face-recognition gallery: SFace and ArcFace embeddings of every
*unmodified* input in a manifest, at the recognition views and iso-surfaces
of bench/attack.py.

    python paper/bench/gallery.py --dataset ixi --data-dir $DATA_DIR --jobs 8

Writes paper/results/embeddings/<dataset>__gallery.npz with ``ids`` and one
array per recognizer (n, len(RECOG_MULT), len(RECOG_YAWS), dim; NaN where
YuNet finds no face). It covers refused scans too: they are released
unmodified, so they enter the release-risk analysis at their clean rate. The
re-identification analysis (analysis/reid.py) matches defaced outputs
against this gallery, and also matches unmodified scans of one contrast
against another as the positive control showing that renders are
recognisable at all.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, read_manifest  # noqa: E402

_A = None


def _init(a):
    global _A
    _A = a
    os.environ["RAYON_NUM_THREADS"] = "2"


def embed(row: dict) -> dict:
    import attack
    import radveil

    try:
        e, _ = attack.embed_volume(radveil.Volume(str(Path(_A.data_dir) / row["image"])), _A.models)
        return e
    except Exception as ex:  # unreadable volume: no gallery entry
        print(f"{row['id']}: {ex}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    import attack

    a.models = str(Path(a.data_dir) / "models")
    rows = read_manifest(RESULTS / "manifests" / f"{a.dataset}.csv")
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a,)) as ex:
        embs = list(ex.map(embed, rows, chunksize=4))
    out = RESULTS / "embeddings" / f"{a.dataset}__gallery.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for r in attack.RECOGNIZERS:
        dim = 512 if r == "arcface" else 128
        blank = np.full((len(attack.RECOG_MULT), len(attack.RECOG_YAWS), dim), np.nan, np.float32)
        arrays[r] = np.stack([e[r] if e is not None else blank for e in embs]).astype(np.float16)
    np.savez_compressed(out, ids=np.array([r["id"] for r in rows]), **arrays)
    found = np.isfinite(arrays["sface"][:, 1, :, 0])
    print(f"{out.name}: {len(rows)} volumes, face found at 1x in {found.any(1).sum()}")


if __name__ == "__main__":
    main()
