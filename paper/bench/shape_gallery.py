#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""3D shape descriptors (bench/shape.py) of every *unmodified* input in a
manifest, at the 1x and 2x surface levels.

    python paper/bench/shape_gallery.py --dataset ixi --data-dir $DATA_DIR

Writes results/shape/<dataset>__gallery.npz, keys "<id>__x<m>__<field>".
Like the face embeddings these are biometric templates: results/shape/ is
git-ignored and only the analysis summary (results/shape_summary.json) is
committed.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, read_manifest  # noqa: E402

_DATA = None


def _init(data):
    global _DATA
    _DATA = data


def describe(row: dict) -> dict:
    import shape

    out = {}
    try:
        p = str(Path(_DATA) / row["image"])
        base = float(shape.descriptor(p)["threshold"])
        for m in (1, 2):
            for k, v in shape.descriptor(p, float(m), threshold=base).items():
                out[f"{row['id']}__x{m}__{k}"] = v
    except Exception as e:  # unreadable or degenerate volume: no gallery entry
        print(f"{row['id']}: {e}", file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    import numpy as np

    rows = read_manifest(RESULTS / "manifests" / f"{a.dataset}.csv")
    allv = {}
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.data_dir,)) as ex:
        for d in ex.map(describe, rows, chunksize=4):
            allv.update(d)
    out = RESULTS / "shape" / f"{a.dataset}__gallery.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **allv)
    print(f"{out.name}: {len({k.split('__')[0] for k in allv})} of {len(rows)} volumes")


if __name__ == "__main__":
    main()
