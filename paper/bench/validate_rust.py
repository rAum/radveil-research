#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate RadVeil's Rust SynthStrip (used by --brain-check / --brain-guard
and by the IXI brain-safety numbers) on the SynthStrip set.

    python paper/bench/validate_rust.py --data-dir $DATA_DIR [--jobs 8]

For every SynthStrip scan with an image: `radveil deskull` mask vs the
original SynthStrip (pinned container, bench/downstream.py), and both vs the
expert mask. Writes results/rust_synthstrip.jsonl (resumable).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, append_jsonl, cli_path, read_jsonl, read_manifest  # noqa: E402

_A = None


def _init(a):
    global _A
    _A = a


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * np.sum(a & b) / max(1, a.sum() + b.sum()))


def case(row: dict) -> dict:
    import downstream
    import nibabel as nib

    data = Path(_A.data_dir)
    rec = {"id": row["id"]}
    with tempfile.TemporaryDirectory() as t:
        r = downstream.validate_rust(data / row["image"], Path(t), _A.cli)
        if "error" in r:
            return {**rec, "error": r["error"]}
        from baselines import input_brain_mask

        load = lambda p: np.squeeze(np.asarray(nib.load(str(p)).dataobj)) > 0
        ref, rust = load(input_brain_mask(data / row["image"])[0]), load(Path(t) / "rv_deskull.nii.gz")
        expert = load(data / row["brain_mask"])
        rec.update({"dice_vs_reference": r["dice"], "hd95_mm": r.get("hd95_mm"),
                    "dice_vs_expert_rust": _dice(rust, expert) if rust.shape == expert.shape else None,
                    "dice_vs_expert_reference": _dice(ref, expert) if ref.shape == expert.shape else None,
                    "synthstrip_s": r.get("synthstrip_s"), "radveil_deskull_s": r.get("radveil_deskull_s")})
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    a.cli = cli_path()
    out = RESULTS / "rust_synthstrip.jsonl"
    done = {r["id"] for r in read_jsonl(out, repair=True) if "error" not in r}
    rows = [r for r in read_manifest(RESULTS / "manifests" / "synthstrip.csv") if r["id"] not in done]
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a,)) as ex:
        for i, f in enumerate(as_completed([ex.submit(case, r) for r in rows]), 1):
            append_jsonl(out, f.result())
            if i % 50 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    recs = sorted(read_jsonl(out), key=lambda r: r["id"])
    out.write_text("".join(__import__("json").dumps(r) + "\n" for r in recs))
    print("wrote", out)


if __name__ == "__main__":
    main()
