#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the case manifests the benchmarks run on.

    python paper/scripts/prepare_datasets.py --data-dir $DATA_DIR

Writes (paths relative to --data-dir, so the manifests are portable and
committed under paper/results/manifests/):

  ixi.csv          one row per IXI T1/T2/PD volume: subject, site, scanner,
                   contrast, sex, age (from IXI.xls), whether the subject was
                   used while developing RadVeil (see DEVELOPMENT), and the
                   SHA-256 of the file, so any copy can be checked against ours
                   (bench/run_cohort.py --verify-data).
  synthstrip.csv   one row per SynthStrip case with an image: subset,
                   modality, split (SynthStrip's own test/valid lists),
                   expert brain mask, SynthSeg label map if shipped.
  datasets.json    counts and the SHA-256 of every downloaded archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "manifests"

# IXI acquisition sites (https://brain-development.org/ixi-dataset/)
IXI_SITES = {
    "Guys": "Philips 1.5T (Guy's Hospital)",
    "HH": "Philips 3T (Hammersmith Hospital)",
    "IOP": "GE 1.5T (Institute of Psychiatry)",
}
# IXI subjects used while developing RadVeil: the 50 IXI subjects of the
# SynthStrip evaluation set (defaults, the brain-floor table) and the three
# extra subjects whose pinnae were hand-marked for the ear model
# (experiments/ears on the `experiments` branch). Every other IXI subject is
# held out; the paper reports both groups.
EXTRA_DEVELOPMENT = {"IXI014", "IXI033", "IXI035"}

IXI_NAME = re.compile(r"IXI(\d{3})-(Guys|HH|IOP)-(\d{4})-(T1|T2|PD)\.nii\.gz$")


def ixi_demographics(xls: Path) -> dict[int, dict]:
    import pandas as pd

    t = pd.read_excel(xls, sheet_name="Table", engine="xlrd")
    demo = {}
    for _, r in t.iterrows():
        sex = {1: "M", 2: "F"}.get(int(r["SEX_ID (1=m, 2=f)"]), "")
        age = "" if pd.isna(r["AGE"]) else round(float(r["AGE"]), 1)
        # IXI.xls lists a few subjects twice; keep the row that has an age
        if int(r["IXI_ID"]) in demo and age == "":
            continue
        demo[int(r["IXI_ID"])] = {"sex": sex, "age": age}
    return demo


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ixi_manifest(data: Path, development: set[str]) -> list[dict]:
    root = data / "ixi"
    if not root.exists():
        return []
    demo = ixi_demographics(root / "IXI.xls") if (root / "IXI.xls").exists() else {}
    rows = []
    for p in sorted(root.glob("*/IXI*.nii.gz")):
        m = IXI_NAME.search(p.name)
        if not m:
            continue
        sid, site, _, contrast = m.groups()
        d = demo.get(int(sid), {})
        rows.append({
            "id": f"ixi_{contrast.lower()}_{sid}",
            "dataset": "ixi",
            "subject": f"IXI{sid}",
            "site": site,
            "scanner": IXI_SITES[site],
            "contrast": contrast,
            "sex": d.get("sex", ""),
            "age": d.get("age", ""),
            "development": int(f"IXI{sid}" in development),
            "image": str(p.relative_to(data)),
            "sha256": _sha256(p),
        })
    return rows


def synthstrip_manifest(data: Path) -> list[dict]:
    root = data / "synthstrip" / "synthstrip_data_v1.5"
    if not root.exists():
        return []
    split = {}
    for s in ("test", "valid"):
        f = root / f"scans_{s}.txt"
        if f.exists():
            split.update({ln.strip(): s for ln in f.read_text().split() if ln.strip()})
    rows = []
    for case in sorted(p for p in root.iterdir() if p.is_dir()):
        img = case / "image.nii.gz"
        if not img.exists():  # CIM PET/CT: masks only, images not redistributable
            continue
        subset, modality = case.name.split("_")[:2]
        lab = case / "labels.nii.gz"
        rows.append({
            "id": case.name,
            "dataset": "synthstrip",
            "subset": subset,
            "modality": modality,
            "split": split.get(case.name, ""),
            "image": str(img.relative_to(data)),
            "brain_mask": str((case / "mask.nii.gz").relative_to(data)),
            "labels": str(lab.relative_to(data)) if lab.exists() else "",
            "sha256": _sha256(img),
            "brain_mask_sha256": _sha256(case / "mask.nii.gz"),
            "labels_sha256": _sha256(lab) if lab.exists() else "",
        })
    return rows


def write(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"skip {path.name}: dataset not found")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{path.name}: {len(rows)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="write manifests here instead")
    ap.add_argument("--check", action="store_true",
                    help="build the manifests in a temporary directory and fail if any row or checksum "
                         "differs from the committed ones (the data is not the data the paper used)")
    args = ap.parse_args()
    global OUT
    committed = OUT
    if args.check:
        import tempfile

        args.out = Path(tempfile.mkdtemp())
    OUT = args.out or OUT
    data = args.data_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    ss = synthstrip_manifest(data)
    development = EXTRA_DEVELOPMENT | {"IXI" + r["id"].rsplit("_", 1)[1] for r in ss if r["subset"] == "ixi"}
    ixi = ixi_manifest(data, development)
    write(ixi, OUT / "ixi.csv")
    write(ss, OUT / "synthstrip.csv")

    sums = {}
    for f in (data / "ixi" / "SHA256",):
        if f.exists():
            for ln in f.read_text().splitlines():
                h, name = ln.split(maxsplit=1)
                sums[name.strip()] = h
    summary = {
        "ixi": {
            "volumes": len(ixi),
            "subjects": len({r["subject"] for r in ixi}),
            "by_contrast": {c: sum(r["contrast"] == c for r in ixi) for c in ("T1", "T2", "PD")},
            "by_site": {s: sum(r["site"] == s for r in ixi) for s in IXI_SITES},
            "development_subjects": len({r["subject"] for r in ixi if r["development"]}),
        },
        "synthstrip": {
            "cases_with_image": len(ss),
            "with_labels": sum(bool(r["labels"]) for r in ss),
            "archive_sha256": "c9f070c40266790d97fce35b8864dfcd96f039b598311030c9d707671b3b0aba",
        },
        "archive_sha256": sums,
    }
    (OUT / "datasets.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.check:
        bad = []
        for name in ("ixi.csv", "synthstrip.csv"):
            a, b = committed / name, OUT / name
            if a.exists() and (not b.exists() or a.read_text() != b.read_text()):
                bad.append(name)
        if bad:
            raise SystemExit(f"data differ from the committed manifests: {', '.join(bad)} "
                             f"(diff {committed} {OUT}); `make manifests-update` only if that is intended")
        print("data match the committed manifests")
        return
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
