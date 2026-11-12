#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Run `radveil auto` (or a baseline defacer) over a dataset manifest and
record everything the paper reports about each case.

    python paper/bench/run_cohort.py --dataset synthstrip --config adaptive \
        --data-dir $DATA_DIR --jobs 4 --threads 3 [--attack] [--qa] [--brain-check] \
        [--downstream] [--shape]

Output: paper/results/runs/<dataset>__<config>.jsonl, one JSON line per
case, sorted by case id when the run finishes, plus <name>.meta.json with
the flags, environment and code revision of every session that wrote to it.

Resuming: cases recorded as ok or refused are skipped; cases recorded as
error (a crash, a signal, a failed metric) are run again. A resume with a
different code revision is refused unless --allow-mixed, so one run never
silently mixes two versions of the code. A dirty working tree is refused
unless --allow-dirty.

Per case: status (ok / refused with the tool's reason / error), the tool's
JSON report minus anything that encodes the original face (landmarks, the
face frame), wall time and peak RSS, integrity (bit-exact outside the region,
header, changed tissue volume, region extent), brain safety against the
expert mask and SynthSeg labels when the dataset has them, and optionally
the re-identification probe (bench/attack.py), the downstream skull-strip
comparison (bench/downstream.py), the 3D shape descriptors (bench/shape.py)
and a visual QA sheet (bench/qa.py).

Wall times from cohort runs share the machine with --jobs concurrent cases;
the per-volume latency the paper quotes comes from bench/profile.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONFIGS, RESULTS, append_jsonl, brain_metrics, classify_exit, cli_path,  # noqa: E402
                    environment, integrity, read_jsonl, read_manifest, run_timed, write_atomic)

# report keys that describe the original face geometry: never stored
FACE_GEOMETRY = ("landmarks_3d", "frame")

_ARGS = None


def baseline_names() -> list[str]:
    try:
        import baselines

        return sorted(baselines.TOOLS)
    except ImportError:
        return []


def _init(args):
    global _ARGS
    _ARGS = args
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("DATA_DIR", str(args.data_dir))  # downstream.py finds FSL there


def _rel(text: str) -> str:
    """Strip the local data directory from messages stored in results."""
    return (text or "").replace(str(Path(_ARGS.data_dir).resolve()), "$DATA_DIR").replace(_ARGS.data_dir, "$DATA_DIR")


def _qa(rec: dict, src, out=None, mask=None, brain=None) -> None:
    """Write the case's visual QA sheet; a failure is recorded, never fatal."""
    import qa

    a = _ARGS
    try:
        qa.sheet(RESULTS / "qa" / a.run_name / f"{rec['id']}.jpg", rec, str(src),
                 None if out is None else str(out), None if mask is None else str(mask),
                 None if brain is None else str(brain), label=a.run_name)
        rec["qa_sheet"] = True
    except Exception as e:
        rec["qa_error"] = f"{type(e).__name__}: {e}"[:300]


def run_baseline(a, src, out, mask_p, rep_p, tmp, env) -> dict:
    """Run a baseline defacer and leave the files `radveil auto` would: the
    output, a region mask and a minimal report. A baseline has no region of
    its own, so its region is the set of voxels whose stored value changed;
    integrity (bit-exact outside) is therefore true by definition for it and
    the analysis does not report it."""
    import baselines
    import nibabel as nib

    r = baselines.run(a.config, src, out, tmp, env)
    if r.get("exit") == 0 and out.exists():
        g = baselines.same_grid(src, out)
        if not (g["shape"] and g["affine"]):
            raise ValueError(f"{a.config} changed the image grid: {g}")
        # A tool may rewrite the whole volume: pydeface multiplies by a float
        # mask and saves with the input's integer header (nibabel requantizes
        # every voxel by up to one step); mri_deface writes float32. The region
        # is therefore the voxels that changed by more than requantization
        # (half a stored step, or 1e-3 of the 99th percentile for floats);
        # sub-step changes are counted separately as an integrity finding.
        img, oimg = nib.load(str(src)), nib.load(str(out))
        x = np.squeeze(np.asarray(img.dataobj, np.float64))
        y = np.squeeze(np.asarray(oimg.dataobj, np.float64))
        slope = oimg.dataobj.slope if hasattr(oimg.dataobj, "slope") else 1.0
        step = (slope or 1.0) if np.issubdtype(oimg.get_data_dtype(), np.integer) else 0.0
        tol = max(0.5 * step, 1e-3 * float(np.percentile(np.abs(x), 99)))
        d = np.abs(x - y)
        nib.save(nib.Nifti1Image((d > tol).astype(np.uint8), img.affine), str(mask_p))
        r["requantized_voxels"] = int(np.count_nonzero((d > 0) & (d <= tol)))
        r["region_tolerance"] = round(tol, 6)
        r["grid"] = g
        rep_p.write_text(json.dumps({"tool": a.config, "tool_version": r.get("tool_version")}))
    return r


def _in_fraction(case_id: str, frac: float) -> bool:
    """Deterministic subsample: the same cases for every configuration."""
    import hashlib

    return int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def run_case(row: dict) -> dict:
    import nibabel as nib

    a = _ARGS
    data = Path(a.data_dir)
    src = data / row["image"]
    tmp = Path(tempfile.mkdtemp(prefix="rvpaper_", dir=a.work_dir))
    out, mask_p, rep_p = tmp / "out.nii.gz", tmp / "mask.nii.gz", tmp / "report.json"
    rec = {"id": row["id"], "git": a.git, "attempt": a.attempts.get(row["id"], 0) + 1}
    try:
        env = dict(os.environ, RAYON_NUM_THREADS=str(a.threads))
        if a.config in a.baselines:
            rec.update(run_baseline(a, src, out, mask_p, rep_p, tmp, env))
        else:
            cmd = [a.cli, "auto", str(src), str(out), *CONFIGS[a.config],
                   "--mask-out", str(mask_p), "--report", str(rep_p)]
            if a.brain_check:
                cmd.append("--brain-check")
            rec.update(run_timed(cmd, env=env))
        rec["stderr"] = _rel(rec.get("stderr", ""))
        # exit 3 = --brain-check found modified brain voxels; the output was still written
        if rec["exit"] not in (0, 3) or not out.exists():
            if a.config in a.baselines:
                # a killed container (out of memory) is our error; any other failure is the tool's
                if rec.get("infra_error") or rec["exit"] in (124, 125, 126, 127, 137, -9):
                    rec["status"], rec["reason"] = "error", "container or pipeline failure"
                else:
                    rec["status"] = "refused"
                    rec["reason"] = f"tool failed ({rec['exception']})" if rec.get("exception") else "tool failed"
            else:
                rec["status"], rec["reason"] = classify_exit(rec["exit"], rec["stderr"])
            if a.qa:
                _qa(rec, src)
            return rec
        rec["status"] = "ok"
        rec.pop("stderr", None)
        report = json.loads(rep_p.read_text())
        for k in FACE_GEOMETRY:
            report.pop(k, None)
        for v in report.get("views", []):
            v.pop("landmarks", None)
        rec["report"] = report

        img = nib.load(str(src))
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
        mimg = nib.load(str(mask_p))
        mask = np.squeeze(np.asarray(mimg.dataobj)) > 0
        rec["integrity"] = integrity(src, out, mask, mimg.affine)
        if row.get("brain_mask"):
            bimg = nib.load(str(data / row["brain_mask"]))
            brain = np.squeeze(np.asarray(bimg.dataobj)) > 0
            labels = None
            if row.get("labels"):
                labels = np.squeeze(np.asarray(nib.load(str(data / row["labels"])).dataobj)).astype(np.int32)
            if brain.shape != mask.shape or not np.allclose(bimg.affine, img.affine, atol=1e-3):
                raise ValueError(f"expert brain mask grid differs from the image ({brain.shape} vs {mask.shape})")
            rec["brain"] = brain_metrics(mask, brain, spacing, labels)
        else:
            vox_ml = float(np.prod(spacing)) / 1000.0
            rec["brain"] = {"mask_voxels": int(mask.sum()), "mask_ml": round(mask.sum() * vox_ml, 2)}
        if a.attack:
            import attack

            t0 = time.perf_counter()
            res = attack.probe(str(src), str(out), a.models)
            emb = RESULTS / "embeddings" / a.run_name
            emb.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(emb / f"{row['id']}.npz", **{k: v.astype(np.float16) for k, v in res["emb"].items()})
            rec["attack"] = res["summary"]
            rec["attack_views"] = [v for v in res["views"] if v["any_score"] is not None]
            rec["attack_s"] = round(time.perf_counter() - t0, 2)
        if a.downstream:
            import downstream

            rec["downstream"] = downstream.compare(src, out, tmp)
            if a.seg and _in_fraction(row["id"], a.seg) and "error" not in rec["downstream"]:
                contrast = row.get("contrast") or row.get("modality", "t1")
                rec["segmentation"] = downstream.segment(src, out, tmp, contrast)
        if a.ref_brain and row.get("subject") in a.ref_subjects:
            # independent brain reference where the dataset has no expert mask:
            # the original SynthStrip on the unmodified input (cached)
            import baselines

            ref, rr = baselines.input_brain_mask(src)
            if rr["exit"] == 0:
                b = np.squeeze(np.asarray(nib.load(str(ref)).dataobj)) > 0
                rec["brain_ref"] = brain_metrics(mask, b, spacing)
        if a.shape:
            import shape

            # the output is described at its clean input's surface level and 2x it
            base = float(shape.descriptor(str(src))["threshold"])
            desc = {}
            for m in (1, 2):
                for k_, v in shape.descriptor(str(out), float(m), threshold=base).items():
                    desc[f"out{m}_{k_}"] = v
            d = RESULTS / "shape" / a.run_name
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{row['id']}.npz", **desc)
        if a.qa:
            _qa(rec, src, out, mask_p, data / row["brain_mask"] if row.get("brain_mask") else None)
        if a.keep_outputs:
            keep = Path(a.keep_outputs) / a.run_name
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy(out, keep / f"{row['id']}.nii.gz")
            shutil.copy(mask_p, keep / f"{row['id']}_mask.nii.gz")
    except Exception as e:  # keep the cohort going; the failure is recorded and retried on resume
        rec["status"] = "error"
        rec["error"] = _rel(f"{type(e).__name__}: {e}")[:400]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rec


def main() -> None:
    names = baseline_names()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="manifest name: ixi | synthstrip")
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS) + names)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument("--threads", type=int, default=4, help="RAYON_NUM_THREADS per case")
    ap.add_argument("--attack", action="store_true", help="run the re-identification probe")
    ap.add_argument("--brain-check", action="store_true", help="pass --brain-check (SynthStrip mask)")
    ap.add_argument("--downstream", action="store_true", help="skull-strip input and output (bench/downstream.py)")
    ap.add_argument("--shape", action="store_true", help="save 3D shape descriptors (bench/shape.py)")
    ap.add_argument("--seg", type=float, default=0.0,
                    help="with --downstream: FAST segmentation on this fraction of cases (deterministic)")
    ap.add_argument("--ref-brain", type=int, default=0,
                    help="brain safety vs the original-SynthStrip input mask for N sampled subjects")
    ap.add_argument("--qa", action="store_true", help="write a visual QA sheet per case (results/qa/)")
    ap.add_argument("--filter", default="", help="key=value[,value] manifest filter, e.g. contrast=T1")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke tests)")
    ap.add_argument("--every", type=int, default=1, help="every N-th case (subsampling)")
    ap.add_argument("--subjects", type=int, default=0,
                    help="random sample of N subjects (all their volumes; seed 0; for slow baselines)")
    ap.add_argument("--only-accepted-by", default="",
                    help="only cases this config accepted on the same dataset (paired subsets)")
    ap.add_argument("--keep-outputs", default="", help="copy outputs + masks here (figures)")
    ap.add_argument("--work-dir", default=None, help="scratch directory for outputs")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--verify-data", action="store_true", help="check every image's SHA-256 against the manifest")
    ap.add_argument("--allow-dirty", action="store_true", help="run with uncommitted code changes")
    ap.add_argument("--allow-mixed", action="store_true", help="resume a run made with another revision")
    args = ap.parse_args()
    args.cli = cli_path()
    args.baselines = names
    args.models = str(Path(args.data_dir) / "models")
    env = environment()
    args.git = env["git"]
    if args.git.endswith("-dirty") and not args.allow_dirty:
        sys.exit("working tree has uncommitted changes to code that affects results; commit them "
                 "or pass --allow-dirty (the run is then marked dirty)")
    if args.attack:
        import attack

        missing = attack.missing_models(args.models)
        if missing:
            sys.exit(f"attacker models missing in {args.models}: {missing}; run paper/scripts/fetch_models.sh")

    rows = read_manifest(RESULTS / "manifests" / f"{args.dataset}.csv")
    if args.filter:
        k, v = args.filter.split("=", 1)
        rows = [r for r in rows if r.get(k) in v.split(",")]
    def sample_subjects(n):
        import random

        subjects = sorted({r.get("subject", r["id"]) for r in rows})
        return set(random.Random(0).sample(subjects, min(n, len(subjects))))

    args.ref_subjects = sample_subjects(args.ref_brain) if args.ref_brain else set()
    if args.subjects:
        pick = sample_subjects(args.subjects)  # same seed: the same subjects as --ref-brain
        rows = [r for r in rows if r.get("subject", r["id"]) in pick]
    if args.only_accepted_by:
        acc = {r["id"] for r in read_jsonl(RESULTS / "runs" / f"{args.dataset}__{args.only_accepted_by}.jsonl")
               if r.get("status") == "ok"}
        rows = [r for r in rows if r["id"] in acc]
    rows = rows[:: args.every]
    if args.limit:
        rows = rows[: args.limit]
    data = Path(args.data_dir)
    absent = [r["image"] for r in rows if not (data / r["image"]).exists()]
    if absent:
        sys.exit(f"{len(absent)} manifest images missing under {data} (e.g. {absent[0]}): wrong --data-dir "
                 f"or incomplete download")
    if args.verify_data:
        from common import sha256_file

        def mismatch(r):
            pairs = (("image", "sha256"), ("brain_mask", "brain_mask_sha256"), ("labels", "labels_sha256"))
            return any(r.get(f) and r.get(h) and sha256_file(data / r[f]) != r[h] for f, h in pairs)

        missing = [r["id"] for r in rows if not r.get("sha256")]
        if missing:
            sys.exit(f"{len(missing)} manifest rows have no checksum (e.g. {missing[0]}): regenerate the "
                     f"manifest with `make manifests-update`")
        bad = [r["id"] for r in rows if mismatch(r)]
        if bad:
            sys.exit(f"{len(bad)} images differ from the manifest checksums (e.g. {bad[0]}): not the data "
                     f"the paper used")

    runs = RESULTS / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}__{args.config}{('__' + args.tag) if args.tag else ''}"
    args.run_name = name
    out = runs / f"{name}.jsonl"
    prev = read_jsonl(out, repair=True)
    done = {r["id"] for r in prev if r.get("status") in ("ok", "refused")}
    args.attempts = {r["id"]: r.get("attempt", 1) for r in prev}
    todo = [r for r in rows if r["id"] not in done]

    meta_p = runs / f"{name}.meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"history": []}
    revs = {r.get("git") for r in prev if r.get("git")} | {h["env"]["git"] for h in meta.get("history", [])}
    if prev and revs and args.git not in revs and not args.allow_mixed:
        sys.exit(f"{name} was made with {sorted(revs)}, this code is {args.git}: delete the run to redo it, "
                 f"or pass --allow-mixed")
    session = {
        "flags": CONFIGS.get(args.config, [args.config]) + (["--brain-check"] if args.brain_check else []),
        "attack": args.attack, "downstream": args.downstream, "seg": args.seg, "ref_brain": args.ref_brain,
        "shape": args.shape, "qa": args.qa,
        "jobs": args.jobs, "threads": args.threads, "cases": len(rows), "todo": len(todo),
        "filter": args.filter, "every": args.every, "limit": args.limit, "subjects": args.subjects,
        "only_accepted_by": args.only_accepted_by,
        "argv": [_rel_arg(x, args.data_dir) for x in sys.argv[1:]], "env": env,
    }
    meta.update({"dataset": args.dataset, "config": args.config})
    meta.setdefault("history", []).append(session)
    meta_p.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{name}: {len(rows)} cases, {len(done)} done, {len(todo)} to run "
          f"({args.jobs} jobs x {args.threads} threads)", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(args,)) as ex:
        futs = [ex.submit(run_case, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            append_jsonl(out, f.result())
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el / 60:.1f} min  eta {el / i * (len(todo) - i) / 60:.1f} min",
                      flush=True)
    # stable order for diffs: one record per case, sorted by id
    recs = sorted(read_jsonl(out), key=lambda r: r["id"])
    write_atomic(out, "".join(json.dumps(r) + "\n" for r in recs))
    if any(r.get("status") == "ok" and r.get("peak_rss_mb") is None for r in recs) and a_is_radveil(args):
        print("warning: peak RSS not recorded (install GNU time on Linux)", file=sys.stderr)


def a_is_radveil(args) -> bool:
    return args.config not in args.baselines


def _rel_arg(x: str, data_dir: str) -> str:
    return x.replace(data_dir, "$DATA_DIR")


if __name__ == "__main__":
    main()
