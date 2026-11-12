#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Latency, scaling and memory profile of RadVeil.

    python paper/bench/profile.py --data-dir $DATA_DIR [--repeats 5] [--quick]

Run it on an otherwise idle machine: every number is a wall-clock time.

Sampling. Volumes are drawn at random (fixed seed) from the scans the
SynthStrip `adaptive` run accepted, two per subset/contrast, so grid sizes
from 2D clinical slabs to 0.5 mm MRA are covered; the cohort latency
distribution uses 40 random accepted scans. Refused scans are not profiled
(they exit after detection, much faster).

Measurements (times in ms unless noted):

  stages      warm in-process Python API, all cores: load, one render, one
              detection, the anonymization alone, the full pipeline without
              and with verification and with brain_check, save. The extra
              cost of verification and brain_check is the median of
              per-repeat paired differences, never a difference of medians.
  cohort_cli  the CLI end to end (process start, load, save .nii.gz), all
              cores, one warm-up run per volume discarded: the per-volume
              latency distribution the paper quotes, with peak RSS.
  threads     CLI at RAYON_NUM_THREADS = 1, 2, 4, 8, all cores, levels in a
              random order per repeat; speed-up = median over volumes of
              t(1 thread) / t(n threads).
  simd        CLI with SIMD kernels vs RADVEIL_SIMD=off, interleaved.
  effort      CLI at --verify-effort fast / balanced / thorough, interleaved.
  throughput  radveil.deface_batch over distinct volumes at 1, 2, 4, 8, all
              workers (Rayon threads divided automatically), 2 repeats.

Writes paper/results/profile.json (with the environment block).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, cli_path, environment, read_jsonl, read_manifest, run_timed  # noqa: E402

YAWS = (-20.0, 0.0, 20.0)
MODE = "adaptive"
SEED = 0


def accepted(quick: bool) -> list[dict]:
    rows = {r["id"]: r for r in read_manifest(RESULTS / "manifests" / "synthstrip.csv")}
    runs = RESULTS / "runs"
    src = runs / ("synthstrip__adaptive__smoke.jsonl" if quick else "synthstrip__adaptive.jsonl")
    done = [rows[r["id"]] for r in read_jsonl(src) if r.get("status") == "ok"]
    if not done:
        sys.exit(f"run the synthstrip/adaptive cohort first ({src.name})")
    return sorted(done, key=lambda r: r["id"])


def stratified(rows: list[dict], per: int, rng) -> list[dict]:
    by = {}
    for r in rows:
        by.setdefault((r["subset"], r["modality"]), []).append(r)
    out = []
    for k in sorted(by):
        out += rng.sample(by[k], min(per, len(by[k])))
    return out


def med(xs):
    return statistics.median(xs)


def stages(path: str, repeats: int) -> dict:
    import radveil

    t = {k: [] for k in ("load", "render", "detect", "anonymize", "save", "pipeline_noverify", "pipeline",
                         "verify_extra", "brain_check_extra")}
    info = {}
    with tempfile.TemporaryDirectory() as tmp:
        for rep in range(repeats + 1):  # first iteration warms caches, dropped
            t0 = time.perf_counter(); vol = radveil.Volume(path); t1 = time.perf_counter()
            views = [vol.render(yaw_deg=y) for y in YAWS]; t2 = time.perf_counter()
            lms = [radveil.detect_landmarks(v["rgb"]) for v in views]; t3 = time.perf_counter()
            pts = radveil.lift_landmarks(views, lms)
            t4 = time.perf_counter(); vol.deface(pts, mode=MODE); t5 = time.perf_counter()
            vol.save(os.path.join(tmp, "o.nii.gz")); t6 = time.perf_counter()
            runs = {}
            for k, kw in (("pipeline_noverify", dict(verify=False)), ("pipeline", {}),
                          ("pipeline_brain_check", dict(brain_check=True))):
                v = radveil.Volume(path)
                a = time.perf_counter(); rep_ = radveil.deface_volume(v, mode=MODE, **kw)
                runs[k] = time.perf_counter() - a
            if rep == 0:
                info = {"shape": list(vol.shape), "spacing": [round(s, 3) for s in vol.spacing],
                        "voxels": int(vol.shape[0] * vol.shape[1] * vol.shape[2]),
                        "verify_rounds": rep_.get("verify_rounds"),
                        "escalated_to_block": rep_.get("escalated_to_block")}
                continue
            for k, dt in (("load", t1 - t0), ("render", (t2 - t1) / 3), ("detect", (t3 - t2) / 3),
                          ("anonymize", t5 - t4), ("save", t6 - t5),
                          ("pipeline_noverify", runs["pipeline_noverify"]), ("pipeline", runs["pipeline"]),
                          ("verify_extra", runs["pipeline"] - runs["pipeline_noverify"]),
                          ("brain_check_extra", runs["pipeline_brain_check"] - runs["pipeline"])):
                t[k].append(1e3 * dt)
    return {**info, **{k: round(med(v), 2) for k, v in t.items()}}


def cli(path: str, extra: list[str], env_extra: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        r = run_timed([cli_path(), "auto", path, os.path.join(tmp, "o.nii.gz"), "--mode", MODE, *extra],
                      env=dict(os.environ, **env_extra))
    if r["exit"] != 0:
        raise RuntimeError(f"{path}: {r['stderr']}")
    if r["peak_rss_mb"] is None:
        raise RuntimeError("peak RSS unavailable: install GNU time (Linux) or use macOS /usr/bin/time")
    return r


def interleaved(path: str, variants: dict[str, tuple[list, dict]], repeats: int, rng) -> dict:
    """Median wall time per variant, variants in a random order each repeat,
    after one warm-up run of each."""
    for extra, env in variants.values():
        cli(path, extra, env)
    walls = {k: [] for k in variants}
    for _ in range(repeats):
        order = list(variants)
        rng.shuffle(order)
        for k in order:
            walls[k].append(cli(path, *variants[k])["wall_s"])
    return {k: med(v) for k, v in walls.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="tiny sample, 1 repeat (smoke test)")
    a = ap.parse_args()
    rng = random.Random(SEED)
    if a.quick:
        a.repeats = 1
    data = Path(a.data_dir)
    pool = accepted(a.quick)
    sample = stratified(pool, 1 if a.quick else 2, rng)
    cohort = rng.sample(pool, min(4 if a.quick else 40, len(pool)))
    ncpu = os.cpu_count() or 1
    levels = sorted({1, 2, 4, 8, ncpu} & set(range(1, ncpu + 1)))
    P: dict = {"env": environment(), "mode": MODE, "repeats": a.repeats, "seed": SEED,
               "sample": [r["id"] for r in sample]}
    print("sample:", P["sample"], flush=True)

    P["stages"] = {}
    for r in sample:
        P["stages"][r["id"]] = stages(str(data / r["image"]), a.repeats)
        print("stages", r["id"], P["stages"][r["id"]], flush=True)

    cases = []
    for r in cohort:
        p = str(data / r["image"])
        cli(p, [], {})  # warm-up
        x = cli(p, [], {})
        cases.append({"id": r["id"], "wall_s": x["wall_s"], "peak_rss_mb": x["peak_rss_mb"]})
    P["cohort_cli"] = {"cases": cases}
    print("cohort", med([c["wall_s"] for c in cases]), flush=True)

    per = {}
    for r in sample:
        per[r["id"]] = interleaved(str(data / r["image"]),
                                   {str(n): ([], {"RAYON_NUM_THREADS": str(n)}) for n in levels}, a.repeats, rng)
    P["threads"] = {"per_volume": per,
                    "speedup": {str(n): round(med([v["1"] / v[str(n)] for v in per.values()]), 3) for n in levels}}
    print("threads", P["threads"]["speedup"], flush=True)

    simd = {r["id"]: interleaved(str(data / r["image"]), {"simd": ([], {}), "portable": ([], {"RADVEIL_SIMD": "off"})},
                                 a.repeats, rng) for r in sample}
    P["simd"] = {"per_volume": simd, "speedup_median": round(med([v["portable"] / v["simd"] for v in simd.values()]), 3)}
    eff = {r["id"]: interleaved(str(data / r["image"]),
                                {e: (["--verify-effort", e], {}) for e in ("fast", "balanced", "thorough")},
                                a.repeats, rng) for r in sample}
    P["effort"] = {e: {"median_s": round(med([v[e] for v in eff.values()]), 3)} for e in ("fast", "balanced", "thorough")}
    P["effort"]["per_volume"] = eff
    print("simd/effort", P["simd"]["speedup_median"], {e: P["effort"][e] for e in ("fast", "balanced", "thorough")},
          flush=True)

    import radveil

    distinct = [str(data / r["image"]) for r in rng.sample(pool, min(len(pool), 2 if a.quick else 4 * ncpu))]
    P["throughput"] = {"volumes": len(distinct), "by_workers": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for w in levels:
            rates = []
            for _ in range(1 if a.quick else 2):
                outs = [os.path.join(tmp, f"{i}.nii.gz") for i in range(len(distinct))]
                t0 = time.perf_counter()
                reps = radveil.deface_batch(distinct, outputs=outs, mode=MODE, workers=w)
                rates.append(len(distinct) / (time.perf_counter() - t0))
                if any("error" in x for x in reps):
                    raise RuntimeError(f"deface_batch errors at {w} workers")
            P["throughput"]["by_workers"][str(w)] = {"volumes_per_s": round(med(rates), 2),
                                                     "rayon_threads_per_worker": max(1, ncpu // w)}
            print("throughput", w, P["throughput"]["by_workers"][str(w)], flush=True)

    out = RESULTS / ("profile_quick.json" if a.quick else "profile.json")
    out.write_text(json.dumps(P, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
