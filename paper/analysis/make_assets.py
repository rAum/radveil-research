#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Turn paper/results/ into everything the manuscript prints.

    python paper/analysis/make_assets.py [--strict]

Reads results/runs/*.jsonl, results/reid_summary.json, results/shape_summary.json,
results/rust_synthstrip.jsonl, results/profile.json, results/qa_review.csv and
results/face_labels.csv (blinded face-presence labels) and
writes
  tex/generated/numbers.tex   \\newcommand macros, one per quoted number
  tex/generated/tab_*.tex     table bodies
  tex/figures/fig_*.pdf       plots (fig_modes.png comes from make_figures.py)
  results/summary.json        the same numbers, machine-readable

Everything generated is deleted first, so a table or figure whose input
disappeared cannot survive from an earlier run. Missing inputs leave their
macros as a bold "??" (listed on stdout); --strict exits non-zero instead.

Conventions. Status per case is ok, refused (the tool declined, with its
reason) or error (a crash or failed metric; never counted as a refusal).
Comparisons between configurations use the paired set: the cases every
compared configuration accepted. Zero counts carry a one-sided 95% upper
bound (Clopper-Pearson). Detection rates carry Wilson 95% intervals.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from common import RESULTS, read_jsonl, read_manifest  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
GEN = PAPER / "tex" / "generated"
FIG = PAPER / "tex" / "figures"
KEEP_FIGURES = {"fig_modes.png"}  # made from the data by make_figures.py, committed

MACROS: dict[str, str] = {}
SUMMARY: dict = {}

RADVEIL = ("adaptive", "adaptive_noverify", "blur", "cut", "adaptive_ears", "adaptive_guard", "adaptive_robust")
BASELINES = ("pydeface", "mri_deface", "quickshear")
CONFIG_LABEL = {
    "adaptive": "RadVeil adaptive",
    "adaptive_noverify": "RadVeil adaptive, no self-check",
    "blur": "RadVeil blur",
    "cut": "RadVeil cut",
    "adaptive_ears": "RadVeil adaptive + ears (cut)",
    "adaptive_guard": "RadVeil adaptive + brain guard",
    "adaptive_robust": "RadVeil adaptive + partial/auto",
    "pydeface": "pydeface",
    "mri_deface": "mri\\_deface",
    "quickshear": "Quickshear",
}
SUBSET_LABEL = {"ixi": "IXI", "fsm": "FSM", "asl": "ASL", "qin": "QIN", "infant": "Infant"}
MODALITY_LABEL = {"t1": "T1w", "t2": "T2w", "pd": "PDw", "mra": "MRA", "dwi": "DWI", "qt1": "qT1",
                  "epi": "PCASL EPI", "flair": "T2-FLAIR"}


# ------------------------------------------------------------------ helpers


def macro(name: str, value, fmt: str = "{}") -> str:
    assert re.fullmatch(r"[A-Za-z]+", name), name
    bad = value is None or (isinstance(value, float) and math.isnan(value))
    s = r"\textbf{??}" if bad else fmt.format(value)
    MACROS[name] = s
    return s


def key(cfg: str) -> str:
    """Macro-name fragment: mri_deface -> MriDeface, controlWhole -> ControlWhole."""
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[^A-Za-z]+", cfg) if p)


def pct(k: int, n: int) -> float | None:
    return None if not n else 100.0 * k / n


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def upper95(k: int, n: int) -> float:
    """One-sided 95% Clopper-Pearson upper bound, in %."""
    from scipy.stats import beta

    return 100.0 if n == 0 else 100.0 * (1.0 if k == n else float(beta.ppf(0.95, k + 1, n - k)))


def fmt_ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f} [{lo:.0f}, {hi:.0f}]" if n else "--"


def fmt_pci(p: float, ci) -> str:
    return f"{100 * p:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def fmt_p(p: float) -> str:
    """A p-value as math-mode LaTeX (the text wraps it in $...$)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return None
    if p >= 0.001:
        return f"{p:.3f}"
    if p <= 0:
        return r"<10^{-300}"
    e = math.floor(math.log10(p))
    m = p / 10**e
    return f"10^{{{e}}}" if abs(m - 1) < 0.05 else rf"{m:.1f}\times10^{{{e}}}"


def run(dataset: str, config: str) -> list[dict]:
    return sorted(read_jsonl(RESULTS / "runs" / f"{dataset}__{config}.jsonl"), key=lambda r: r["id"])


def table(path: str, rows: list[list[str]]) -> None:
    (GEN / path).write_text("\n".join(" & ".join(r) + r" \\" for r in rows) + "\n")


def tex_escape(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def ok(R):
    return [r for r in R if r.get("status") == "ok"]


def status_counts(R) -> Counter:
    return Counter(r.get("status", "?") for r in R)


# ------------------------------------------------------------- SynthStrip


def synthstrip_outcomes(man: dict) -> None:
    R = run("synthstrip", "adaptive")
    if not R:
        return
    groups = defaultdict(list)
    for r in R:
        m = man[r["id"]]
        groups[(m["subset"], m["modality"])].append(r)
    order = ["ixi", "fsm", "asl", "qin", "infant"]
    rows = []
    for k in sorted(groups, key=lambda k: (order.index(k[0]) if k[0] in order else 9, k[1])):
        g = groups[k]
        a = ok(g)
        rounds = Counter(min(r["report"].get("verify_rounds", 0), 3) for r in a)
        rows.append([SUBSET_LABEL.get(k[0], k[0]), MODALITY_LABEL.get(k[1], k[1]), str(len(g)), str(len(a)),
                     str(status_counts(g)["error"]),
                     str(sum(bool(r["report"].get("escalated_to_block")) for r in a)),
                     "/".join(str(rounds.get(i, 0)) for i in range(4)) if a else "--"])
    a = ok(R)
    rounds = Counter(min(r["report"].get("verify_rounds", 0), 3) for r in a)
    rows.append([r"\midrule \textbf{All}", "", str(len(R)), str(len(a)), str(status_counts(R)["error"]),
                 str(sum(bool(r["report"].get("escalated_to_block")) for r in a)),
                 "/".join(str(rounds.get(i, 0)) for i in range(4))])
    table("tab_synthstrip_outcomes.tex", rows)

    sc = status_counts(R)
    macro("ssCases", len(R))
    macro("ssAccepted", sc["ok"])
    macro("ssRefused", sc["refused"])
    macro("ssErrors", sc["error"])
    reasons = Counter(r.get("reason", "?") for r in R if r["status"] == "refused")
    macro("ssRefusedNoFace", reasons.get("no face found", 0))
    macro("ssRefusedRejected", reasons.get("candidates failed 3D consensus", 0))
    macro("ssRefusedFov", reasons.get("face cut by the field of view", 0))
    macro("ssEscalated", sum(bool(r["report"].get("escalated_to_block")) for r in a))
    # verification may end with a face still detected; the output is written anyway
    macro("ssResidualAccepted", sum((r["report"].get("residual_detections") or 0) > 0 for r in a))
    struct = [r for r in R if man[r["id"]]["modality"] in ("t1", "t2", "pd", "flair", "qt1")]
    macro("ssStructCases", len(struct))
    macro("ssStructAccepted", sum(r["status"] == "ok" for r in struct))
    for split in ("test", "valid"):
        g = [r for r in R if man[r["id"]]["split"] == split]
        macro(f"ss{split.title()}Cases", len(g))
        macro(f"ss{split.title()}Accepted", sum(r["status"] == "ok" for r in g))
    integ = [r["integrity"] for r in a]
    macro("ssBitExact", sum(i["bit_exact_outside"] for i in integ))
    macro("ssHeaderKept", sum(i["dtype_preserved"] and i["affine_preserved"] and i["scaling_preserved"] for i in integ))
    macro("ssScaledInputs", sum(i.get("input_has_scaling", False) for i in integ))
    tissue = [i["changed_tissue_ml"] for i in integ if i.get("changed_tissue_ml") is not None]
    if tissue:
        macro("ssTissueMlMedian", float(np.median(tissue)), "{:.0f}")
        macro("ssRegionMlMedian", float(np.median([r["brain"]["mask_ml"] for r in a])), "{:.0f}")
    SUMMARY["synthstrip_outcomes"] = {"cases": len(R), "status": dict(sc), "refusal_reasons": dict(reasons),
                                      "errors": [(r["id"], r.get("error") or r.get("reason")) for r in R
                                                 if r["status"] == "error"]}


def brain_safety(man: dict) -> None:
    """Expert-mask brain safety on the paired set (cases every listed
    configuration accepted) and on each configuration's own accepted set."""
    runs = {c: {r["id"]: r for r in ok(run("synthstrip", c))} for c in RADVEIL + BASELINES}
    runs = {c: v for c, v in runs.items() if v}
    if not runs:
        return
    paired = set.intersection(*(set(v) for c, v in runs.items() if c != "adaptive_robust"))
    macro("bsPairedN", len(paired))
    rows, dists = [], {}
    for cfg, recs in runs.items():
        R = [r for r in recs.values() if r.get("brain", {}).get("min_dist_mm") is not None]
        P = [r for r in R if r["id"] in paired]
        if not R:
            continue
        d = np.array([r["brain"]["min_dist_mm"] for r in P], float)
        touched = [r for r in P if r["brain"]["brain_overlap_voxels"] > 0]
        touched_all = [r for r in R if r["brain"]["brain_overlap_voxels"] > 0]
        lab = [r for r in P if "label_overlap_voxels" in r["brain"]]
        lab_touched = sum(any(v > 0 for g, v in r["brain"]["label_overlap_voxels"].items() if g != "ventricles_csf")
                          for r in lab)
        seg_d = np.array([r["brain"]["seg_min_dist_mm"] for r in lab if r["brain"].get("seg_min_dist_mm") is not None])
        tissue = [r["integrity"].get("changed_tissue_ml") for r in P if r.get("integrity")]
        tissue = [t for t in tissue if t is not None]
        dists[cfg] = d
        rows.append([CONFIG_LABEL[cfg], str(len(P)), str(len(touched)),
                     f"{upper95(len(touched), len(P)):.1f}",
                     f"{max((r['brain']['brain_overlap_ml'] for r in touched), default=0):.2f}",
                     f"{np.median(d):.1f} ({np.percentile(d, 25):.1f}--{np.percentile(d, 75):.1f})" if len(d) else "--",
                     f"{d.min():.1f}" if len(d) else "--",
                     f"{np.median(tissue):.0f}" if tissue else "--",
                     str(len(lab)), str(lab_touched), f"{np.median(seg_d):.1f}" if len(seg_d) else "--"])
        k = key(cfg)
        macro(f"bs{k}N", len(R))
        macro(f"bs{k}Touched", len(touched_all))
        macro(f"bs{k}AllUpper", upper95(len(touched_all), len(R)), "{:.1f}")
        dall = np.array([r["brain"]["min_dist_mm"] for r in R], float)
        macro(f"bs{k}AllMedianDist", float(np.median(dall)), "{:.1f}")
        macro(f"bs{k}AllMinDist", float(dall.min()), "{:.1f}")
        macro(f"bs{k}PairedTouched", len(touched))
        macro(f"bs{k}Upper", upper95(len(touched), len(P)), "{:.1f}")
        if len(d):
            macro(f"bs{k}MedianDist", float(np.median(d)), "{:.1f}")
            macro(f"bs{k}MinDist", float(d.min()), "{:.1f}")
        macro(f"bs{k}LabN", len(lab))
        macro(f"bs{k}LabTouched", lab_touched)
        if len(seg_d):
            macro(f"bs{k}SegMedianDist", float(np.median(seg_d)), "{:.1f}")
        if tissue:
            macro(f"bs{k}TissueMl", float(np.median(tissue)), "{:.0f}")
        if cfg == "adaptive":  # per-subset zero bounds for the default configuration
            per = defaultdict(lambda: [0, 0])
            for r in R:
                s = man[r["id"]]["subset"]
                per[s][0] += r["brain"]["brain_overlap_voxels"] > 0
                per[s][1] += 1
            for s, (t, n) in per.items():
                macro(f"bsAdaptive{s.title()}N", n)
                macro(f"bsAdaptive{s.title()}Upper", upper95(t, n), "{:.0f}")
        SUMMARY.setdefault("brain_safety", {})[cfg] = {
            "n": len(R), "paired_n": len(P), "touched_ids": [r["id"] for r in touched_all]}
    table("tab_brain_safety.tex", rows)
    if dists:
        import matplotlib.pyplot as plt

        # ECDF rather than overlapping histograms: curves further left run
        # closer to the brain, and the 3 mm guard line reads as a fraction.
        # The face region is identical across adaptive/blur/cut, so one curve
        # stands for all three (Okabe-Ito palette, distinct line styles).
        fig, ax = plt.subplots(figsize=(5.4, 2.6))
        for cfg, lab, c, ls in (("adaptive", "RadVeil (face region, any mode)", "#0072B2", "-"),
                                ("adaptive_guard", "RadVeil + brain guard", "#009E73", "-"),
                                ("pydeface", "pydeface", "#E69F00", "--"),
                                ("mri_deface", "mri_deface", "#D55E00", "-."),
                                ("quickshear", "Quickshear", "#CC79A7", ":")):
            if cfg in dists and len(dists[cfg]):
                x = np.sort(dists[cfg])
                y = np.arange(1, len(x) + 1) / len(x)
                ax.step(np.concatenate([[0.0], x]), np.concatenate([[0.0], y]), where="post",
                        color=c, ls=ls, lw=1.6, label=lab)
        ax.axvline(3, color="0.5", ls=":", lw=1)
        ax.text(3.4, 0.04, "brain-guard margin, 3 mm", fontsize=7, color="0.35")
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("closest approach to the expert brain mask (mm)")
        ax.set_ylabel("cumulative fraction of scans")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / "fig_brain_distance.pdf", metadata={"CreationDate": None})
        plt.close(fig)


def fsm_predefaced(man: dict) -> None:
    """Accepted FSM scans whose unmodified input shows no face to YuNet
    (zero detected views in the attack's input pass): the FSM subset was
    partly defaced before release, and accepting such a scan places the
    region over an already-removed face."""
    R = {r["id"]: r for r in ok(run("synthstrip", "adaptive"))
         if man[r["id"]]["subset"] == "fsm" and "attack" in r}
    if not R:
        return
    nf = [r for r in R.values() if r["attack"]["yunet"]["input_detected_views"] == 0]
    macro("fsmAccepted", len(R))
    macro("fsmAcceptedNoFace", len(nf))
    d = [r["brain"]["min_dist_mm"] for r in nf if r.get("brain", {}).get("min_dist_mm") is not None]
    if d:
        macro("fsmNoFaceMinDist", float(min(d)), "{:.1f}")


def ixi_brain_reference() -> None:
    """IXI brain safety against an independent reference (the original
    SynthStrip on the unmodified input) for every tool, on the subject sample
    all tools were run on."""
    rows = []
    for cfg in ("adaptive", "adaptive_noverify", "cut", "adaptive_ears") + BASELINES:
        R = [r for r in ok(run("ixi", cfg)) if r.get("brain_ref", {}).get("min_dist_mm") is not None]
        if not R:
            continue
        t = sum(r["brain_ref"]["brain_overlap_voxels"] > 0 for r in R)
        d = np.array([r["brain_ref"]["min_dist_mm"] for r in R])
        rows.append([CONFIG_LABEL[cfg], str(len(R)), str(t), f"{upper95(t, len(R)):.1f}",
                     f"{np.median(d):.1f}", f"{d.min():.1f}",
                     f"{max((r['brain_ref']['brain_overlap_ml'] for r in R), default=0):.2f}"])
        macro(f"ixiRef{key(cfg)}N", len(R))
        macro(f"ixiRef{key(cfg)}Touched", t)
        macro(f"ixiRef{key(cfg)}Upper", upper95(t, len(R)), "{:.1f}")
        macro(f"ixiRef{key(cfg)}MedianDist", float(np.median(d)), "{:.1f}")
    if rows:
        table("tab_ixi_brain.tex", rows)


def baseline_integrity() -> None:
    for cfg in BASELINES:
        R = ok(run("synthstrip", cfg))
        if R:
            q = [r.get("requantized_voxels", 0) for r in R]
            macro(f"requant{key(cfg)}Scans", sum(x > 0 for x in q))
            macro(f"requant{key(cfg)}N", len(R))
            macro(f"requant{key(cfg)}Median", float(np.median(q)), "{:.0f}")
            macro(f"dtype{key(cfg)}Changed", sum(not r.get("integrity", {}).get("dtype_preserved", True) for r in R))


def attempts() -> None:
    n = sum(r.get("attempt", 1) > 1 for p in (RESULTS / "runs").glob("*.jsonl") if "__smoke" not in p.name
            for r in read_jsonl(p))
    macro("retriedCases", n)


def ears() -> None:
    R = ok(run("synthstrip", "adaptive_ears"))
    if not R:
        return
    det, n = Counter(), 0
    for r in R:
        for roi in r["report"]["deface"].get("rois", []):
            if roi["label"].startswith("ear"):
                n += 1
                det[bool(roi.get("detected"))] += 1
    macro("earScans", len(R))
    macro("earRois", n)
    macro("earDetected", det[True])
    macro("earDetectedPct", pct(det[True], n), "{:.1f}")


def robust_gain(man: dict) -> None:
    a = {r["id"]: r for r in run("synthstrip", "adaptive")}
    b = {r["id"]: r for r in run("synthstrip", "adaptive_robust")}
    if not a or not b:
        return
    gained = [i for i in b if b[i]["status"] == "ok" and a.get(i, {}).get("status") != "ok"]
    macro("robustAccepted", sum(r["status"] == "ok" for r in b.values()))
    macro("robustGained", len(gained))
    macro("robustGainedTouched", sum(b[i].get("brain", {}).get("brain_overlap_voxels", 0) > 0 for i in gained))
    SUMMARY["robust_gained"] = Counter(f"{man[i]['subset']}_{man[i]['modality']}" for i in gained)


def noverify_acceptance() -> None:
    """Verification cannot change acceptance; list any case where the two
    runs disagree (only errors should)."""
    a = {r["id"]: r["status"] for r in run("synthstrip", "adaptive")}
    b = {r["id"]: r["status"] for r in run("synthstrip", "adaptive_noverify")}
    if a and b:
        diff = sorted(i for i in a if i in b and a[i] != b[i])
        SUMMARY["adaptive_vs_noverify_status_differences"] = [(i, a[i], b[i]) for i in diff]
        macro("noverifyStatusDiffs", len(diff))


# --------------------------------------------------------------- attacks


def attacks() -> None:
    runs = {c: {r["id"]: r for r in ok(run("synthstrip", c)) if "attack" in r}
            for c in ("adaptive", "adaptive_noverify", "blur", "cut", "adaptive_ears") + BASELINES}
    runs = {c: v for c, v in runs.items() if v}
    if not runs:
        return
    paired = set.intersection(*(set(v) for v in runs.values()))
    macro("atkPairedN", len(paired))
    rows = []
    for cfg, recs in runs.items():
        cells = [CONFIG_LABEL[cfg]]
        for det in ("mediapipe", "yunet"):
            base = [r for i, r in recs.items() if i in paired and r["attack"][det]["input_detected_views"] > 0]
            any_ = sum(r["attack"][det]["output_residual_views"] > 0 for r in base)
            core = sum(any(v["det"] == det and v["on_face"] and v["out_score"] >= 0.5 and abs(v["yaw"]) <= 20
                           and v["thr_mult"] == 1.0 for v in r.get("attack_views", [])) for r in base)
            cells += [str(len(base)), fmt_ci(any_, len(base)), fmt_ci(core, len(base))]
            k = key(cfg) + det.title()
            macro(f"atk{k}N", len(base))
            macro(f"atk{k}AnyPct", pct(any_, len(base)), "{:.0f}")
            macro(f"atk{k}CorePct", pct(core, len(base)), "{:.0f}")
            SUMMARY.setdefault("attack", {}).setdefault(cfg, {})[det] = {"n": len(base), "any": any_, "core": core}
        rows.append(cells)
    table("tab_attack.tex", rows)


# -------------------------------------------------------------- IXI cohort


def _cluster_ci(values: np.ndarray, subj: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    su = np.unique(subj)
    sums = np.array([values[subj == s].sum() for s in su])
    cnts = np.array([np.sum(subj == s) for s in su])
    pick = rng.integers(0, len(su), (B, len(su)))
    st = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(np.percentile(st, 2.5) * 100), float(np.percentile(st, 97.5) * 100)


def ixi_cohort() -> None:
    R = run("ixi", "adaptive")
    if not R:
        return
    man = {r["id"]: r for r in read_manifest(RESULTS / "manifests" / "ixi.csv")}
    rows = []
    for site in ("Guys", "HH", "IOP"):
        for c in ("T1", "T2", "PD"):
            g = [r for r in R if man[r["id"]]["site"] == site and man[r["id"]]["contrast"] == c]
            if not g:
                continue
            a = ok(g)
            bc = [r["report"]["brain"] for r in a if r["report"].get("brain")]
            touched = sum(b.get("voxels_modified", 0) > 0 for b in bc)
            rows.append([site, c, str(len(g)), fmt_ci(len(a), len(g)), str(status_counts(g)["error"]),
                         str(sum(bool(r["report"].get("escalated_to_block")) for r in a)),
                         f"{touched} ({upper95(touched, len(bc)):.1f})"])
    table("tab_ixi.tex", rows)
    a = ok(R)
    sc = status_counts(R)
    macro("ixiVolumes", len(R))
    macro("ixiSubjects", len({man[r["id"]]["subject"] for r in R}))
    macro("ixiAccepted", sc["ok"])
    macro("ixiAcceptedPct", pct(sc["ok"], len(R)), "{:.1f}")
    macro("ixiRefused", sc["refused"])
    macro("ixiErrors", sc["error"])
    macro("ixiEscalated", sum(bool(r["report"].get("escalated_to_block")) for r in a))
    macro("ixiResidualAccepted", sum((r["report"].get("residual_detections") or 0) > 0 for r in a))
    macro("ixiBitExact", sum(r["integrity"]["bit_exact_outside"] for r in a))
    bc = [r for r in a if r["report"].get("brain")]
    touched = [r for r in bc if r["report"]["brain"].get("voxels_modified", 0) > 0]
    macro("ixiBrainChecked", len(bc))
    macro("ixiBrainTouched", len(touched))
    macro("ixiBrainTouchedUpper", upper95(len(touched), len(bc)), "{:.1f}")
    for c in ("T1", "T2", "PD"):
        g = [r for r in R if man[r["id"]]["contrast"] == c]
        macro(f"ixiAcc{dict(T1='TOne', T2='TTwo', PD='Pd')[c]}Pct", pct(sum(r["status"] == "ok" for r in g), len(g)),
              "{:.1f}")
    # descriptive subgroups with subject-cluster intervals (no inferential reading)
    subj = np.array([man[r["id"]]["subject"] for r in R])
    acc = np.array([r["status"] == "ok" for r in R], float)
    ages = np.array([float(man[r["id"]]["age"]) if man[r["id"]]["age"] else np.nan for r in R])
    groups = {"held-out subjects": np.array([man[r["id"]]["development"] == "0" for r in R]),
              "development subjects": np.array([man[r["id"]]["development"] == "1" for r in R]),
              "age under 40": ages < 40, "age 40 to 59": (ages >= 40) & (ages < 60), "age 60 and over": ages >= 60,
              "age unknown": np.isnan(ages),
              "men": np.array([man[r["id"]]["sex"] == "M" for r in R]),
              "women": np.array([man[r["id"]]["sex"] == "F" for r in R])}
    sub_rows = []
    for name, m in groups.items():
        if m.sum():
            lo, hi = _cluster_ci(acc[m], subj[m])
            sub_rows.append([name, str(int(m.sum())), f"{100 * acc[m].mean():.1f} [{lo:.1f}, {hi:.1f}]"])
            macro(f"ixiAcc{re.sub(r'[^A-Za-z]', '', name.title())}Pct", 100 * acc[m].mean(), "{:.1f}")
    table("tab_ixi_groups.tex", sub_rows)
    SUMMARY["ixi"] = {"volumes": len(R), "status": dict(sc), "brain_touched": [r["id"] for r in touched],
                      "refusal_reasons": Counter(r.get("reason") for r in R if r["status"] == "refused")}
    for cfg in BASELINES + ("cut", "adaptive_ears"):
        B = run("ixi", cfg)
        if B:
            macro(f"ixi{key(cfg)}Accepted", sum(r["status"] == "ok" for r in B))
            macro(f"ixi{key(cfg)}Failed", status_counts(B)["error"] + status_counts(B)["refused"])


# ------------------------------------------------------------ re-identification


def _pci(x: dict, key="rank1", ci="rank1_ci") -> str:
    return fmt_pci(x[key], x[ci])


def reid() -> None:
    p = RESULTS / "reid_summary.json"
    if not p.exists():
        return
    S = json.loads(p.read_text())
    for d, v in S.get("refused_faces", {}).items():
        if v:
            macro(f"refusedFace{d.title()}", v["refused_with_detectable_face"])
            macro(f"refused{d.title()}", v["refused"])
    X = S.get("ixi", {}).get("recognizers", {})
    rows, depth_rows, paired_rows = [], [], []
    for rec in ("arcface", "sface"):
        R = X.get(rec)
        if not R:
            continue
        rl = {"arcface": "ArcFace", "sface": "SFace"}[rec]
        ctl = R["control"]["fused"]
        v3 = ctl.get("verify_at_0.001", {})
        rows.append([rf"\midrule {rl}", "clean (control)", str(ctl["n"]), _pci(ctl), "--", f"{ctl['auc']:.2f}",
                     f"{100 * v3.get('tar', float('nan')):.1f}", "--"])
        macro(f"reid{rl}Gallery", R["gallery"])
        macro(f"reid{rl}GalleryFour", R["gallery_per_depth"]["4.0"])
        macro(f"reid{rl}ProbeScans", R["probe_scans"])
        macro(f"reid{rl}Excluded", R["excluded_no_clean_face"])
        macro(f"reid{rl}ControlRankFused", 100 * ctl["rank1"], "{:.1f}")
        macro(f"reid{rl}ControlRankOne", 100 * R["control"]["1.0"]["rank1"], "{:.1f}")
        macro(f"reid{rl}ControlAuc", ctl["auc"], "{:.2f}")
        macro(f"reid{rl}ControlTar", 100 * v3.get("tar", float("nan")), "{:.1f}")
        macro(f"reid{rl}Chance", 100 * ctl["chance"], "{:.2f}")
        for cfg, C in R["configs"].items():
            o = C.get("oracle")
            if not o:
                continue
            rr = o["release_risk"]["fused"]
            acc = o.get("accepted_only", {}).get("fused")
            va = (acc or {}).get("verify_at_0.001", {})
            van = (acc or {}).get("verify_anon_cal_at_0.001", {})
            rows.append(["", CONFIG_LABEL.get(cfg, cfg), f"{C['acceptance']['accepted']}/{C['acceptance']['probe_scans']}",
                         _pci(rr), _pci(acc) if acc else "--", f"{acc['auc']:.2f}" if acc else "--",
                         f"{100 * va['tar']:.1f}" if va else "--", f"{100 * van['tar']:.1f}" if van else "--"])
            k = rl + key(cfg)
            macro(f"reid{k}Accepted", C["acceptance"]["accepted"])
            macro(f"reid{k}Risk", 100 * rr["rank1"], "{:.1f}")
            macro(f"reid{k}RiskLo", 100 * rr["rank1_ci"][0], "{:.1f}")
            macro(f"reid{k}RiskHi", 100 * rr["rank1_ci"][1], "{:.1f}")
            if acc:
                macro(f"reid{k}AccRank", 100 * acc["rank1"], "{:.1f}")
                macro(f"reid{k}AccRankLo", 100 * acc["rank1_ci"][0], "{:.1f}")
                macro(f"reid{k}AccRankHi", 100 * acc["rank1_ci"][1], "{:.1f}")
                macro(f"reid{k}AccControl", 100 * acc["control_rank1_same_probes"], "{:.1f}")
                macro(f"reid{k}AccAuc", acc["auc"], "{:.2f}")
                macro(f"reid{k}AccP", fmt_p(acc["p_vs_chance"]))
                macro(f"reid{k}AccTenRank", 100 * acc["rank10"], "{:.1f}")
                if va:
                    macro(f"reid{k}AccTar", 100 * va["tar"], "{:.1f}")
                if van:
                    macro(f"reid{k}AccTarAnonCal", 100 * van["tar"], "{:.1f}")
                for depth, nm in (("1.0", "One"), ("2.0", "Two"), ("4.0", "Four")):
                    a = o["accepted_only"].get(depth)
                    if a:
                        macro(f"reid{k}AccDepth{nm}", 100 * a["rank1"], "{:.1f}")
                        macro(f"reid{k}AccDepth{nm}Control", 100 * a["control_rank1_same_probes"], "{:.1f}")
                        macro(f"reid{k}AccDepth{nm}Chance", 100 * a["chance"], "{:.2f}")
                c1, c2 = o.get("gallery_size_curve_fused", {}), o.get("control_gallery_size_curve_fused_same_probes", {})
                if "84" in c1 and "84" in c2:
                    macro(f"reid{k}RankAtEightyFour", 100 * c1["84"], "{:.1f}")
                    macro(f"reid{k}ControlRankAtEightyFour", 100 * c2["84"], "{:.1f}")
            b = C.get("blind", {}).get("accepted_only", {}).get("fused")
            if b:
                macro(f"reid{k}BlindRank", 100 * b["rank1"], "{:.1f}")
            same = C.get("same_scan_bound_fused")
            if same:
                macro(f"reid{k}SameRank", 100 * same["rank1"], "{:.1f}")
            mt = C.get("matched_anonymized_gallery_fused")
            if mt:
                macro(f"reid{k}MatchedRank", 100 * mt["rank1"], "{:.1f}")
            if cfg == "adaptive" and o.get("accepted_only"):
                for depth in ("0.5", "1.0", "2.0", "4.0", "fused"):
                    a = o["accepted_only"].get(depth)
                    if a:
                        dv = a["diff_vs_control"]
                        depth_rows.append([rl, "max over surfaces" if depth == "fused" else f"{depth}$\\times$",
                                           str(a["gallery"]), str(a["n"]),
                                           fmt_pci(a["control_rank1_same_probes"], a["control_rank1_ci_same_probes"]),
                                           _pci(a), f"{a['auc']:.2f}",
                                           f"{100 * dv['diff']:.1f} [{100 * dv['ci'][0]:.1f}, {100 * dv['ci'][1]:.1f}]"])
        P = R.get("paired_sample")
        if P:
            macro(f"reid{rl}PairedProbes", P["probes"])
            macro(f"reid{rl}PairedSubjects", P["subjects"])
            for cfg, v in P["rows"].items():
                d = v.get("diff_vs_adaptive")
                paired_rows.append([rl if cfg == "control" else "", "clean (control)" if cfg == "control"
                                    else CONFIG_LABEL.get(cfg, cfg), str(v.get("accepted", "--")), _pci(v),
                                    f"{v['auc']:.2f}",
                                    f"{100 * d['diff']:+.1f} [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]" if d else "--"])
                macro(f"reid{rl}Paired{key(cfg)}Rank", 100 * v["rank1"], "{:.1f}")
            paired_rows[-len(P["rows"])][0] = rf"\midrule {rl}" if paired_rows[:-len(P["rows"])] else rl
    if rows:
        table("tab_reid.tex", rows)
    if depth_rows:
        table("tab_reid_depth.tex", depth_rows)
    if paired_rows:
        table("tab_reid_paired.tex", paired_rows)


def deconv() -> None:
    p = RESULTS / "deconv_summary.json"
    if not p.exists():
        return
    D = json.loads(p.read_text())
    macro("deconvN", D["n"])
    macro("deconvIters", D["iterations"])
    rows = []
    for rec, rl in (("arcface", "ArcFace"), ("sface", "SFace")):
        cells = [rl]
        for name in ("clean", "plain", "sigma", "sigma_x_adaptive"):
            v = D["rows"].get(f"{rec}_{name}")
            cells.append(_pci(v) if v else "--")
            if v:
                macro(f"deconv{rl}{key(name)}", 100 * v["rank1"], "{:.1f}")
        rows.append(cells)
    table("tab_deconv.tex", rows)


# ------------------------------------------------------------- downstream, shape


def downstream_and_rust() -> None:
    rows = []
    for ds in ("synthstrip", "ixi"):
        for cfg in ("adaptive", "cut") + BASELINES:
            R = [r for r in ok(run(ds, cfg)) if r.get("downstream") and "dice" in r["downstream"]]
            if not R:
                continue
            dice = np.array([r["downstream"]["dice"] for r in R])
            dv = np.array([r["downstream"].get("vol_diff_pct", np.nan) for r in R], float)
            rows.append([{"ixi": "IXI", "synthstrip": "SynthStrip"}[ds], CONFIG_LABEL[cfg], str(len(R)),
                         f"{np.median(dice):.4f}", f"{dice.min():.4f}", f"{np.nanmedian(np.abs(dv)):.2f}",
                         f"{np.nanmax(np.abs(dv)):.2f}"])
            k = {"ixi": "Ixi", "synthstrip": "Ss"}[ds] + key(cfg)
            macro(f"ds{k}N", len(R))
            macro(f"ds{k}DiceMedian", float(np.median(dice)), "{:.4f}")
            macro(f"ds{k}DiceMin", float(dice.min()), "{:.4f}")
            macro(f"ds{k}VolMax", float(np.nanmax(np.abs(dv))), "{:.2f}")
    if rows:
        table("tab_downstream.tex", rows)
    seg_rows = []
    for ds in ("synthstrip",):
        for cfg in ("adaptive", "cut") + BASELINES:
            R = [r for r in ok(run(ds, cfg)) if r.get("segmentation") and "class1_dice" in r["segmentation"]]
            if not R:
                continue
            cells = [CONFIG_LABEL[cfg], str(len(R))]
            for c in (1, 2, 3):
                dc = np.array([r["segmentation"][f"class{c}_dice"] for r in R])
                dv = np.array([abs(r["segmentation"][f"class{c}_vol_diff_pct"]) for r in R])
                cells += [f"{np.median(dc):.3f} ({dc.min():.3f})", f"{np.median(dv):.2f} ({dv.max():.2f})"]
                macro(f"seg{key(cfg)}ClassDice{['', 'One', 'Two', 'Three'][c]}", float(np.median(dc)), "{:.3f}")
            macro(f"seg{key(cfg)}N", len(R))
            seg_rows.append(cells)
    if seg_rows:
        table("tab_segmentation.tex", seg_rows)
    rr = read_jsonl(RESULTS / "rust_synthstrip.jsonl")
    if rr:
        d = np.array([r["dice_vs_reference"] for r in rr if r.get("dice_vs_reference") is not None])
        e = np.array([r["dice_vs_expert_rust"] for r in rr if r.get("dice_vs_expert_rust") is not None])
        e0 = np.array([r["dice_vs_expert_reference"] for r in rr if r.get("dice_vs_expert_reference") is not None])
        if len(d):
            macro("rustN", len(d))
            macro("rustDiceMedian", float(np.median(d)), "{:.4f}")
            macro("rustDiceMin", float(d.min()), "{:.4f}")
        if len(e) and len(e0):
            macro("rustExpertDice", float(np.mean(e)), "{:.3f}")
            macro("refExpertDice", float(np.mean(e0)), "{:.3f}")


def shape_attack() -> None:
    p = RESULTS / "shape_summary.json"
    if not p.exists():
        return
    S = json.loads(p.read_text())

    # row labels use the config tokens; print the paper's names instead
    sub = {k: v.replace("RadVeil ", "") for k, v in CONFIG_LABEL.items()}
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, sub), key=len, reverse=True)) + r")\b")

    def clean(label: str) -> str:
        label = label.replace("clean, same probes as below, moved", "clean, moved (same probes as adaptive)")
        return tex_escape(pat.sub(lambda m: sub[m.group(1)], label))

    def block(label: str) -> str:  # visual grouping of the long table
        for kw in ("paired sample", "moved", "2x surface", "outside the face"):
            if kw in label:
                return kw
        return "whole head"

    rows, prev = [], None
    for name, v in S.get("rows", {}).items():
        lab = v.get("label", name)
        b = block(lab)
        first = clean(lab)
        if prev is not None and b != prev:
            first = r"\addlinespace " + first
        prev = b
        rows.append([first, str(v["n"]), fmt_pci(v["rank1"], v["rank1_ci"]),
                     f"{100 * v['rank10']:.1f}"])
        macro(f"shape{key(name)}Rank", 100 * v["rank1"], "{:.1f}")
    if rows:
        table("tab_shape.tex", rows)
    if "gallery" in S:
        macro("shapeGallery", S["gallery"])
        macro("shapeChance", 100 / S["gallery"], "{:.2f}")


# ------------------------------------------------------------- profiling


def profiling() -> None:
    p = RESULTS / "profile.json"
    if not p.exists():
        return
    P = json.loads(p.read_text())
    env = P["env"]
    macro("hwCpu", tex_escape(env.get("cpu", "unknown CPU")))
    macro("hwCores", env.get("cpu_count"))
    macro("hwMem", env.get("mem_gb"))
    macro("swVersion", tex_escape(env.get("radveil_cli", "")))
    macro("swGit", env.get("git", "")[:10])
    st = P["stages"]
    rows = []
    for i, s in st.items():
        rows.append([tex_escape(i), r"$\times$".join(map(str, s["shape"])),
                     f"{s['load']:.0f}", f"{s['render']:.1f}", f"{s['detect']:.1f}", f"{s['anonymize']:.0f}",
                     f"{s['pipeline_noverify']:.0f}", f"{s['verify_extra']:.0f}", f"{s['pipeline']:.0f}",
                     f"{s['brain_check_extra']:.0f}", f"{s['save']:.0f}"])
    table("tab_stages.tex", rows)
    macro("profStageN", len(st))
    macro("profMedianPipelineMs", float(np.median([s["pipeline"] for s in st.values()])), "{:.0f}")
    macro("profMedianRenderMs", float(np.median([s["render"] for s in st.values()])), "{:.1f}")
    macro("profMedianDetectMs", float(np.median([s["detect"] for s in st.values()])), "{:.1f}")
    macro("profMedianVerifyShare", 100 * float(np.median([s["verify_extra"] / s["pipeline"] for s in st.values()])),
          "{:.0f}")
    macro("profMedianBrainCheckS", float(np.median([s["brain_check_extra"] for s in st.values()])) / 1000, "{:.1f}")
    C = P["cohort_cli"]
    w = np.array([c["wall_s"] for c in C["cases"]])
    rss = np.array([c["peak_rss_mb"] for c in C["cases"]])
    macro("profCohortN", len(w))
    macro("profCohortMedianS", float(np.median(w)), "{:.2f}")
    macro("profCohortPNinetyS", float(np.percentile(w, 90)), "{:.2f}")
    macro("profCohortMaxS", float(w.max()), "{:.2f}")
    macro("profPeakRssMedian", float(np.median(rss)), "{:.0f}")
    macro("profPeakRssMax", float(rss.max()), "{:.0f}")
    T = P["threads"]
    levels = sorted(int(k) for k in T["speedup"])
    macro("profMaxThreads", levels[-1])
    macro("profSpeedupMax", T["speedup"][str(levels[-1])], "{:.1f}")
    macro("profSimdSpeedup", P["simd"]["speedup_median"], "{:.2f}")
    for e in ("fast", "balanced", "thorough"):
        macro(f"profEffort{e.title()}S", P["effort"][e]["median_s"], "{:.2f}")
    tp = P["throughput"]
    wk = sorted(int(k) for k in tp["by_workers"])
    macro("profThroughputMax", max(tp["by_workers"][str(k)]["volumes_per_s"] for k in wk), "{:.1f}")
    macro("profThroughputOne", tp["by_workers"]["1"]["volumes_per_s"], "{:.1f}")
    macro("profThroughputVolumes", tp["volumes"])

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    ax = axs[0]
    meas = [T["speedup"][str(n)] for n in levels]
    ax.plot(levels, meas, "o-", color="#0072B2", label="measured")
    ax.plot(levels, levels, ":", color="0.6", label="linear (ideal)")
    ax.set_ylim(0, max(meas) * 1.3)  # don't let the ideal line squash the data
    ax.set_xticks(levels)
    ax.set_xlabel("threads (one volume, CLI)")
    ax.set_ylabel("speed-up vs 1 thread")
    ax.set_title("(a) thread scaling", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axs[1]
    ax.plot(wk, [tp["by_workers"][str(k)]["volumes_per_s"] for k in wk], "o-", color="#0072B2")
    ax.set_xticks(wk)
    ax.set_xlabel("concurrent volumes (deface_batch)")
    ax.set_ylabel("volumes / s")
    ax.set_title("(b) batch throughput", loc="left", fontsize=8)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_scaling.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def qa_review() -> None:
    """Human review of the QA sheets (analysis/qa_gallery.py -> results/qa_review.csv).
    The number of sheets comes from the run records, which are committed."""
    import csv

    n = sum(bool(r.get("qa_sheet")) for p in (RESULTS / "runs").glob("*.jsonl") if "__smoke" not in p.name
            for r in read_jsonl(p))
    macro("qaSheets", n or None)
    p = RESULTS / "qa_review.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(p.open()))
    macro("qaReviewed", len(rows))
    macro("qaFlagged", sum(r["verdict"] == "bad" for r in rows))
    SUMMARY["qa_flagged"] = [{k: r[k] for k in ("run", "id", "note")} for r in rows if r["verdict"] == "bad"]


def face_labels() -> None:
    """Blinded human labels of face presence in the inputs
    (analysis/face_labels.py -> results/face_labels.csv) crossed with what
    each configuration did. Only a refusal or error on a full face is a false
    refusal; on a partial face (e.g. a slab around the eyes) there may be too
    little face to fit and verify, so refusing is defensible; such a scan still
    shows part of a face and is counted apart (accepted or not), not judged."""
    import csv

    p = RESULTS / "face_labels.csv"
    if not p.exists():
        return
    by: dict[tuple[str, str], dict[str, str]] = {}
    for r in csv.DictReader(p.open()):
        by.setdefault((r["dataset"], r["id"]), {})[r["reviewer"] or "?"] = r["label"]
    reviewers = sorted({w for v in by.values() for w in v})
    macro("flReviewers", len(reviewers))
    if len(reviewers) >= 2:  # Cohen's kappa, first two reviewers, face present vs not
        a, b = reviewers[:2]
        both = [(v[a] in ("face", "partial"), v[b] in ("face", "partial")) for v in by.values()
                if a in v and b in v and "unsure" not in (v[a], v[b])]
        if both:
            po = sum(x == y for x, y in both) / len(both)
            pa, pb = sum(x for x, _ in both) / len(both), sum(y for _, y in both) / len(both)
            pe = pa * pb + (1 - pa) * (1 - pb)
            macro("flKappa", None if pe == 1 else (po - pe) / (1 - pe), "{:.2f}")
            macro("flKappaN", len(both))

    def consensus(v: dict[str, str]) -> str:
        c = Counter(v.values()).most_common()
        return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else "unsure"

    label = {k: consensus(v) for k, v in by.items()}
    rows = []
    for ds in ("synthstrip", "ixi"):
        lab = {i: l for (d, i), l in label.items() if d == ds}
        if not lab:
            continue
        n = Counter(lab.values())
        pre = "fl" + key(ds)
        macro(pre + "Labelled", len(lab))
        for l in ("face", "partial", "none", "unsure"):
            macro(pre + key(l), n[l])
        ent: list[tuple[list[str], list[str]]] = []  # (configs sharing the row, cells)
        for cfg in RADVEIL + BASELINES:
            R = {r["id"]: r.get("status") for r in run(ds, cfg) if r["id"] in lab}
            if not R:
                continue
            face = [i for i in R if lab[i] == "face"]
            part = [i for i in R if lab[i] == "partial"]
            none = [i for i in R if lab[i] == "none"]
            refused = sum(R[i] == "refused" for i in face)
            err = sum(R[i] == "error" for i in face)
            cells = [str(len(face)),
                     str(len(face) - refused - err), str(refused), str(err), fmt_ci(refused + err, len(face)),
                     str(len(part)), str(sum(R[i] == "ok" for i in part)),
                     str(len(none)), str(sum(R[i] != "ok" for i in none)), str(sum(R[i] == "ok" for i in none))]
            # the decision rarely depends on the mode: merge identical RadVeil rows
            if ent and ent[-1][1] == cells and cfg in RADVEIL and ent[-1][0][-1] in RADVEIL:
                ent[-1][0].append(cfg)
            else:
                ent.append(([cfg], cells))
            SUMMARY.setdefault("false_refusals", {})[f"{ds}__{cfg}"] = sorted(
                (i, lab[i], R[i]) for i in face if R[i] != "ok")
            if cfg == "adaptive":
                macro(pre + "FaceN", len(face))
                macro(pre + "FalseRefusals", refused + err)
                macro(pre + "FalseRefusalErrors", err)
                macro(pre + "FalseRefusalPct", pct(refused + err, len(face)), "{:.1f}")
                macro(pre + "FalseRefusalInfant", sum(R[i] != "ok" and i.startswith("infant") for i in face))
                notok = [i for i in R if R[i] != "ok"]  # every refusal and error is labelled, also on IXI
                macro(pre + "NotOkN", len(notok))
                macro(pre + "NotOkPartial", sum(lab[i] == "partial" for i in notok))
                macro(pre + "PartialN", len(part))
                macro(pre + "PartialAccepted", sum(R[i] == "ok" for i in part))
                macro(pre + "NoFaceN", len(none))
                macro(pre + "CorrectRefusals", sum(R[i] != "ok" for i in none))
                macro(pre + "AcceptedNoFace", sum(R[i] == "ok" for i in none))
        present = {c for cfgs, _ in ent for c in cfgs if c in RADVEIL}
        for cfgs, cells in ent:
            if len(cfgs) == 1:
                name = CONFIG_LABEL[cfgs[0]]
            elif set(cfgs) == present:
                name = "RadVeil, every mode"
            elif set(cfgs) == present - {"adaptive_robust"}:
                name = "RadVeil, all modes but partial/auto"
            elif len(cfgs) == 2:
                name = "RadVeil " + " / ".join(CONFIG_LABEL[c].replace("RadVeil ", "") for c in cfgs)
            else:
                name = f"RadVeil, {len(cfgs)} modes alike"
            rows.append(["IXI" if ds == "ixi" else "SynthStrip", name] + cells)
    if rows:
        table("tab_face_labels.tex", rows)


def main() -> None:
    strict = "--strict" in sys.argv
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for f in list(GEN.glob("*.tex")) + [f for f in FIG.glob("fig_*") if f.name not in KEEP_FIGURES]:
        f.unlink()
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "font.family": "serif"})
    man = {r["id"]: r for r in read_manifest(RESULTS / "manifests" / "synthstrip.csv")}
    ds = json.loads((RESULTS / "manifests" / "datasets.json").read_text())
    macro("ssWithImage", ds["synthstrip"]["cases_with_image"])
    macro("ssWithLabels", ds["synthstrip"]["with_labels"])
    macro("ixiDevSubjects", ds["ixi"].get("development_subjects"))

    synthstrip_outcomes(man)
    brain_safety(man)
    ears()
    fsm_predefaced(man)
    ixi_brain_reference()
    baseline_integrity()
    attempts()
    robust_gain(man)
    noverify_acceptance()
    attacks()
    ixi_cohort()
    reid()
    deconv()
    downstream_and_rust()
    shape_attack()
    profiling()
    qa_review()
    face_labels()

    tex = (PAPER / "tex" / "radveil.tex").read_text()
    used = set(re.findall(r"\\([a-z]+[A-Z][A-Za-z]*)", tex))
    for k in sorted(used - set(MACROS)):
        macro(k, None)
    pending = sorted(k for k, v in MACROS.items() if "??" in v and k in used)
    lines = ["% generated by paper/analysis/make_assets.py -- do not edit"]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    for t in sorted(set(re.findall(r"\\tablebody\{(\w+)\}", tex))):
        if not (GEN / f"{t}.tex").exists():
            (GEN / f"{t}.tex").write_text(r"\multicolumn{3}{l}{\textbf{?? (run pending)}} \\" + "\n")
            pending.append(f"table:{t}")
    (RESULTS / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str) + "\n")
    print(f"{len(MACROS)} macros -> {GEN / 'numbers.tex'}")
    if pending:
        print(f"{len(pending)} pending: {' '.join(pending)}")
        if strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
