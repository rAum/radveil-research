#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Face re-identification analysis on IXI.

    python paper/analysis/reid.py      # -> paper/results/reid_summary.json

Needs the (git-ignored) embeddings in results/embeddings/; writes the
committed summary that analysis/make_assets.py turns into the paper's
tables, so the PDF rebuilds without embeddings or data.

Design (see the manuscript, Section 5):

* Gallery: one template per subject from the unmodified T1 (mean of the
  normalised embeddings of the -40..+40 deg renders). At each iso-surface
  depth the gallery holds only subjects with a template at that depth, and
  chance is 1/(that gallery size). Probes: the subjects' T2 and PD.
* Fused attacker: maximum similarity over the four depths, for genuine and
  impostor pairs alike; a gallery subject without a template at some depth
  simply contributes fewer depths.
* Alignment: "oracle" = the better of aligning the output on the input's
  face location and on the output's own detection there (per comparison);
  "blind" = the output's own best detection anywhere.
* Clustering: T2 and PD of one subject are one acquisition. Intervals are
  subject-level cluster-bootstrap intervals; with five or fewer hits the
  interval is Clopper-Pearson over subjects instead (a subject counts as
  hit if any of its probes is). Clean vs anonymized differences are paired
  cluster-bootstrap differences with a two-sided bootstrap p-value. The test
  against chance counts subjects, not probes.
* Verification: thresholds at FAR 1% and 0.1% fitted on the impostor scores
  of half the subjects (excluding comparisons with test-half identities) and
  applied to the other half, 20 random halvings; mean and 95% range. Two
  attackers: calibrated on clean impostors, and calibrated on anonymized
  impostors (an attacker who anonymizes their own data with the same tool).
* Release risk (intention to treat): every probe a configuration was run on;
  a refused scan is released unmodified and contributes its clean outcome.
  Accepted-only figures are labelled as such and paired with the same
  probes' clean outcome.
* Paired comparison: all configurations on the probes every configuration
  was run on (the baselines' subject sample), so tools are compared on the
  same scans.
* Matched condition: anonymized probes against an anonymized T1 gallery.
* Same scan: anonymized T1 vs its own original, a sanity bound only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from common import RESULTS, read_jsonl, read_manifest  # noqa: E402

MULTS = (0.5, 1.0, 2.0, 4.0)  # attack.RECOG_MULT
RECOGNIZERS = ("arcface", "sface")
FARS = (0.01, 0.001)
B = 2000
SPLITS = 20
SEED = 0
CONFIGS = ["adaptive", "adaptive_noverify", "cut", "adaptive_ears", "pydeface", "mri_deface", "quickshear"]


# ------------------------------------------------------------- similarity


def _templates(arr: np.ndarray) -> np.ndarray:
    """(n, mult, yaw, dim) -> (n, mult, dim) normalised mean over yaws; NaN rows
    where no view had a face."""
    a = arr.astype(np.float32)
    ok = np.isfinite(a[..., 0])
    s = np.where(ok[..., None], a, 0).sum(2)
    n = ok.sum(2)[..., None]
    t = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return t / np.linalg.norm(t, axis=-1, keepdims=True)


def _sim(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """(np, mult, d) x (ng, mult, d) -> (np, ng, mult) cosine, NaN where
    either side is missing."""
    with np.errstate(all="ignore"):
        S = np.einsum("pmd,gmd->pgm", np.nan_to_num(P), np.nan_to_num(G))
    miss = ~np.isfinite(P[..., 0])[:, None, :] | ~np.isfinite(G[..., 0])[None, :, :]
    S[miss] = np.nan
    return S


def _fuse(S: np.ndarray) -> np.ndarray:
    """Max over the last axis, NaN only where every entry is missing."""
    with np.errstate(all="ignore"):
        F = np.nanmax(np.where(np.isfinite(S), S, -np.inf), -1)
    return np.where(np.isfinite(F), F, np.nan)


def _metrics(S: np.ndarray, truth: np.ndarray) -> dict:
    """Per-probe outcomes from a (np, ng) similarity matrix; truth[i] is the
    gallery index of probe i. Ties go to the genuine candidate; a probe with
    no genuine score (no face could be aligned) gets the worst rank."""
    n, g = S.shape
    gen = S[np.arange(n), truth]
    Sm = np.where(np.isfinite(S), S, -np.inf)
    rank = (Sm > np.where(np.isfinite(gen), gen, np.inf)[:, None]).sum(1) + 1
    rank = np.where(np.isfinite(gen), rank, g)
    imp = S.copy()
    imp[np.arange(n), truth] = np.nan
    return {"rank": rank, "gen": gen, "imp": imp, "truth": truth}


def _sub(m: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in m.items()}


# ------------------------------------------------------------- statistics


def _auc(gen: np.ndarray, imp: np.ndarray) -> float:
    gen, imp = gen[np.isfinite(gen)], imp[np.isfinite(imp)]
    if not len(gen) or not len(imp):
        return float("nan")
    from scipy.stats import mannwhitneyu

    return float(mannwhitneyu(gen, imp).statistic / (len(gen) * len(imp)))


def _clusters(subj: np.ndarray):
    su, inv = np.unique(subj, return_inverse=True)
    return su, inv


def _ci(hit: np.ndarray, subj: np.ndarray, rng) -> list[float]:
    """95% interval of a clustered proportion: cluster bootstrap, or with few
    hits Clopper-Pearson over subjects (conservative)."""
    from scipy.stats import beta

    su, inv = _clusters(subj)
    if hit.sum() <= 5:
        k = int(np.bincount(inv, weights=hit, minlength=len(su)).astype(bool).sum())
        n = len(su)
        lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
        return [lo, hi]
    sums = np.bincount(inv, weights=hit, minlength=len(su))
    cnts = np.bincount(inv, minlength=len(su)).astype(float)
    pick = rng.integers(0, len(su), (B, len(su)))
    st = sums[pick].sum(1) / cnts[pick].sum(1)
    return [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def _paired_diff(a: np.ndarray, b: np.ndarray, subj: np.ndarray, rng) -> dict:
    """mean(a) - mean(b) for paired binary outcomes, cluster-bootstrap 95%
    interval and two-sided bootstrap p-value."""
    su, inv = _clusters(subj)
    da = np.bincount(inv, weights=a.astype(float), minlength=len(su))
    db = np.bincount(inv, weights=b.astype(float), minlength=len(su))
    cnt = np.bincount(inv, minlength=len(su)).astype(float)
    pick = rng.integers(0, len(su), (B, len(su)))
    d = (da[pick].sum(1) - db[pick].sum(1)) / cnt[pick].sum(1)
    est = float((a.astype(float) - b.astype(float)).mean())
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"diff": est, "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p": p}


def _p_chance(hit: np.ndarray, subj: np.ndarray, g: int) -> float:
    """One-sided test against chance with subjects as units: a subject is hit
    if any of its probes is; under chance each probe hits with p = 1/g."""
    from scipy.stats import binomtest

    su, inv = _clusters(subj)
    k = int(np.bincount(inv, weights=hit, minlength=len(su)).astype(bool).sum())
    m = np.bincount(inv, minlength=len(su))
    p0 = float(np.mean(1 - (1 - 1.0 / g) ** m))
    return float(binomtest(k, len(su), p0, alternative="greater").pvalue)


def _splits(subj: np.ndarray, seed: int) -> list[np.ndarray]:
    """SPLITS random halvings of the subjects (boolean: probe in half A)."""
    rng = np.random.default_rng(seed)
    su = np.unique(subj)
    out = []
    for _ in range(SPLITS):
        half = set(rng.permutation(su)[: len(su) // 2])
        out.append(np.array([s in half for s in subj]))
    return out


def _tar(m_cal: dict, m: dict, splits, gallery_subj: np.ndarray, subj: np.ndarray, far: float) -> dict:
    """Cross-validated TAR and realised FAR at a threshold fitted on m_cal's
    impostors of one half (comparisons with the other half's identities
    excluded) and applied to m on the other half."""
    tars, fars = [], []
    for in_a in splits:
        half_a = set(subj[in_a])
        col_a = np.array([s in half_a for s in gallery_subj])
        for fit, test, cols in ((in_a, ~in_a, col_a), (~in_a, in_a, ~col_a)):
            imp = m_cal["imp"][fit][:, cols]
            imp = imp[np.isfinite(imp)]
            if not len(imp) or not test.any():
                continue
            thr = np.quantile(imp, 1 - far)
            tars.append(float(np.mean(np.nan_to_num(m["gen"][test], nan=-np.inf) >= thr)))
            ti = m["imp"][test][:, ~cols]
            ti = ti[np.isfinite(ti)]
            fars.append(float(np.mean(ti >= thr)) if len(ti) else float("nan"))
    if not tars:
        return {}
    return {"tar": float(np.mean(tars)),
            "tar_range": [float(np.percentile(tars, 2.5)), float(np.percentile(tars, 97.5))],
            "far_realised": float(np.nanmean(fars))}


def _summary(m: dict, subj: np.ndarray, rng, g: int, gallery_subj=None, cal: dict | None = None,
             cal_anon: dict | None = None, splits=None) -> dict:
    hit = (m["rank"] == 1).astype(float)
    out = {"n": int(len(hit)), "subjects": int(len(np.unique(subj))), "gallery": int(g),
           "rank1": float(hit.mean()), "rank1_ci": _ci(hit, subj, rng),
           "rank5": float((m["rank"] <= 5).mean()), "rank10": float((m["rank"] <= 10).mean()),
           "mean_rank": float(m["rank"].mean()), "auc": _auc(m["gen"], m["imp"]),
           "chance": 1.0 / g, "p_vs_chance": _p_chance(hit, subj, g)}
    if cal is not None and splits is not None:
        for far in FARS:
            out[f"verify_at_{far:g}"] = _tar(cal, m, splits, gallery_subj, subj, far)
            if cal_anon is not None:
                out[f"verify_anon_cal_at_{far:g}"] = _tar(cal_anon, m, splits, gallery_subj, subj, far)
    return out


def _gallery_size_curve(S: np.ndarray, truth: np.ndarray, rng, sizes=(10, 50, 84, 200), draws=200) -> dict:
    """Rank-1 (ties to the genuine) when the candidate list has N subjects,
    the true one included: mean over `draws` random lists per probe."""
    n, g = S.shape
    Sm = np.where(np.isfinite(S), S, -np.inf)
    gen = Sm[np.arange(n), truth]
    out = {}
    for N in sizes:
        if N >= g:
            continue
        hits = 0
        for i in range(n):
            others = np.delete(Sm[i], truth[i])
            draw = np.array([rng.choice(others, N - 1, replace=False).max() for _ in range(draws)])
            hits += np.sum(np.isfinite(gen[i]) & (gen[i] >= draw))
        out[str(N)] = float(hits / (n * draws))
    return out


# ------------------------------------------------------------- analysis


def _load(edir: Path, i: str, key: str):
    z = np.load(edir / f"{i}.npz")
    return _templates(z[key][None])[0] if key in z.files else None


def _depth_views(S: np.ndarray, truth: np.ndarray, gvalid: np.ndarray) -> dict:
    """Per-depth metrics on the gallery subjects valid at that depth (probes
    of other subjects dropped), and fused metrics on the full gallery.
    Values: (metrics, probe mask, gallery size)."""
    res = {}
    for j, k in enumerate(MULTS):
        cols = gvalid[:, j]
        keep = cols[truth]
        idx = np.cumsum(cols) - 1
        res[str(k)] = (_metrics(S[keep][:, cols, j], idx[truth[keep]]), keep, int(cols.sum()))
    res["fused"] = (_metrics(_fuse(S), truth), np.ones(len(truth), bool), S.shape[1])
    return res


def analyse_ixi() -> dict:
    man = {r["id"]: r for r in read_manifest(RESULTS / "manifests" / "ixi.csv")}
    gz = RESULTS / "embeddings" / "ixi__gallery.npz"
    if not gz.exists():
        return {}
    g = np.load(gz)
    ids = [i for i in g["ids"].tolist() if i in man]
    one = MULTS.index(1.0)
    runs = {c: {r["id"]: r for r in read_jsonl(RESULTS / "runs" / f"ixi__{c}.jsonl")} for c in CONFIGS}
    runs = {c: v for c, v in runs.items() if v and (RESULTS / "embeddings" / f"ixi__{c}").exists()}
    res = {"design": {"mults": MULTS, "bootstrap": B, "splits": SPLITS, "fars": FARS}, "recognizers": {}}
    for rec in RECOGNIZERS:
        rng = np.random.default_rng(SEED)
        clean = dict(zip(g["ids"].tolist(), _templates(g[rec])))
        t1 = {man[i]["subject"]: i for i in ids if man[i]["contrast"] == "T1"}
        gsub = sorted(s for s, i in t1.items() if np.isfinite(clean[i][one, 0]))
        gi = {s: k for k, s in enumerate(gsub)}
        G = np.stack([clean[t1[s]] for s in gsub])
        gvalid = np.isfinite(G[..., 0])
        gsub_arr = np.array(gsub)
        probe_ids = sorted(i for i in ids if man[i]["contrast"] in ("T2", "PD") and man[i]["subject"] in gi
                           and np.isfinite(clean[i][one, 0]))
        subj = np.array([man[i]["subject"] for i in probe_ids])
        truth = np.array([gi[s] for s in subj])
        splits = _splits(subj, SEED)
        Sc = _sim(np.stack([clean[i] for i in probe_ids]), G)
        ctrl = _depth_views(Sc, truth, gvalid)
        R = {"gallery": len(gsub),
             "gallery_per_depth": {str(k): int(gvalid[:, j].sum()) for j, k in enumerate(MULTS)},
             "probe_scans": len(probe_ids),
             "excluded_no_clean_face": sum(1 for i in ids if man[i]["contrast"] in ("T2", "PD")
                                           and man[i]["subject"] in gi and not np.isfinite(clean[i][one, 0])),
             "control": {}, "configs": {}}
        for k, (m, keep, gk) in ctrl.items():
            if k == "fused":
                R["control"][k] = _summary(m, subj, rng, gk, gsub_arr, m, None, splits)
                R["control"][k]["gallery_size_curve"] = _gallery_size_curve(_fuse(Sc), truth, rng)
            else:
                R["control"][k] = _summary(m, subj[keep], rng, gk)

        nan_t = np.full_like(G[0], np.nan)

        def anonymized(cfg, variant):
            """Similarities of every probe: anonymized where accepted, else clean."""
            edir = RESULTS / "embeddings" / f"ixi__{cfg}"
            r = runs[cfg]
            acc = np.array([r.get(i, {}).get("status") == "ok" and (edir / f"{i}.npz").exists() for i in probe_ids])
            P = [(_load(edir, i, f"{rec}_out_{variant}") if a else clean[i]) for i, a in zip(probe_ids, acc)]
            S = _sim(np.stack([p if p is not None else nan_t for p in P]), G)
            if variant == "oracle":  # the better of the two oracle alignments, per comparison
                Pn = [(_load(edir, i, f"{rec}_out_near") if a else clean[i]) for i, a in zip(probe_ids, acc)]
                Sn = _sim(np.stack([p if p is not None else nan_t for p in Pn]), G)
                S = np.where(np.isfinite(Sn) & (~np.isfinite(S) | (Sn > S)), Sn, S)
            seen = np.array([i in r for i in probe_ids])
            return S, acc, seen

        cache = {}
        for cfg in runs:
            C = {}
            for variant in ("oracle", "blind"):
                S, acc, seen = anonymized(cfg, variant)
                cache[(cfg, variant)] = (S, acc, seen)
                an = _depth_views(S, truth, gvalid)
                C["acceptance"] = {"probe_scans": int(seen.sum()), "accepted": int(acc.sum()),
                                   "refused": int(sum(runs[cfg].get(i, {}).get("status") == "refused"
                                                      for i in probe_ids)),
                                   "error": int(sum(runs[cfg].get(i, {}).get("status") == "error"
                                                    for i in probe_ids))}
                m_f = an["fused"][0]
                V = {"release_risk": {"fused": _summary(_sub(m_f, seen), subj[seen], rng, len(gsub), gsub_arr,
                                                        _sub(ctrl["fused"][0], seen), _sub(m_f, seen),
                                                        [s[seen] for s in splits])},
                     "accepted_only": {}}
                for k, (m, keep, gk) in an.items():
                    if k == "fused":
                        sel = acc
                        mk, ck = _sub(m, sel), _sub(ctrl["fused"][0], sel)
                    else:
                        sel = keep & acc
                        mk, ck = _sub(m, acc[keep]), _sub(ctrl[k][0], acc[keep])
                    if not sel.any():
                        continue
                    sj = subj[sel]
                    if k == "fused":
                        s = _summary(mk, sj, rng, gk, gsub_arr, ck, mk, [x[sel] for x in splits])
                    else:
                        s = _summary(mk, sj, rng, gk)
                    s["control_rank1_same_probes"] = float((ck["rank"] == 1).mean())
                    s["control_rank1_ci_same_probes"] = _ci((ck["rank"] == 1).astype(float), sj, rng)
                    s["diff_vs_control"] = _paired_diff(mk["rank"] == 1, ck["rank"] == 1, sj, rng)
                    V["accepted_only"][k] = s
                if variant == "oracle" and acc.any():
                    V["gallery_size_curve_fused"] = _gallery_size_curve(_fuse(S)[acc], truth[acc], rng)
                    V["control_gallery_size_curve_fused_same_probes"] = _gallery_size_curve(
                        _fuse(Sc)[acc], truth[acc], rng)
                C[variant] = V
            edir = RESULTS / "embeddings" / f"ixi__{cfg}"

            def ok(i):
                return runs[cfg].get(i, {}).get("status") == "ok" and (edir / f"{i}.npz").exists()

            # matched condition: anonymized probes vs an anonymized T1 gallery
            t1_ok = sorted(s for s in gsub if ok(t1[s]))
            acc_p = [i for i in probe_ids if ok(i) and man[i]["subject"] in set(t1_ok)]
            if len(t1_ok) > 10 and acc_p:
                G2 = np.stack([_load(edir, t1[s], f"{rec}_out_oracle") for s in t1_ok])
                P2 = np.stack([_load(edir, i, f"{rec}_out_oracle") for i in acc_p])
                tr2 = np.array([t1_ok.index(man[i]["subject"]) for i in acc_p])
                C["matched_anonymized_gallery_fused"] = _summary(
                    _metrics(_fuse(_sim(P2, G2)), tr2), np.array([man[i]["subject"] for i in acc_p]),
                    rng, len(t1_ok))
            own = [t1[s] for s in gsub if ok(t1[s])]
            if own:
                S3 = _sim(np.stack([_load(edir, i, f"{rec}_out_oracle") for i in own]), G)
                tr3 = np.array([gi[man[i]["subject"]] for i in own])
                C["same_scan_bound_fused"] = _summary(_metrics(_fuse(S3), tr3),
                                                      np.array([man[i]["subject"] for i in own]), rng, len(gsub))
            R["configs"][cfg] = C
        # paired comparison on the probes every configuration was run on
        if runs:
            common = np.logical_and.reduce([cache[(c, "oracle")][2] for c in runs])
            P = {"probes": int(common.sum()), "subjects": int(len(np.unique(subj[common]))), "rows": {}}
            P["rows"]["control"] = _summary(_sub(ctrl["fused"][0], common), subj[common], rng, len(gsub))
            ma = None
            if "adaptive" in runs:
                ma = _sub(_metrics(_fuse(cache[("adaptive", "oracle")][0]), truth), common)
            for c in runs:
                S, acc, _ = cache[(c, "oracle")]
                m = _sub(_metrics(_fuse(S), truth), common)
                s = _summary(m, subj[common], rng, len(gsub))
                s["accepted"] = int(acc[common].sum())
                if c != "adaptive" and ma is not None:
                    s["diff_vs_adaptive"] = _paired_diff(m["rank"] == 1, ma["rank"] == 1, subj[common], rng)
                P["rows"][c] = s
            R["paired_sample"] = P
        res["recognizers"][rec] = R
    return res


def refused_faces(dataset: str, config: str = "adaptive") -> dict:
    """How many scans the tool refused still show a face: YuNet finds one on
    the unmodified input although the tool's own detector did not."""
    gz = RESULTS / "embeddings" / f"{dataset}__gallery.npz"
    runs = {r["id"]: r for r in read_jsonl(RESULTS / "runs" / f"{dataset}__{config}.jsonl")}
    if not gz.exists() or not runs:
        return {}
    g = np.load(gz)
    face = {i: bool(np.isfinite(e[MULTS.index(1.0), :, 0].astype(np.float32)).any())
            for i, e in zip(g["ids"].tolist(), g["arcface"])}
    ref = [i for i, r in runs.items() if r.get("status") == "refused"]
    return {"refused": len(ref), "refused_with_detectable_face": sum(face.get(i, False) for i in ref),
            "ids": sorted(i for i in ref if face.get(i, False))}


def main() -> None:
    out = {"ixi": analyse_ixi(), "refused_faces": {d: refused_faces(d) for d in ("synthstrip", "ixi")}}
    p = RESULTS / "reid_summary.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
