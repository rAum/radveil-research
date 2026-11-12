# RadVeil paper: data, benchmarks, manuscript

Everything needed to reproduce, check or extend the RadVeil evaluation
paper ([`tex/radveil.tex`](tex/radveil.tex), built as `tex/radveil.pdf`).
Every number in the PDF is a macro generated from the per-case results in
[`results/`](results/); none is typed into the manuscript by hand.

```
paper/
├── Makefile                  one target per stage (see below)
├── requirements.txt          top-level pins; requirements.lock = full hashed set
├── scripts/                  datasets, models, containers
│   ├── sha256.sh             portable checksum helpers (sha256sum or shasum)
│   ├── fetch_ixi.sh          IXI T1/T2/PD + demographics, resumable, checksummed
│   ├── fetch_synthstrip.sh   SynthStrip evaluation set v1.5, checksummed
│   ├── fetch_models.sh       attackers: YuNet, SFace, ArcFace (pinned commit + SHA-256)
│   ├── fetch_containers.sh   baselines + original SynthStrip (pinned containers), FSL FAST (native)
│   └── prepare_datasets.py   -> results/manifests/ (per-image SHA-256, development flag)
├── bench/                    measurements (write results/)
│   ├── common.py             configurations, timed runs, integrity + brain metrics
│   ├── run_cohort.py         one tool x one dataset; one JSON line per case; resumable
│   ├── baselines.py          pydeface, mri_deface, Quickshear (containers)
│   ├── downstream.py         original SynthStrip on input vs output; FAST tissue segmentation
│   ├── validate_rust.py      RadVeil's Rust SynthStrip vs the original and expert masks
│   ├── attack.py             detection probe + SFace/ArcFace embeddings at 4 depths
│   ├── gallery.py            embeddings of every unmodified input (re-ID gallery)
│   ├── shape.py              head-shape descriptor (recognizer-free linkage attack)
│   ├── shape_gallery.py      descriptors of every unmodified input
│   ├── qa.py                 one visual QA sheet per case
│   └── profile.py            latency distribution, stages, threads, SIMD, effort, throughput
├── analysis/                 results/ -> tex/generated/ + tex/figures/
│   ├── reid.py               face re-ID analysis -> results/reid_summary.json
│   ├── shape_reid.py         shape matching (radial maps + ICP)
│   ├── shape_summary.py      shape linkage (incl. pose-perturbed control) -> results/shape_summary.json
│   ├── deconv_attack.py      blur inversion with the known kernel -> results/deconv_summary.json
│   ├── make_assets.py        macros, tables, plots, results/summary.json (--strict)
│   ├── qa_gallery.py         results/qa/index.html: browse, filter, review every case
│   └── make_figures.py       Figure 1 (needs the data; output committed)
├── results/                  the record (see "Result files"); LICENSE: CC BY-SA 4.0
└── tex/                      radveil.tex, refs.bib, generated/, figures/, radveil.pdf
```

## Reproduce

Requirements: Rust ≥ 1.98, [uv](https://docs.astral.sh/uv/), Docker, a TeX
distribution with `pdflatex` + `bibtex`, ~60 GB free disk during download
(30 GB after), macOS or Linux. On Linux install GNU `time` (peak RSS).

```bash
make -C paper env                         # .venv from requirements.lock + radveil CLI/Python
make -C paper data DATA_DIR=/big/disk     # downloads (resumable; rerun after a failure)
make -C paper all  DATA_DIR=/big/disk     # every run, profile, analysis, PDF
```

The PDF from the committed results only (no data, no Rust):

```bash
make -C paper pdf-only
```

A ten-minute check of every stage on a handful of cases (written with a
`__smoke` tag, never mixed with the real results):

```bash
make -C paper smoke DATA_DIR=/big/disk
```

Runs refuse to start on a dirty working tree (anything that affects results:
`crates/`, `python/`, `Cargo.lock`, `Cargo.toml`, `pyproject.toml`,
`paper/bench`, `paper/scripts`); every record carries the code revision and
its attempt number, and resuming with a different revision is refused.

Data integrity: `make manifests` rebuilds the manifests from your data in a
temporary directory and fails if any row or checksum differs from the
committed ones (it never rewrites them; `make manifests-update` does, only
when the data set is meant to change). `--verify-data` (on in `make`) checks
every image, expert mask and label map against the manifest before a run.
The paper used Rust 1.98 (recorded per run in `.meta.json`).

### Stages and cost (Apple M4 Pro, 14 cores)

| target | what | time |
|---|---|---|
| `data` | IXI ~13 GB (slow server), SynthStrip ~7 GB, models ~0.5 GB, containers | 2–4 h |
| `gallery` | face embeddings + shape descriptors of every input | ~30 min |
| `synthstrip` | 7 RadVeil configurations × 582 scans; baselines on structural scans (pydeface on RadVeil-accepted ones); downstream + FAST on a 20% sample | ~6 h |
| `ixi` | 4 RadVeil configurations × 1,700+ volumes; baselines on 100 subjects; attack + shape; independent brain reference on the sample | ~8 h |
| `rust`, `deconv`, `profile` | validation; blur inversion; latency (**run on an idle machine**) | ~2 h |

Baseline containers are x86-64: on Apple silicon they run emulated and the
original SynthStrip needs ~7 GB in Docker's VM, so containers share a memory
budget (`RADVEIL_DOCKER_BUDGET_GB`, default 7) and original-SynthStrip masks
of the inputs are cached in `~/.cache/radveil-paper/synthstrip` (keyed by
input hash and image digest). FSL FAST runs natively from a pinned conda
environment under `$DATA_DIR/containers/fsl`.
| `analysis`, `pdf` | | ~20 min (shape ICP) |

## Result files

- `manifests/*.csv` — one row per case: paths relative to `DATA_DIR`,
  SHA-256, IXI site/scanner/sex/age and whether the subject was used during
  RadVeil's development (`development`), SynthStrip subset/modality/split and
  label availability. `datasets.json`: counts and archive checksums;
  `containers.json`: baseline container digests.
- `runs/<dataset>__<config>.jsonl` — per case, sorted by id: `status` (ok /
  refused with `reason` / error), the tool's report without face geometry,
  wall time and peak RSS, integrity (incl. changed tissue volume), brain
  metrics, downstream skull-strip comparison, detection summary and the views
  that fired, QA flag, code revision. `.meta.json`: every session that wrote
  to the run, with flags and environment.
- `reid_summary.json`, `shape_summary.json` — re-identification results.
- `rust_synthstrip.jsonl`, `profile.json`, `summary.json`, `qa_review.csv`.

Not committed, recomputed locally: `embeddings/` and `shape/` (biometric
templates of dataset participants) and `qa/` (sheets, a few hundred MB).

## Visual review

Every case leaves `results/qa/<run>/<case>.jpg`: input and output renders at
−40°, 0°, +40°; sagittal, axial and coronal slices with the region (amber)
and expert brain mask (blue); up to two output views in which an attacker's
detector fired, with its box. Refused cases show the input and the reason.

`make -C paper qa` builds `results/qa/index.html`. Open it in a browser:
filter by run, status, escalation, attacker hit, distance to the brain or
review state; click a sheet to page through with ←/→ and mark it **o** (looks
right) or **p** (problem, with a note). "Copy review CSV" exports the
verdicts; save them as `results/qa_review.csv` and commit. The paper reports
how many sheets were reviewed and flagged.

## Extending

- **Configuration**: add flags to `CONFIGS` in `bench/common.py`; add the name
  to the Makefile lists and to `CONFIG_LABEL` in `analysis/make_assets.py`.
- **Baseline defacer**: add it to `TOOLS` in `bench/baselines.py` with a pinned
  container; the region is derived as the changed voxels.
- **Dataset**: add a `<name>_manifest()` to `scripts/prepare_datasets.py`
  (`id`, `image`, `sha256`, optional `brain_mask`, `labels`); re-ID needs
  several scans per subject (`analysis/reid.py`).
- **Number in the text**: compute it in `make_assets.py`, `macro("name", v)`,
  write `\name` in the .tex. Unregistered names render as **??** and are
  listed; `make_assets.py --strict` fails on them.

## Open items

- Review the QA sheets of the final runs and commit `results/qa_review.csv`.
- Generative refacing attack (Abramian & Eklund 2019) on the anonymized region.
- Structure segmentation (SynthSeg/FreeSurfer; needs a FreeSurfer licence);
  tissue segmentation (FAST) is done.
- Photo-based re-identification (needs consented photographs).
- Pin the IXI tar checksums in `scripts/fetch_ixi.sh` (`expected`) from
  `$DATA_DIR/ixi/SHA256` after the first complete download.
- Move to the target journal's template late.

## Data, licences, ethics

IXI is CC BY-SA 3.0; the SynthStrip set CC BY 4.0 (IXI part CC BY-SA 4.0).
IXI is not defaced: faces in it are identifiable, which is why the paper can
measure re-identification and why renders, embeddings and shape descriptors
are not redistributed here. `results/` is CC BY-SA 4.0 (see
`results/LICENSE`); the manuscript too, because Figure 1 shows one IXI
subject. ArcFace weights (InsightFace buffalo_l) are for non-commercial
research only.
