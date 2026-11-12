# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0
#
# Reproduce the RadVeil paper: data -> runs -> analysis -> PDF.
#
#   make -C paper env                     # venv (hashed lock) + radveil CLI/Python from this checkout
#   make -C paper data DATA_DIR=/big/disk # IXI (~13 GB) + SynthStrip (~7 GB) + models + containers
#   make -C paper all  DATA_DIR=/big/disk # every run, profile, analysis, PDF (~12 h on a laptop)
#   make -C paper pdf-only                # PDF from the committed results: no Rust, no data
#   make -C paper smoke DATA_DIR=...      # a few cases through every stage (~10 min)
#
# Every number in the paper is generated from results/ by analysis/; nothing
# is typed into the .tex by hand. See README.md.

.NOTPARALLEL:
SHELL    := bash
DATA_DIR ?= $(CURDIR)/data
JOBS     ?= 4
THREADS  ?= 3
PY       := $(CURDIR)/.venv/bin/python
RUN      := $(PY) bench/run_cohort.py --data-dir $(DATA_DIR) --jobs $(JOBS) --threads $(THREADS) --qa --verify-data

# SynthStrip set: every RadVeil configuration with the attack; downstream
# skull-strip comparison for adaptive and cut. Baselines run in emulated x86
# containers on Apple silicon (pydeface ~4 min/scan, mri_deface ~1 min):
# mri_deface and Quickshear on every structural scan, pydeface on the scans
# RadVeil accepted (the paired set every comparison uses).
SYNTHSTRIP_ATTACK := adaptive adaptive_noverify blur cut adaptive_ears
SYNTHSTRIP_PLAIN  := adaptive_guard adaptive_robust
DOWNSTREAM        := adaptive cut
SEG               := --seg 0.2  # FAST segmentation on a fixed 20% of downstream cases
STRUCTURAL        := --filter modality=t1,t2,pd,flair,qt1
BJOBS             ?= 8
BRUN              := $(PY) bench/run_cohort.py --data-dir $(DATA_DIR) --jobs $(BJOBS) --threads 2 --qa --verify-data
# IXI: RadVeil on every volume; baselines on a fixed random sample of
# subjects, on which every tool also gets an independent brain reference
# (original SynthStrip on the input) for paired comparisons
IXI_CONFIGS       := adaptive_noverify cut adaptive_ears
IXI_SUBJECTS      ?= 100

.PHONY: all env analysis-env cli containers data manifests manifests-update gallery synthstrip ixi rust \
        deconv profile figures qa analysis pdf pdf-only smoke

all: env manifests gallery synthstrip ixi rust deconv profile figures qa analysis pdf

env: analysis-env cli
	uv pip install -q -p .venv --reinstall-package radveil ..

analysis-env: .venv/.lock-ok
.venv/.lock-ok: requirements.lock
	uv venv --clear -q .venv -p 3.12
	uv pip sync -q -p .venv requirements.lock
	touch $@

cli:
	cd .. && cargo build --release --locked -p radveil-cli

data:
	DATA_DIR=$(DATA_DIR) scripts/fetch_synthstrip.sh
	DATA_DIR=$(DATA_DIR) scripts/fetch_ixi.sh
	DATA_DIR=$(DATA_DIR) scripts/fetch_models.sh
	$(MAKE) containers

containers:
	@docker info >/dev/null 2>&1 || { echo "Docker is needed for the baselines"; exit 1; }
	DATA_DIR=$(DATA_DIR) scripts/fetch_containers.sh

# check the data on disk against the committed manifests (never rewrites them)
manifests:
	$(PY) scripts/prepare_datasets.py --data-dir $(DATA_DIR) --check

# regenerate the committed manifests: only when the data set is meant to change
manifests-update:
	$(PY) scripts/prepare_datasets.py --data-dir $(DATA_DIR)

gallery:
	$(PY) bench/gallery.py --dataset synthstrip --data-dir $(DATA_DIR)
	$(PY) bench/gallery.py --dataset ixi --data-dir $(DATA_DIR)
	$(PY) bench/shape_gallery.py --dataset ixi --data-dir $(DATA_DIR)

synthstrip:
	for c in $(SYNTHSTRIP_ATTACK); do \
	  ds=$$([[ " $(DOWNSTREAM) " == *" $$c "* ]] && echo --downstream); \
	  $(RUN) --dataset synthstrip --config $$c --attack $$ds $${ds:+$(SEG)} || exit 1; done
	for c in $(SYNTHSTRIP_PLAIN); do $(RUN) --dataset synthstrip --config $$c || exit 1; done
	for c in mri_deface quickshear; do \
	  $(BRUN) --dataset synthstrip --config $$c --attack --downstream $(SEG) $(STRUCTURAL) || exit 1; done
	$(BRUN) --dataset synthstrip --config pydeface --attack --downstream $(SEG) --only-accepted-by adaptive

ixi:
	$(RUN) --dataset ixi --config adaptive --attack --shape --brain-check --ref-brain $(IXI_SUBJECTS)
	for c in $(IXI_CONFIGS); do \
	  $(RUN) --dataset ixi --config $$c --attack --shape --ref-brain $(IXI_SUBJECTS) || exit 1; done
	for c in quickshear mri_deface pydeface; do \
	  $(BRUN) --dataset ixi --config $$c --attack --shape --subjects $(IXI_SUBJECTS) --ref-brain $(IXI_SUBJECTS) || exit 1; done

# blur-inversion attack on a sample of accepted adaptive IXI scans
deconv:
	$(PY) analysis/deconv_attack.py --data-dir $(DATA_DIR)

# RadVeil's Rust SynthStrip against the original (container) and the expert masks
rust:
	$(PY) bench/validate_rust.py --data-dir $(DATA_DIR)

# latency numbers: run alone on an idle machine
profile:
	$(PY) bench/profile.py --data-dir $(DATA_DIR)

# the one figure made from the data rather than from results/
figures:
	$(PY) analysis/make_figures.py --data-dir $(DATA_DIR)

# human review page for the per-case QA sheets: open results/qa/index.html
qa:
	$(PY) analysis/qa_gallery.py

# results/ -> summaries -> macros, tables, plots; the embedding/shape
# summaries are skipped when their git-ignored inputs are absent
analysis:
	@if [ -f results/embeddings/ixi__gallery.npz ]; then $(PY) analysis/reid.py; fi
	@if [ -f results/shape/ixi__gallery.npz ]; then $(PY) analysis/shape_summary.py; fi
	$(PY) analysis/make_assets.py $(if $(PENDING_OK),,--strict)

# fails on LaTeX errors and on unresolved references or citations
pdf:
	cd tex && pdflatex -interaction=nonstopmode -halt-on-error radveil.tex >/dev/null
	cd tex && bibtex radveil
	cd tex && pdflatex -interaction=nonstopmode -halt-on-error radveil.tex >/dev/null
	cd tex && pdflatex -interaction=nonstopmode -halt-on-error radveil.tex | grep "Output written"
	@! grep -E "(Reference|Citation) .* undefined" tex/radveil.log

pdf-only: analysis-env
	$(PY) analysis/make_assets.py $(if $(PENDING_OK),,--strict)
	$(MAKE) pdf

# a few cases through every stage, tagged __smoke so they never mix with
# the real results (the committed manifests are not touched)
smoke: export PENDING_OK = 1
smoke:
	$(RUN) --dataset synthstrip --config adaptive --attack --downstream --every 60 --tag smoke --allow-dirty
	$(RUN) --dataset synthstrip --config pydeface --attack --every 60 --tag smoke --allow-dirty
	$(RUN) --dataset ixi --config adaptive --attack --shape --brain-check --limit 3 --tag smoke --allow-dirty
	$(PY) bench/profile.py --data-dir $(DATA_DIR) --quick
	$(PY) analysis/qa_gallery.py
