# RadVeil research

Experiments, benchmarks and the evaluation paper for
[RadVeil](https://github.com/rAum/radveil), the attacker-model face
anonymization tool for head MRI/CT.

Everything lives under [`paper/`](paper/):

- `paper/README.md` — layout, how the chain fits together
- `paper/scripts/` — dataset, model and container downloads (checksummed)
- `paper/bench/` — the benchmark: coverage, integrity, brain safety,
  attacks (detection, re-identification, blur inversion, head-shape
  linkage), downstream effects, profiling
- `paper/analysis/` — turns `paper/results/` into every number, table and
  figure in the manuscript
- `paper/tex/` — the manuscript; no number in it is typed by hand

Quick start:

```
make -C paper pdf-only                 # build the PDF from committed results (no data, no Rust)
make -C paper data DATA_DIR=/big/disk  # IXI (~13 GB) + SynthStrip (~7 GB) + models + containers
make -C paper all  DATA_DIR=/big/disk  # reproduce everything (~12 h on a laptop)
```

Face embeddings and head-shape descriptors are biometric templates of
dataset participants: they are computed locally and never committed or
distributed. Licences: code Apache-2.0; manuscript and IXI-derived results
CC BY-SA 4.0 (see `paper/tex/radveil.tex` and `paper/results/LICENSE`).
