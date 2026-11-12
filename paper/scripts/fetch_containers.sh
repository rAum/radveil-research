#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0
#
# Baseline defacers and the original SynthStrip as pinned Docker images
# (see bench/baselines.py). Writes the resolved digests to
# results/manifests/containers.json.
#
#   DATA_DIR=/big/disk paper/scripts/fetch_containers.sh
#
# Needs Docker. Images are linux/amd64 (run under emulation on Apple silicon).
# mri_deface is no longer served by FreeSurfer; its 2007 binary and atlases
# are fetched from Internet Archive captures of
# surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/ and checked by SHA-256.
set -euo pipefail
DATA_DIR=${DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}
PAPER=$(cd "$(dirname "$0")/.." && pwd)
PLAT=linux/amd64
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

pull() { docker pull -q --platform $PLAT "$1" >/dev/null; echo "pulled $1"; }
pull freesurfer/synthstrip@sha256:16181aeb75cab4fa0bf7ab4f79ba77eca466cc0e28a384f3ecb007c4a376ea09
pull poldracklab/pydeface@sha256:c648b8d70f31a186ff003ed2ad15d491d7b4a7d9adaa4665cf95526415fe3773

ctx="$DATA_DIR/containers"
mkdir -p "$ctx"
ARCHIVE=https://web.archive.org/web
fetch() {  # <name> <wayback timestamp> <sha256>
  local f="$ctx/$1"
  [[ -f "$f" ]] || curl -fsSL --retry 5 -o "$f" "$ARCHIVE/${2}id_/https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/$1"
  echo "$3  $f" | sha256 -c - || { rm -f "$f"; exit 1; }
}
fetch mri_deface_linux 20250331071620 b35fee28977399754f49970365dd634d0ae60f6178d15169c9ebae25c8ba9cf2
fetch face.gca.gz 20250206201518 f72f92579771b913e906d4197e32e59f4b214806ff0193e57d0ee7273399cc95
fetch talairach_mixed_with_skull.gca.gz 20250206171133 9e061c872b45db73536eff56a84045c29117b6e57412ee19a879404edd6c9aa8
for g in face.gca talairach_mixed_with_skull.gca; do [[ -f "$ctx/$g" ]] || gunzip -k "$ctx/$g.gz"; done
docker build --platform $PLAT -q -t radveil-paper/mri_deface:2007 -f "$PAPER/bench/docker/mri_deface/Dockerfile" "$ctx" >/dev/null
echo "built radveil-paper/mri_deface:2007"

# FSL FAST (tissue segmentation for the downstream check) runs natively, not
# emulated: a pinned micromamba creates $DATA_DIR/containers/fsl/env with
# fsl-fast4 from FSL's conda channel (FSL licence: non-commercial use).
fsl="$ctx/fsl"
if [[ ! -x "$fsl/env/bin/fast" ]]; then
  mkdir -p "$fsl"
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) plat=osx-arm64; mm_sha=8b5c314fa5bcfb5476c1af4b9c8f9ce87a4f6e3cb89bb0d792965dac2904c921 ;;
    Linux-x86_64) plat=linux-64; mm_sha="" ;;
    *) echo "no FSL build for $(uname -s)-$(uname -m)"; exit 1 ;;
  esac
  curl -fsSL -o "$fsl/micromamba.tar.bz2" \
    "https://api.anaconda.org/download/conda-forge/micromamba/2.0.5/$plat/micromamba-2.0.5-0.tar.bz2"
  [[ -z "$mm_sha" ]] || echo "$mm_sha  $fsl/micromamba.tar.bz2" | sha256 -c -
  tar -xjf "$fsl/micromamba.tar.bz2" -C "$fsl" bin/micromamba
  MAMBA_ROOT_PREFIX="$fsl/root" "$fsl/bin/micromamba" create -y -q -p "$fsl/env" \
    -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ -c conda-forge "fsl-fast4=2111.3"
fi
FAST_PKG=$(MAMBA_ROOT_PREFIX="$fsl/root" "$fsl/bin/micromamba" list -p "$fsl/env" 2>/dev/null | awk '$1=="fsl-fast4"{print $2"-"$3}')

out="$PAPER/results/manifests/containers.json"
SS=freesurfer/synthstrip@sha256:16181aeb75cab4fa0bf7ab4f79ba77eca466cc0e28a384f3ecb007c4a376ea09
PD=poldracklab/pydeface@sha256:c648b8d70f31a186ff003ed2ad15d491d7b4a7d9adaa4665cf95526415fe3773
cat >"$out" <<JSON
{
  "platform": "$PLAT",
  "synthstrip": {"ref": "$SS", "image_id": "$(docker image inspect --format '{{.Id}}' $SS)"},
  "pydeface": {"ref": "$PD", "image_id": "$(docker image inspect --format '{{.Id}}' $PD)"},
  "mri_deface": {"ref": "radveil-paper/mri_deface:2007",
                 "image_id": "$(docker image inspect --format '{{.Id}}' radveil-paper/mri_deface:2007)",
                 "files_sha256": {"mri_deface_linux": "b35fee28977399754f49970365dd634d0ae60f6178d15169c9ebae25c8ba9cf2",
                                  "face.gca.gz": "f72f92579771b913e906d4197e32e59f4b214806ff0193e57d0ee7273399cc95",
                                  "talairach_mixed_with_skull.gca.gz": "9e061c872b45db73536eff56a84045c29117b6e57412ee19a879404edd6c9aa8"}},
  "fsl_fast": {"package": "fsl-fast4 $FAST_PKG", "micromamba": "2.0.5", "native": true}
}
JSON
cat "$out"
