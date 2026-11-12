#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0
#
# Download the SynthStrip evaluation set (Hoopes et al. 2022) into
# $DATA_DIR/synthstrip. It ships expert brain masks for all 622 scans and
# SynthSeg/FreeSurfer label maps for the adult MPRAGE scans.
#
#   DATA_DIR=/big/disk paper/scripts/fetch_synthstrip.sh
#
# Already have the tar or the extracted folder? Point SYNTHSTRIP_TAR at the
# tar (checked against the published SHA-256) or symlink the extracted
# folder to $DATA_DIR/synthstrip/synthstrip_data_v1.5.
# License: CC BY 4.0 (IXI part CC BY-SA 4.0).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/sha256.sh"
DATA_DIR=${DATA_DIR:-$(cd "$here/.." && pwd)/data}
URL=https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/data/synthstrip_data_v1.5.tar
SHA=c9f070c40266790d97fce35b8864dfcd96f039b598311030c9d707671b3b0aba

out="$DATA_DIR/synthstrip"
mkdir -p "$out"
if [[ -f "$out/.done" || -L "$out/synthstrip_data_v1.5" ]]; then
  echo "SynthStrip already extracted in $out"; exit 0
fi
tar_path=${SYNTHSTRIP_TAR:-$out/synthstrip_data_v1.5.tar}
if [[ ! -f "$tar_path" ]]; then
  curl -fSL --retry 5 -C - -o "$tar_path" "$URL"
fi
if [[ -n "${SYNTHSTRIP_TAR:-}" ]]; then verify "$tar_path" "$SHA" keep; else verify "$tar_path" "$SHA"; fi
tar -x -C "$out" -f "$tar_path"
touch "$out/.done"
echo "SynthStrip ready in $out/synthstrip_data_v1.5"
