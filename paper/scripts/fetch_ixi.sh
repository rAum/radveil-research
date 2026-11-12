#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0
#
# Download the IXI dataset (T1, T2, PD + demographics) into $DATA_DIR/ixi.
#
#   DATA_DIR=/big/disk paper/scripts/fetch_ixi.sh [T1 T2 PD]
#
# IXI (https://brain-development.org/ixi-dataset/) is CC BY-SA 3.0. The server
# publishes no checksums and resets long transfers, so:
#  * each tar is downloaded to $DATA_DIR/ixi/tars with HTTP range resume until
#    it has the advertised length (an interrupted run continues where it
#    stopped), then checked against EXPECTED below, the SHA-256 of the copy
#    the paper used, and only then extracted;
#  * prepare_datasets.py records the SHA-256 of every extracted .nii.gz in the
#    committed manifests and `run_cohort.py --verify-data` checks them, which
#    also validates copies obtained from a mirror.
# Contrasts download in parallel. Set KEEP_TARS=1 to keep the 13 GB of tars.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/sha256.sh"
DATA_DIR=${DATA_DIR:-$(cd "$here/.." && pwd)/data}
BASE=https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI
if [[ $# -gt 0 ]]; then CONTRASTS=("$@"); else CONTRASTS=(T1 T2 PD); fi

# SHA-256 of the files as downloaded for the paper ("" = not pinned)
expected() {
  case "$1" in
    IXI.xls) echo 8880c4417d8c22f9489439f1b8977b1003ed7908be0294e44462aa71c2a3e78f ;;
    T1) echo 56bbe911fbe967f7d1b830a552995263bdba281e0a6eb78e62f1c38d88a946a9 ;;
    T2) echo 39af50f87be8f7d0be2f166a64f5f59b5e43577255eb5613553dae37c348c2e4 ;;
    PD) echo c00124d2409b68f9800938ede547cff71cafb0d51e19bd5138d85a4c4e144130 ;;
  esac
}

out="$DATA_DIR/ixi"
mkdir -p "$out/tars"
[[ -f "$out/IXI.xls" ]] || curl -fsSL --retry 5 -o "$out/IXI.xls" "$BASE/IXI.xls"
want=$(expected IXI.xls); [[ -z "$want" ]] || verify "$out/IXI.xls" "$want"

fetch() {
  local c=$1 dir="$out/$1" tar="$out/tars/IXI-$1.tar"
  if [[ -f "$dir/.done" ]]; then echo "IXI-$c: already extracted"; return 0; fi
  local size; size=$(curl -fsSIL "$BASE/IXI-$c.tar" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
  [[ -n "$size" ]] || { echo "IXI-$c: no Content-Length from the server" >&2; return 1; }
  local tries=0
  while [[ ! -f "$tar" || $(wc -c <"$tar" | tr -d ' ') -lt $size ]]; do
    tries=$((tries + 1)); [[ $tries -le 200 ]] || { echo "IXI-$c: giving up" >&2; return 1; }
    curl -fsSL -C - --retry 10 --retry-all-errors --retry-delay 10 -o "$tar" "$BASE/IXI-$c.tar" || sleep 10
  done
  local got; got=$(sha256 "$tar")
  local want; want=$(expected "$c")
  if [[ -n "$want" && "$got" != "$want" ]]; then
    echo "IXI-$c: checksum $got does not match the paper's $want; deleting the tar" >&2; rm -f "$tar"; return 1
  fi
  rm -rf "$dir"; mkdir -p "$dir"
  tar -x -C "$dir" -f "$tar"
  echo "$got  IXI-$c.tar" >"$dir/.sha256"
  touch "$dir/.done"
  [[ "${KEEP_TARS:-0}" == 1 ]] || rm -f "$tar"
  echo "IXI-$c: $(ls "$dir" | wc -l | tr -d ' ') volumes, sha256 $got"
}

pids=() fail=0
for c in "${CONTRASTS[@]}"; do fetch "$c" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
cat "$out"/*/.sha256 >"$out/SHA256" 2>/dev/null || true
[[ $fail == 0 ]] || { echo "some contrasts failed; rerun to resume them" >&2; exit 1; }
echo "IXI ready in $out"; cat "$out/SHA256"
