#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0
#
# Download the attacker models used by bench/attack.py, pinned by commit and
# SHA-256:
#   YuNet   face detector (OpenCV Zoo, 2023mar, MIT): a CNN of a different
#           family and training set than RadVeil's built-in MediaPipe detector
#   SFace   face recognition (OpenCV Zoo, 2021dec, Apache-2.0)
#   ArcFace face recognition, InsightFace buffalo_l pack (w600k_r50). The
#           weights are licensed for non-commercial research only.
#
#   DATA_DIR=/big/disk paper/scripts/fetch_models.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/sha256.sh"
DATA_DIR=${DATA_DIR:-$(cd "$here/.." && pwd)/data}
out="$DATA_DIR/models"
mkdir -p "$out"
ZOO=https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models
fetch() {  # <url> <sha256> [file name]
  local f="$out/${3:-$(basename "$1")}"
  [[ -f "$f" ]] || curl -fsSL --retry 5 -o "$f" "$1"
  verify "$f" "$2"   # deletes the file on mismatch, so a rerun refetches it
}
fetch $ZOO/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
fetch $ZOO/face_recognition_sface/face_recognition_sface_2021dec.onnx \
  0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
fetch https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
  80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f
if [[ ! -f "$out/buffalo_l/w600k_r50.onnx" ]]; then
  mkdir -p "$out/buffalo_l"
  unzip -o -q "$out/buffalo_l.zip" w600k_r50.onnx -d "$out/buffalo_l"
fi
verify "$out/buffalo_l/w600k_r50.onnx" 4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43
