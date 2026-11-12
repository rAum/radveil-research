# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Attack one anonymized volume: face detection over a dense grid of views
and iso-surfaces, and face-recognition embeddings for re-identification.

Detection (a proxy, reported but not the security metric):

* views: 7 yaws (-60..60 deg) x 3 pitches (-20, 0, 20 deg); RadVeil's own
  verification uses the detection yaws plus a 3x3 grid (balanced effort);
* iso-surfaces: the input's threshold x {0.5, 1, 2, 4} (84 output views);
* detectors: RadVeil's built-in MediaPipe BlazeFace + Face Mesh (the detector
  the pipeline verifies against; roll search on; its API returns one face
  per image) and OpenCV Zoo YuNet (independent; every face it returns).
  Scores down to 0.3 are stored; the analysis applies the operating
  threshold (0.5) so it can also re-threshold.

A fire is *on-face* when its box centre lies within half an input face-box
width of the input's face in the same view. Views where the input itself
has no detection cannot be classified and count as off-face. Per view the
best on-face fire and the best fire overall are stored separately, so
either can be re-thresholded.

Recognition (the security metric, analysed by analysis/reid.py):

* renders at yaw -40/-20/0/+20/+40, pitch 0, at the input threshold x
  {0.5, 1, 2, 4} (RECOG_MULT): the deeper surfaces lie below RadVeil's
  skin-deep anonymized layer, where the face shape survives;
* recognizers: OpenCV Zoo SFace and InsightFace ArcFace (w600k_r50, the
  buffalo_l pack; non-commercial weights), both on the standard 5-point
  112x112 alignment from YuNet landmarks;
* alignment of the output, stored separately so the analysis can take the
  better one per comparison: "oracle" aligns on the input's face location
  (the attacker knows where the face was), "near" on the output's own YuNet
  face where one lies on that location (NaN otherwise), "blind" on the best
  YuNet face anywhere on the output (NaN when there is none).

Returned embeddings, keyed "<rec>_<in|out_oracle|out_near|out_blind>", have shape
(len(RECOG_MULT), len(RECOG_YAWS), dim), NaN where no face could be aligned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

YAWS = (-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0)
PITCHES = (-20.0, 0.0, 20.0)
THRESHOLD_MULT = (0.5, 1.0, 2.0, 4.0)
RECOG_YAWS = (-40.0, -20.0, 0.0, 20.0, 40.0)  # pitch 0
RECOG_MULT = (0.5, 1.0, 2.0, 4.0)
SIZE = 512
STORE_MIN_SCORE = 0.3  # stored; analysis applies OPERATING_SCORE
OPERATING_SCORE = 0.5

MODEL_FILES = {
    "yunet": "face_detection_yunet_2023mar.onnx",
    "sface": "face_recognition_sface_2021dec.onnx",
    "arcface": "buffalo_l/w600k_r50.onnx",
}
RECOGNIZERS = ("sface", "arcface")

# standard 5-point template for 112x112 face crops (ArcFace / SFace)
_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                      [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

_MODELS: dict = {}


def missing_models(models_dir: str) -> list[str]:
    return [f for f in MODEL_FILES.values() if not (Path(models_dir) / f).exists()]


def _model(kind: str, models_dir: str):
    if kind not in _MODELS:
        import cv2

        cv2.setNumThreads(1)
        path = str(Path(models_dir) / MODEL_FILES[kind])
        if kind == "yunet":
            _MODELS[kind] = cv2.FaceDetectorYN.create(path, "", (SIZE, SIZE), STORE_MIN_SCORE, 0.3, 50)
        elif kind == "sface":
            _MODELS[kind] = cv2.FaceRecognizerSF.create(path, "")
        else:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            _MODELS[kind] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _MODELS[kind]


def _bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[..., ::-1])


def yunet_faces(rgb: np.ndarray, models_dir: str) -> np.ndarray:
    """YuNet rows (x, y, w, h, 5 landmarks, score), shape (n, 15)."""
    det = _model("yunet", models_dir)
    det.setInputSize((rgb.shape[1], rgb.shape[0]))
    _, faces = det.detect(_bgr(rgb))
    return np.zeros((0, 15), np.float32) if faces is None else faces


def _center(f) -> np.ndarray:
    return np.array([f[0] + f[2] / 2, f[1] + f[3] / 2])


def _aligned(rgb: np.ndarray, face_row: np.ndarray) -> np.ndarray:
    """112x112 BGR crop by a similarity transform of the 5 landmarks."""
    import cv2

    src = np.asarray(face_row[4:14], np.float32).reshape(5, 2)
    m, _ = cv2.estimateAffinePartial2D(src, _TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(_bgr(rgb), m, (112, 112), borderValue=0)


def embed(rgb: np.ndarray, face_row: np.ndarray, rec: str, models_dir: str) -> np.ndarray:
    """L2-normalised embedding of the face aligned by ``face_row``."""
    crop = _aligned(rgb, face_row)
    if rec == "sface":
        e = _model("sface", models_dir).feature(crop).ravel()
    else:
        x = (crop[..., ::-1].astype(np.float32) - 127.5) / 127.5  # RGB, [-1, 1]
        sess = _model("arcface", models_dir)
        e = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0].ravel()
    e = e.astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-12)


def _mp_boxes(rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
    from radveil._radveil import detect_face

    r = detect_face(np.ascontiguousarray(rgb), try_rotations=True, min_score=STORE_MIN_SCORE)
    if r is None:
        return []
    lm, score = r
    lm = np.asarray(lm)
    lo, hi = np.nanmin(lm, axis=0), np.nanmax(lm, axis=0)
    c = (lo + hi) / 2
    return [(float(c[0]), float(c[1]), float(max(hi - lo)), float(score))]


def _yunet_boxes(rgb: np.ndarray, models_dir: str) -> list[tuple[float, float, float, float]]:
    """(cx, cy, w, score) per face."""
    return [(f[0] + f[2] / 2, f[1] + f[3] / 2, f[2], float(f[14])) for f in yunet_faces(rgb, models_dir)]


DETECTORS = {"mediapipe": lambda rgb, m: _mp_boxes(rgb), "yunet": _yunet_boxes}


def embed_volume(vol, models_dir: str, ref: dict | None = None):
    """Recognition embeddings of one volume over RECOG_MULT x RECOG_YAWS.

    Without ``ref`` (a clean input): align on the volume's own best YuNet
    face; returns ({rec: array}, faces) where faces[(k, yaw)] is that face row
    and thresholds[yaw] the base threshold. With ``ref`` (an output; ref is the
    input's (faces, thresholds)): returns {rec_oracle: ..., rec_blind: ...}."""
    shape = (len(RECOG_MULT), len(RECOG_YAWS))
    if ref is None:
        out = {r: np.full(shape + (512 if r == "arcface" else 128,), np.nan, np.float32) for r in RECOGNIZERS}
        faces, thr = {}, {}
        for j, yaw in enumerate(RECOG_YAWS):
            thr[yaw] = float(vol.render(yaw_deg=yaw, size=SIZE)["threshold"])
            for i, k in enumerate(RECOG_MULT):
                rgb = vol.render(yaw_deg=yaw, size=SIZE, threshold=thr[yaw] * k)["rgb"]
                f = yunet_faces(rgb, models_dir)
                if len(f):
                    best = f[np.argmax(f[:, 14])]
                    faces[(k, yaw)] = best
                    for r in RECOGNIZERS:
                        out[r][i, j] = embed(rgb, best, r, models_dir)
        return out, (faces, thr)
    faces, thr = ref
    out = {f"{r}_{v}": np.full(shape + (512 if r == "arcface" else 128,), np.nan, np.float32)
           for r in RECOGNIZERS for v in ("oracle", "near", "blind")}
    for j, yaw in enumerate(RECOG_YAWS):
        for i, k in enumerate(RECOG_MULT):
            rgb = vol.render(yaw_deg=yaw, size=SIZE, threshold=thr[yaw] * k)["rgb"]
            fo = yunet_faces(rgb, models_dir)
            blind = fo[np.argmax(fo[:, 14])] if len(fo) else None
            # the input's face at this view (its own surface level, else the 1x level)
            f0 = faces.get((k, yaw), faces.get((1.0, yaw)))
            near = None
            if f0 is not None:
                cand = [f for f in fo if np.linalg.norm(_center(f) - _center(f0)) < 0.5 * f0[2]]
                near = max(cand, key=lambda f: f[14]) if cand else None
            for r in RECOGNIZERS:
                for v, f in (("oracle", f0), ("near", near), ("blind", blind)):
                    if f is not None:
                        out[f"{r}_{v}"][i, j] = embed(rgb, f, r, models_dir)
    return out


def probe(src: str, out: str, models_dir: str) -> dict:
    """Attack one input/output pair: detection summary, the views that fired,
    and recognition embeddings of input and output."""
    import radveil

    vin, vout = radveil.Volume(src), radveil.Volume(out)
    views = []
    for yaw in YAWS:
        for pitch in PITCHES:
            base = vin.render(yaw_deg=yaw, pitch_deg=pitch, size=SIZE)
            thr = float(base["threshold"])
            ref = {name: f(base["rgb"], models_dir) for name, f in DETECTORS.items()}
            for k in THRESHOLD_MULT:
                r = vout.render(yaw_deg=yaw, pitch_deg=pitch, size=SIZE, threshold=thr * k)
                for name, f in DETECTORS.items():
                    hits = f(r["rgb"], models_dir)
                    ins = [b for b in ref[name] if b[3] >= OPERATING_SCORE]
                    rin = max(ins, key=lambda b: b[3]) if ins else None
                    face_hits = [h for h in hits if rin is not None
                                 and np.hypot(h[0] - rin[0], h[1] - rin[1]) < 0.5 * rin[2]]
                    bf = max(face_hits, key=lambda h: h[3]) if face_hits else None
                    ba = max(hits, key=lambda h: h[3]) if hits else None
                    views.append({
                        "det": name, "yaw": yaw, "pitch": pitch, "thr_mult": k,
                        "in_score": None if rin is None else round(rin[3], 3),
                        # best fire on the face, and best fire anywhere (may be the same)
                        "out_score": None if bf is None else round(bf[3], 3),
                        "on_face": None if bf is None else True,
                        "any_score": None if ba is None else round(ba[3], 3),
                        # box centre and width in render pixels, for the QA sheets
                        "out_box": None if bf is None else [round(float(x), 1) for x in bf[:3]],
                        "in_box": None if rin is None else [round(float(x), 1) for x in rin[:3]],
                    })
    summary = {}
    for name in DETECTORS:
        v = [x for x in views if x["det"] == name]
        summary[name] = {
            "views": len(v),
            "input_detected_views": sum(x["in_score"] is not None for x in v if x["thr_mult"] == 1.0),
            "output_any_views": sum((x["any_score"] or 0) >= OPERATING_SCORE for x in v),
            "output_residual_views": sum((x["out_score"] or 0) >= OPERATING_SCORE for x in v),
        }
    emb_in, ref = embed_volume(vin, models_dir)
    emb_out = embed_volume(vout, models_dir, ref)
    emb = {f"{r}_in": e for r, e in emb_in.items()}
    emb.update({f"{r.rsplit('_', 1)[0]}_out_{r.rsplit('_', 1)[1]}": e for r, e in emb_out.items()})
    return {"summary": summary, "views": views, "emb": emb}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--models", default=str(Path(__file__).resolve().parents[1] / "data" / "models"))
    a = ap.parse_args()
    r = probe(a.input, a.output, a.models)
    print(json.dumps(r["summary"], indent=2))
    for rec in RECOGNIZERS:
        a_, b_ = r["emb"][f"{rec}_in"], r["emb"][f"{rec}_out_oracle"]
        print(rec, "input/output cosine [mult x yaw]\n", np.round(np.nansum(a_ * b_, axis=-1), 3))
