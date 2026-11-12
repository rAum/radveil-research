# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Visual QA sheet: one JPEG per case so a person can check every outcome
the numbers summarise.

Accepted case (4 columns):
  row 1  input surface renders at yaw -40, 0, +40 and the input mid-sagittal slice
  row 2  the same views of the output; the slice shows the anonymized region
         (amber) and, where the dataset has one, the expert brain mask (blue)
  row 3  an axial and a coronal slice through the region, then the output
         views in which an attacker's detector fired on the face, with its
         box drawn (red: on the original face location, grey: elsewhere)

Refused case: row 1 only, titled with the refusal reason, so a reviewer can
confirm there really was no usable face (EPI, DWI, a face cut by the field
of view, an already-defaced scan).

The title line carries what the run recorded: status, verify rounds,
escalation, closest approach to the brain, attacker hits.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

YAWS = (-40.0, 0.0, 40.0)
RENDER = 320
MAX_HITS = 2


def _canonical(path_or_img):
    import nibabel as nib

    img = nib.load(str(path_or_img)) if not hasattr(path_or_img, "affine") else path_or_img
    img = nib.as_closest_canonical(nib.squeeze_image(img))  # axes R, A, S
    return np.asarray(img.dataobj, np.float32), img.header.get_zooms()[:3]


def _slices(vol, idx):
    """Sagittal, axial and coronal planes of a RAS volume at voxel idx, for
    display: sagittal with superior up and anterior right; axial (anterior
    up) and coronal (superior up) in radiological convention, patient right
    on the viewer's left."""
    x, y, z = idx
    sag = vol[x, :, :].T[::-1]
    axi = vol[:, :, z].T[::-1, ::-1]
    cor = vol[:, y, :].T[::-1, ::-1]
    return sag, axi, cor


def _show(ax, img, spacing_hw, mask=None, brain=None, title=None):
    lo, hi = np.percentile(img, (1, 99.5)) if img.size else (0, 1)
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi if hi > lo else lo + 1,
              aspect=spacing_hw[0] / spacing_hw[1], interpolation="nearest")
    if brain is not None and brain.any():
        ax.contour(brain.astype(float), levels=[0.5], colors="#3b82f6", linewidths=0.6)
    if mask is not None and mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors="#f59e0b", linewidths=0.7)
    if title:
        ax.set_title(title, fontsize=7)


def sheet(dest: Path, rec: dict, src: str, out: str | None = None, mask_path: str | None = None,
          brain_path: str | None = None, label: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import radveil

    vin = radveil.Volume(src)
    ok = out is not None
    rows = 3 if ok else 1
    fig, axs = plt.subplots(rows, 4, figsize=(8, 2.15 * rows + 0.5), squeeze=False)
    for ax in axs.flat:
        ax.set_axis_off()

    din, sp = _canonical(src)
    renders_in = [vin.render(yaw_deg=y, size=RENDER) for y in YAWS]
    for j, (y, r) in enumerate(zip(YAWS, renders_in)):
        axs[0, j].imshow(r["rgb"])
        axs[0, j].set_title(f"input, yaw {y:+.0f}°", fontsize=7)

    if not ok:
        mid = tuple(s // 2 for s in din.shape)
        sag = _slices(din, mid)[0]
        _show(axs[0, 3], sag, (sp[2], sp[1]), title="input, mid-sagittal")
    else:
        import nibabel as nib

        dout, _ = _canonical(out)
        m, _ = _canonical(nib.load(mask_path))
        m = m > 0
        b = None
        if brain_path:
            b, _ = _canonical(brain_path)
            b = (b > 0) if b.shape == m.shape else None
        if m.any():
            ijk = np.argwhere(m)
            # coronal near the region's back wall (the skin it cut), axial through its centre
            c = (din.shape[0] // 2, int(np.percentile(ijk[:, 1], 15)), int(np.round(ijk[:, 2].mean())))
        else:
            c = tuple(s // 2 for s in din.shape)
        mid = c  # sagittal at the head midline
        si = _slices(din, mid)
        so = _slices(dout, mid)
        sm = _slices(m, mid)
        sb = _slices(b, mid) if b is not None else (None, None, None)
        _show(axs[0, 3], si[0], (sp[2], sp[1]), title="input, mid-sagittal")
        vout = radveil.Volume(out)
        for j, y in enumerate(YAWS):
            axs[1, j].imshow(vout.render(yaw_deg=y, size=RENDER, threshold=renders_in[j]["threshold"])["rgb"])
            axs[1, j].set_title(f"output, yaw {y:+.0f}°", fontsize=7)
        _show(axs[1, 3], so[0], (sp[2], sp[1]), sm[0], sb[0], "output, sagittal")
        _show(axs[2, 0], so[1], (sp[1], sp[0]), sm[1], sb[1], "output, axial")
        _show(axs[2, 1], so[2], (sp[2], sp[0]), sm[2], sb[2], "output, coronal")

        hits = sorted((v for v in rec.get("attack_views", []) if v.get("out_box")),
                      key=lambda v: (not v["on_face"], -(v["out_score"] or 0)))[:MAX_HITS]
        if not rec.get("attack"):
            axs[2, 2].text(0.5, 0.5, "attack not run", ha="center", va="center", fontsize=7,
                           transform=axs[2, 2].transAxes)
        elif not hits:
            axs[2, 2].text(0.5, 0.5, "no detector fired\nin 63 views x 2 detectors", ha="center",
                           va="center", fontsize=7, transform=axs[2, 2].transAxes)
        import attack

        for j, v in enumerate(hits):
            base = vin.render(yaw_deg=v["yaw"], pitch_deg=v["pitch"], size=attack.SIZE)
            r = vout.render(yaw_deg=v["yaw"], pitch_deg=v["pitch"], size=attack.SIZE,
                            threshold=base["threshold"] * v["thr_mult"])
            ax = axs[2, 2 + j]
            ax.imshow(r["rgb"])
            cx, cy, w = v["out_box"]
            ax.add_patch(patches.Rectangle((cx - w / 2, cy - w / 2), w, w, fill=False, lw=1.2,
                                           ec="#ef4444" if v["on_face"] else "#9ca3af"))
            ax.set_title(f"{v['det']} {v['out_score']:.2f} {'ON FACE' if v['on_face'] else 'off face'}\n"
                         f"yaw {v['yaw']:+.0f} pitch {v['pitch']:+.0f} thr x{v['thr_mult']}", fontsize=6)

    rep = rec.get("report", {})
    br = rec.get("brain", {})
    bits = [label, rec["id"], rec.get("status", "")]
    if ok:
        if "verify_rounds" in rep:
            bits += [f"verify rounds {rep['verify_rounds']}",
                     "ESCALATED to fill" if rep.get("escalated_to_block") else "not escalated"]
        elif rep.get("tool"):
            bits.append(f"baseline {rep['tool']} (brain mask: {rep.get('brain_mask')})")
        if br.get("min_dist_mm") is not None:
            bits.append(f"brain: {br['brain_overlap_voxels']} vox touched, closest {br['min_dist_mm']} mm")
        elif (rep.get("brain") or {}).get("margin_mm") is not None:
            bits.append(f"brain (SynthStrip): {rep['brain']['voxels_modified']} vox, margin {rep['brain']['margin_mm']:.1f} mm")
        at = rec.get("attack") or {}
        if at:
            bits.append("on-face hits " + ", ".join(f"{k} {v['output_residual_views']}/{v['views']}" for k, v in at.items()))
    else:
        msg = rec.get("stderr") or rec.get("error") or ""
        reason = msg.rsplit(".nii.gz: ", 1)[-1] if ".nii.gz: " in msg else msg
        bits.append(reason)
    fig.suptitle("\n".join(textwrap.wrap(" · ".join(b for b in bits if b), 140)), fontsize=7.5, x=0.01, ha="left")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.5 / (2.15 * rows + 0.5),
                        wspace=0.04, hspace=0.25)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110, pil_kwargs={"quality": 82})
    plt.close(fig)
