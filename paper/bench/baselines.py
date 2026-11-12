# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Baseline defacers, run in pinned Docker images so that no FSL/FreeSurfer
install is needed (scripts/fetch_containers.sh pulls and builds them and
records the digests in results/manifests/containers.json).

  pydeface    Gulban et al., FSL FLIRT registration of a template face mask
              (poldracklab/pydeface image, pydeface 2.1.1.dev0 + FSL flirt).
  mri_deface  Bischoff-Grethe et al. 2007, atlas-based (GCA) face removal;
              the 2007 Linux binary and its two atlases, built into a pinned
              Ubuntu image (bench/docker/mri_deface/Dockerfile).
  quickshear  Schimke & Hale 2011 (pip quickshear 1.2.0), shearing plane
              under the convex hull of a brain mask. The mask comes from the
              *original* SynthStrip (freesurfer/synthstrip:1.6), not from the
              dataset's expert mask (brain safety is scored against that, so
              using it would be circular) and not from RadVeil's own
              reimplementation (the software under test). The tool's buffer is
              in voxels along the superior axis; we fix it at 10 mm and convert
              with that axis's voxel size, so thick-slice scans get the same
              physical margin as 1 mm scans. Quickshear 1.2.0 pairs plane
              heights with columns via zip(np.nonzero(ys > 0)[0], ys) and can
              mis-cut or raise IndexError on unusual masks; such exceptions are
              recorded by type as tool failures.

`run()` writes an output on the input's grid or reports a refusal; the
caller derives the region as the voxels whose stored value changed.
All images are x86-64; on Apple silicon they run under emulation, so their
wall times are reported but not compared with RadVeil's native timings.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

IMAGES = {
    # name -> (image reference pinned by digest, platform)
    "pydeface": "poldracklab/pydeface@sha256:c648b8d70f31a186ff003ed2ad15d491d7b4a7d9adaa4665cf95526415fe3773",
    "synthstrip": "freesurfer/synthstrip@sha256:16181aeb75cab4fa0bf7ab4f79ba77eca466cc0e28a384f3ecb007c4a376ea09",
    # built locally from bench/docker/mri_deface (digest recorded by fetch_containers.sh)
    "mri_deface": "radveil-paper/mri_deface:2007",
}
PLATFORM = "linux/amd64"

TOOLS = {
    "pydeface": "pydeface 2.1.1.dev0 (FSL FLIRT, template face mask)",
    "mri_deface": "mri_deface 2007 Linux binary (GCA atlases)",
    "quickshear": "Quickshear 1.2.0, 10 mm buffer, original-SynthStrip brain mask",
}
QUICKSHEAR_BUFFER_MM = 10.0

# Docker's VM has ~7.6 GB by default and the x86 images run emulated on
# Apple silicon. Measured peak memory: SynthStrip 6.9 GB (so it runs alone),
# pydeface and mri_deface ~1-2 GB. Containers share a machine-wide budget
# (GB) instead of fixed slots; override with RADVEIL_DOCKER_BUDGET_GB.
WEIGHT_GB = {"synthstrip": 7.0, "pydeface": 2.0, "mri_deface": 2.0}
BUDGET_GB = float(os.environ.get("RADVEIL_DOCKER_BUDGET_GB", 7.0))
LOCKS = Path(os.environ.get("TMPDIR", "/tmp")) / "radveil-paper-docker"


class _budget:
    """Reserve WEIGHT_GB[name] of the machine-wide budget until exit. The
    ledger maps pid -> GB; entries of dead processes are dropped."""

    def __init__(self, name: str):
        self.w = min(WEIGHT_GB.get(name, 2.0), BUDGET_GB)

    def _update(self, fn):
        import fcntl
        import json as _json

        LOCKS.mkdir(parents=True, exist_ok=True)
        with open(LOCKS / "ledger.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            p = LOCKS / "ledger.json"
            led = _json.loads(p.read_text()) if p.exists() else {}
            led = {k: v for k, v in led.items() if _alive(int(k))}
            ok = fn(led)
            p.write_text(_json.dumps(led))
            return ok

    def __enter__(self):
        me = str(os.getpid())

        def take(led):
            if sum(led.values()) + self.w <= BUDGET_GB + 1e-9:
                led[me] = led.get(me, 0) + self.w
                return True
            return False

        while not self._update(take):
            time.sleep(1.0)
        return self

    def __exit__(self, *exc):
        me = str(os.getpid())

        def give(led):
            led[me] = led.get(me, 0) - self.w
            if led[me] <= 1e-9:
                led.pop(me)
            return True

        self._update(give)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# persistent cache of original-SynthStrip input masks, keyed by input hash
# and image digest (a different SynthStrip build never reuses a mask)
CACHE = Path(os.environ.get("RADVEIL_SYNTHSTRIP_CACHE", Path.home() / ".cache" / "radveil-paper" / "synthstrip"))


def input_brain_mask(src: Path) -> tuple[Path, dict]:
    """Original-SynthStrip mask of an *unmodified* input, cached by the
    file's SHA-256 so Quickshear, the downstream comparison and the Rust
    validation run the (slow, emulated) container once per input."""
    import hashlib

    h = hashlib.sha256(Path(src).read_bytes()).hexdigest()
    m = CACHE / IMAGES["synthstrip"].rsplit(":", 1)[-1][:16] / f"{h}.nii.gz"
    if m.exists():
        return m, {"exit": 0, "wall_s": 0.0, "stderr": "", "cached": True}
    m.parent.mkdir(parents=True, exist_ok=True)
    tmp = m.parent / f".{h}.{os.getpid()}"
    tmp.mkdir(exist_ok=True)
    r = synthstrip_mask(Path(src), tmp / "m.nii.gz")
    if r["exit"] == 0 and (tmp / "m.nii.gz").exists():
        os.replace(tmp / "m.nii.gz", m)
    shutil.rmtree(tmp, ignore_errors=True)
    return m, r


def _docker(image: str, mounts: dict[Path, str], args: list[str], timeout: int = 3600) -> dict:
    """Run a container with the given host->container mounts; return exit,
    wall time and the stderr tail."""
    # run as the calling user so outputs are not root-owned; tools that write
    # scratch files (mri_deface) need a writable HOME and working directory
    cname = f"rvpaper-{os.getpid()}-{time.monotonic_ns()}"
    cmd = ["docker", "run", "--rm", "--name", cname, "--platform", PLATFORM,
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-w", "/tmp"]
    for host, ctr in mounts.items():
        cmd += ["-v", f"{Path(host).resolve()}:{ctr}"]
    cmd += [image, *args]
    name = next((k for k, v in IMAGES.items() if v == image), "other")
    with _budget(name):
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            # the client timed out; stop the container too, inside the budget
            subprocess.run(["docker", "kill", cname], capture_output=True)
            return {"exit": 124, "wall_s": round(time.perf_counter() - t0, 3), "peak_rss_mb": None,
                    "stderr": f"docker: timed out after {timeout} s", "infra_error": True}
    tail = [ln for ln in (r.stderr + r.stdout).strip().splitlines() if ln.strip()]
    # 137 = killed (out of memory in Docker's VM), 125-127 = daemon/image/
    # command problems: our infrastructure failing, never the tool refusing
    infra = r.returncode in (125, 126, 127, 137) or (r.stderr or "").lstrip().startswith("docker:")
    return {"exit": r.returncode, "wall_s": round(time.perf_counter() - t0, 3), "peak_rss_mb": None,
            "stderr": "\n".join(tail[-6:])[-1000:], "oom_killed": r.returncode == 137, "infra_error": infra}


def synthstrip_mask(src: Path, mask_out: Path) -> dict:
    """Brain mask from the original SynthStrip container."""
    src, mask_out = Path(src).resolve(), Path(mask_out).resolve()
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    if src.parent == mask_out.parent:
        return _docker(IMAGES["synthstrip"], {src.parent: "/io"},
                       ["-i", f"/io/{src.name}", "-m", f"/io/{mask_out.name}"])
    # stage the input next to the mask: one local mount, so the datasets may
    # live on a network share that Docker Desktop cannot bind-mount reliably
    staged = mask_out.parent / f".in.{os.getpid()}.{src.name}"
    shutil.copyfile(src, staged)
    try:
        return _docker(IMAGES["synthstrip"], {mask_out.parent: "/io"},
                       ["-i", f"/io/{staged.name}", "-m", f"/io/{mask_out.name}"])
    finally:
        staged.unlink(missing_ok=True)


def _quickshear(src: Path, out: Path, tmp: Path) -> dict:
    import nibabel as nib
    import quickshear

    mask_p, r = input_brain_mask(src)
    if r["exit"] != 0 or not mask_p.exists():
        # no brain mask means the pipeline around Quickshear failed, not Quickshear
        return {**r, "stderr": "synthstrip: " + r["stderr"], "infra_error": True}
    t0 = time.perf_counter()
    anat, mask = nib.load(str(src)), nib.load(str(mask_p))
    # quickshear's buffer runs along the superior axis of its RPS frame
    ornt = nib.io_orientation(mask.affine)
    s_axis = int([i for i in range(3) if ornt[i, 0] == 2][0])
    buff = QUICKSHEAR_BUFFER_MM / float(mask.header.get_zooms()[s_axis])
    nib.save(quickshear.quickshear(anat, mask, buff=buff), str(out))
    return {"exit": 0, "wall_s": round(r["wall_s"] + time.perf_counter() - t0, 3), "peak_rss_mb": None,
            "stderr": "", "mask_s": r["wall_s"], "buffer_vox": round(buff, 2)}


def run(tool: str, src: Path, out: Path, tmp: Path, env: dict | None = None) -> dict:
    """Run baseline `tool` on `src`, writing `out` (.nii.gz on the input grid).

    Returns {"exit", "wall_s", "peak_rss_mb" (None: inside Docker), "stderr",
    "tool_version", "refused"}; "refused" is True when no output was produced.
    `env` is accepted for symmetry with the RadVeil runner and unused (the
    containers run single-process with their own defaults)."""
    src, out, tmp = Path(src), Path(out), Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tool in ("pydeface", "mri_deface"):  # mount a local copy (see synthstrip_mask)
        staged = tmp / "in"
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir()
        shutil.copyfile(src, staged / src.name)
    if tool == "pydeface":
        r = _docker(IMAGES["pydeface"], {staged: "/in", out.parent: "/out"},
                    [f"/in/{src.name}", "--outfile", f"/out/{out.name}", "--force"])
    elif tool == "mri_deface":
        r = _docker(IMAGES["mri_deface"], {staged: "/in", out.parent: "/out"},
                    [f"/in/{src.name}", "/opt/mri_deface/talairach_mixed_with_skull.gca",
                     "/opt/mri_deface/face.gca", f"/out/{out.name}"])
    elif tool == "quickshear":
        try:
            r = _quickshear(src, out, tmp)
        except Exception as e:  # upstream raises on degenerate masks (see the pairing bug noted above)
            r = {"exit": 1, "wall_s": None, "peak_rss_mb": None, "stderr": f"{type(e).__name__}: {e}"[:400],
                 "exception": type(e).__name__}
    else:
        raise ValueError(f"unknown baseline {tool!r} (one of {', '.join(TOOLS)})")
    r["tool_version"] = TOOLS[tool]
    r["refused"] = r["exit"] != 0 or not out.exists()
    return r


def same_grid(src: Path, out: Path) -> dict:
    """Whether a baseline kept the input grid, datatype and scaling (mri_deface
    rewrites the header through FreeSurfer's I/O)."""
    import nibabel as nib
    import numpy as np

    a, b = nib.load(str(src)), nib.load(str(out))
    return {"shape": a.shape == b.shape, "affine": bool(np.allclose(a.affine, b.affine, atol=1e-4)),
            "dtype": str(a.get_data_dtype()) == str(b.get_data_dtype())}


if __name__ == "__main__":
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description="run one baseline on one volume")
    ap.add_argument("tool", choices=sorted(TOOLS))
    ap.add_argument("input")
    ap.add_argument("output")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as t:
        r = run(a.tool, Path(a.input), Path(a.output), Path(t))
    if not r["refused"]:
        r["grid"] = same_grid(Path(a.input), Path(a.output))
    print(json.dumps(r, indent=2))
