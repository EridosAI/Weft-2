"""V2 Phase 1 — concurrent arm-run dispatcher (§7.7, instr §9).

Runs arm-runs as independent subprocesses (`python -m v2.src.phase1.arm_runner`),
up to `n_concurrent` at a time on the single GPU. Includes a VRAM monitor and a
`measure_concurrency` helper used by the §7.1 smoke to validate 2x/3x parallelism
(per-arm wall-clock overhead + VRAM peak) before locking the Phase 1 parallelism.

No auto-fallback (§9.3): callers decide on OOM / wall-clock degradation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def vram_used_total_mb():
    """(used_mb, total_mb) for GPU 0, or (None, None) if nvidia-smi is unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True)
        used, total = out.strip().splitlines()[0].split(",")
        return int(used), int(total)
    except Exception:  # noqa: BLE001
        return None, None


class VramMonitor:
    """Background poller recording peak VRAM during a `with` block."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.peak_mb = 0
        self.total_mb = None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            used, total = vram_used_total_mb()
            if used is not None:
                self.peak_mb = max(self.peak_mb, used)
                self.total_mb = total

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _launch(spec: dict, repo_root: str):
    """Spawn one arm_runner subprocess for `spec`; return (Popen, spec_file_path)."""
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(spec, f)
    f.close()
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root
    p = subprocess.Popen(
        [sys.executable, "-m", "v2.src.phase1.arm_runner", "--spec-file", f.name],
        cwd=repo_root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p, f.name


def run_batch(specs: list[dict], n_concurrent: int, repo_root: str) -> list[dict]:
    """Run `specs` with <= n_concurrent subprocesses; return result dicts (order = completion)."""
    results, running, queue = [], [], list(specs)
    while queue or running:
        while queue and len(running) < n_concurrent:
            spec = queue.pop(0)
            p, fn = _launch(spec, repo_root)
            running.append((p, fn, spec))
        time.sleep(1.0)
        still = []
        for p, fn, spec in running:
            if p.poll() is None:
                still.append((p, fn, spec))
            else:
                out, _ = p.communicate()
                of = Path(spec["out_file"])
                res = json.loads(of.read_text()) if of.exists() else {"error": out, "spec": spec}
                results.append(res)
                try:
                    os.unlink(fn)
                except OSError:
                    pass
        running = still
    return results


def measure_concurrency(spec_template: dict, n: int, repo_root: str, out_dir: str) -> dict:
    """Launch n identical (seed-offset) arm-runs concurrently; return wall-clock + VRAM peak.

    Used by the §7.1 smoke: compare n=2 / n=3 wall-clock to the n=1 baseline and the
    VRAM peak (+50% margin) against total VRAM.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    specs = []
    for i in range(n):
        s = json.loads(json.dumps(spec_template))  # deep copy
        s["seed"] = int(spec_template["seed"]) + i
        s["label"] = f"concurrency_n{n}_arm{i}"
        s["out_file"] = str(Path(out_dir) / f"concurrency_n{n}_arm{i}.json")
        specs.append(s)

    t0 = time.time()
    with VramMonitor(interval=1.5) as vm:
        procs = [_launch(s, repo_root) for s in specs]
        oom = False
        for (p, fn), s in zip(procs, specs):
            out, _ = p.communicate()
            if p.returncode != 0 or "out of memory" in (out or "").lower() \
                    or "CUDA out of memory" in (out or ""):
                oom = oom or ("out of memory" in (out or "").lower())
            try:
                os.unlink(fn)
            except OSError:
                pass
    wall = time.time() - t0
    return {"n": n, "wall_s": round(wall, 1),
            "vram_peak_mb": vm.peak_mb, "vram_total_mb": vm.total_mb,
            "oom": oom}
