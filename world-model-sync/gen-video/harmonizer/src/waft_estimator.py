# SPDX-License-Identifier: Apache-2.0
"""WAFT optical-flow backend isolated from harmonizer's PyTorch dependencies."""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch

from waft_protocol import receive_message, send_message

DEFAULT_WAFT_REPO = Path(__file__).resolve().parents[3] / "WAFT"


def waft_paths(repo=None, config=None, checkpoint=None):
    repo = Path(repo or DEFAULT_WAFT_REPO).resolve()
    config = Path(config).resolve() if config else repo / "config/a1/tar-c-t.json"
    checkpoint = Path(checkpoint).resolve() if checkpoint else repo.parent / "tar-c-t.pth"
    for path in (repo / "pixi.toml", config, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Missing WAFT file: {path}")
    return repo, config, checkpoint


class WAFTOpticalFlow:
    """One persistent, frozen WAFT worker per training rank, on that rank's GPU."""

    def __init__(self, device, *, repo=None, config=None, checkpoint=None, iterations=None, timeout=600):
        repo, config, checkpoint = waft_paths(repo, config, checkpoint)
        pixi = os.environ.get("PIXI_EXECUTABLE") or os.environ.get("PIXI_EXE") or shutil.which("pixi")
        if pixi is None:
            raise RuntimeError("WAFT requires pixi on PATH, or PIXI_EXECUTABLE set to its absolute path")
        self.timeout = timeout
        self.lock = threading.Lock()
        self.process = None
        # Remove the parent Python/CUDA environment's paths. Keep GPU visibility,
        # scheduler settings and normal user environment for rank device mapping.
        env = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH"):
            env.pop(name, None)
        command = [
            pixi,
            "run",
            "--manifest-path",
            str(repo / "pixi.toml"),
            "--locked",
            "--",
            "python",
            "-u",
            str(Path(__file__).with_name("waft_worker.py")),
            "--repo",
            str(repo),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--device",
            str(torch.device(device)),
        ]
        if iterations is not None:
            command.extend(["--iterations", str(iterations)])
        self.process = subprocess.Popen(
            command, cwd=repo, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        try:
            ready = self._receive()
            if not bool(ready.get("ready", False)):
                raise RuntimeError("WAFT worker did not report ready")
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    def _receive(self):
        message = receive_message(self.process.stdout, timeout=self.timeout)
        if "error" in message:
            raise RuntimeError(f"WAFT estimation failed:\n{message['error'].item()}")
        return message

    @torch.no_grad()
    def backward(self, curr, prev):
        """Estimate curr->prev flow in pixels, accepting harmonizer RGB [-1,1]."""
        if curr.shape != prev.shape or curr.ndim != 4 or curr.shape[1] != 3:
            raise ValueError("WAFT expects matching RGB [B,3,H,W] image pairs")
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("WAFT worker is no longer running")
            images = [((image.detach().float().cpu() + 1.0) * 127.5).clamp(0, 255).numpy() for image in (curr, prev)]
            try:
                send_message(self.process.stdin, image1=images[0], image2=images[1])
                result = self._receive()
            except Exception:
                self.close()
                raise
            flow = result["flow"]
            if flow.shape != (curr.shape[0], 2, *curr.shape[-2:]) or not np.isfinite(flow).all():
                raise RuntimeError("WAFT returned invalid flow dimensions or non-finite values")
            return torch.from_numpy(flow).to(device=curr.device, dtype=torch.float32)

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.stdout:
            process.stdout.close()
        atexit.unregister(self.close)
