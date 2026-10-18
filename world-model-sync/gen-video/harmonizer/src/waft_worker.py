# SPDX-License-Identifier: Apache-2.0
"""Frozen WAFT-a1 estimator, executed with WAFT's Python/PyTorch environment."""

from __future__ import annotations

import json
import sys
import traceback
from argparse import Namespace
from pathlib import Path

import click
import numpy as np
import torch

from waft_protocol import receive_message, send_message


def load_model(repo, config, checkpoint, device, iterations):
    # WAFT's utils is a namespace package; harmonizer's regular utils package
    # would otherwise take precedence even with the WAFT directory first.
    worker_dir = Path(__file__).resolve().parent
    sys.path[:] = [str(repo), *[entry for entry in sys.path if Path(entry or ".").resolve() != worker_dir]]
    # Construct only a1: no a2 backbone or independent pretrained weights needed.
    from model.waft_a1 import ViTWarpV8

    if device.type == "cpu":
        # xFormers' memory-efficient kernels require CUDA. The bundled DINOv2
        # already has a mathematically equivalent dense-attention fallback.
        from thirdparty.DepthAnythingV2.depth_anything_v2.dinov2_layers import attention

        attention.XFORMERS_AVAILABLE = False
    args = Namespace(**json.loads(config.read_text()))
    if args.algorithm != "waft-a1":
        raise ValueError("This adapter supports WAFT a1 configurations; select config/a1/tar-c-t.json")
    args.load_backbone_weights = False
    if iterations is not None:
        args.iters = iterations
    if args.iters < 1:
        raise ValueError("WAFT iterations must be positive")
    model = ViTWarpV8(args)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Strict loading guarantees disabled backbone initialization is fully
    # replaced by the supplied checkpoint, rather than leaving random weights.
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


@click.command()
@click.option("--repo", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--config", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--checkpoint", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--device", default="cuda:0")
@click.option("--iterations", type=click.IntRange(min=1), default=None)
@click.option("--check", is_flag=True, help="Load the real checkpoint and estimate flow on a small image pair.")
def main(repo, config, checkpoint, device, iterations, check):
    # Keep upstream progress/logging off the binary response channel.
    output = sys.stdout.buffer
    sys.stdout = sys.stderr
    torch.set_num_threads(4)
    try:
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model = load_model(repo.resolve(), config.resolve(), checkpoint.resolve(), device, iterations)
        if check:
            with torch.inference_mode():
                image = torch.full((1, 3, 64, 96), 127.5, device=device)
                flow = model(image, image)["flow"][-1]
            assert flow.shape == (1, 2, 64, 96) and torch.isfinite(flow).all()
            print(f"WAFT checkpoint loaded strictly; flow={tuple(flow.shape)} device={device}", file=sys.stderr)
            return
        send_message(output, ready=np.asarray(True), torch_version=np.asarray(torch.__version__))
        while True:
            try:
                request = receive_message(sys.stdin.buffer)
            except EOFError:
                return
            if "close" in request:
                return
            image1 = torch.from_numpy(request["image1"]).to(device=device, dtype=torch.float32)
            image2 = torch.from_numpy(request["image2"]).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                flow = model(image1, image2)["flow"][-1]
            send_message(output, flow=flow.cpu().numpy())
    except Exception:
        if check:
            raise
        send_message(output, error=np.asarray(traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
