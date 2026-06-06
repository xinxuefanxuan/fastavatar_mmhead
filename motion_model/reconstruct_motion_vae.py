#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_model.models import TemporalConvVAE


def channel_stats(name: str, arr: np.ndarray) -> None:
    norms = np.linalg.norm(arr, axis=1)
    print(f"[{name}] norm mean={norms.mean():.6f} max={norms.max():.6f}")


def axis_stats(name: str, arr: np.ndarray) -> None:
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    print(
        f"[{name}] axis min=({mins[0]:.6f}, {mins[1]:.6f}, {mins[2]:.6f}) "
        f"max=({maxs[0]:.6f}, {maxs[1]:.6f}, {maxs[2]:.6f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--motion_key", type=str, default="motion_norm")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_mu", action="store_true", default=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    latent_dim = int(ckpt.get("args", {}).get("latent_dim", 64))
    model = TemporalConvVAE(in_dim=56, latent_dim=latent_dim).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = np.load(args.input_npz, allow_pickle=True)
    if args.motion_key not in data:
        raise SystemExit(f"missing key {args.motion_key} in {args.input_npz}")
    motion_norm = np.asarray(data[args.motion_key], dtype=np.float32)

    x = torch.from_numpy(motion_norm[None, ...]).to(args.device)
    with torch.no_grad():
        mu, logvar = model.encode(x)
        z = mu if args.use_mu else model.reparameterize(mu, logvar)
        pred_norm = model.decode(z)[0].cpu().numpy().astype(np.float32)

    stats = json.loads(args.norm_stats.read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    motion_raw = pred_norm * std[None, :] + mean[None, :]

    orig_raw = motion_norm * std[None, :] + mean[None, :]
    orig_expr, orig_head, orig_jaw = orig_raw[:, 0:50], orig_raw[:, 50:53], orig_raw[:, 53:56]
    rec_expr, rec_head, rec_jaw = motion_raw[:, 0:50], motion_raw[:, 50:53], motion_raw[:, 53:56]

    print("[Original stats]")
    channel_stats("expr", orig_expr)
    channel_stats("head", orig_head)
    channel_stats("jaw", orig_jaw)
    axis_stats("head", orig_head)
    axis_stats("jaw", orig_jaw)

    print("[Reconstructed stats]")
    channel_stats("expr", rec_expr)
    channel_stats("head", rec_head)
    channel_stats("jaw", rec_jaw)
    axis_stats("head", rec_head)
    axis_stats("jaw", rec_jaw)

    out = {
        "motion": motion_raw.astype(np.float32),
        "motion_raw": motion_raw.astype(np.float32),
        "motion_norm": pred_norm.astype(np.float32),
        "expr_delta": rec_expr.astype(np.float32),
        "head_delta": rec_head.astype(np.float32),
        "jaw_delta": rec_jaw.astype(np.float32),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **out)
    print(f"saved VAE reconstruction npz: {args.output_npz}")


if __name__ == "__main__":
    main()
