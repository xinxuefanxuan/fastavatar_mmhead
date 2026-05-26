#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_model.models import TemporalConvVAE


def print_stats(motion_raw: np.ndarray) -> None:
    expr = motion_raw[:, :50]
    head = motion_raw[:, 50:53]
    jaw = motion_raw[:, 53:56]
    for name, arr in [("expr", expr), ("head", head), ("jaw", jaw)]:
        norms = np.linalg.norm(arr, axis=1)
        print(f"[{name}] norm mean={norms.mean():.6f} max={norms.max():.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primitive", type=str, required=True)
    ap.add_argument("--prototype_path", type=Path, required=True)
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--target_len", type=int, default=64)
    ap.add_argument("--latent_scale", type=float, default=1.0)
    ap.add_argument("--noise_scale", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    prot = torch.load(args.prototype_path, map_location="cpu")
    label_map = prot["label_map"]
    if args.primitive not in label_map:
        raise SystemExit(f"unknown primitive={args.primitive}, valid={list(label_map.keys())}")

    mean_mu = np.asarray(prot["label_to_mean_mu"][args.primitive], dtype=np.float32)
    std_mu = np.asarray(prot["label_to_std_mu"][args.primitive], dtype=np.float32)

    z_vec = mean_mu * float(args.latent_scale)
    if args.noise_scale > 0:
        eps = np.random.randn(*z_vec.shape).astype(np.float32)
        z_vec = z_vec + float(args.noise_scale) * std_mu * eps

    ckpt = torch.load(args.vae_checkpoint, map_location=args.device)
    latent_dim = int(ckpt.get("args", {}).get("latent_dim", 64))
    vae = TemporalConvVAE(in_dim=56, latent_dim=latent_dim).to(args.device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()

    z = torch.from_numpy(z_vec[None, None, :]).to(args.device).repeat(1, args.target_len, 1)
    with torch.no_grad():
        pred_norm = vae.decode(z.transpose(1, 2))[0].cpu().numpy().astype(np.float32)

    stats = json.loads(args.norm_stats.read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    motion_raw = pred_norm * std[None, :] + mean[None, :]

    out = {
        "motion": motion_raw.astype(np.float32),
        "motion_raw": motion_raw.astype(np.float32),
        "motion_norm": pred_norm.astype(np.float32),
        "expr_delta": motion_raw[:, 0:50].astype(np.float32),
        "head_delta": motion_raw[:, 50:53].astype(np.float32),
        "jaw_delta": motion_raw[:, 53:56].astype(np.float32),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **out)
    print_stats(motion_raw)
    print(f"saved prototype-generated npz: {args.output_npz}")


if __name__ == "__main__":
    main()
