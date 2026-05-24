#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_model.models import TemporalConvVAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    latent_dim = int(ckpt.get("args", {}).get("latent_dim", 64))
    model = TemporalConvVAE(in_dim=56, latent_dim=latent_dim).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    z = torch.randn((1, args.num_frames, latent_dim), device=args.device)
    with torch.no_grad():
        pred_norm = model.decode(z.transpose(1, 2))[0].cpu().numpy().astype(np.float32)

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
    print(f"saved sampled motion npz: {args.output_npz}")


if __name__ == "__main__":
    main()
