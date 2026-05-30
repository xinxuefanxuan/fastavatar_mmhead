#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from motion_model.models import TemporalConvVAE


class PrimitiveMLP(nn.Module):
    def __init__(self, num_labels: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_labels, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, latent_dim))

    def forward(self, label_idx: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embedding(label_idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primitive_checkpoint", type=Path, required=True)
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, required=True)
    ap.add_argument("--primitive_label", type=str, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pckpt = torch.load(args.primitive_checkpoint, map_location=args.device)
    label_map = pckpt["label_map"]
    if args.primitive_label not in label_map:
        raise SystemExit(f"unknown primitive_label={args.primitive_label}, valid={list(label_map.keys())}")

    vckpt = torch.load(args.vae_checkpoint, map_location=args.device)
    latent_dim = int(vckpt.get("args", {}).get("latent_dim", 64))

    vae = TemporalConvVAE(in_dim=56, latent_dim=latent_dim).to(args.device)
    vae.load_state_dict(vckpt["model"])
    vae.eval()

    predictor = PrimitiveMLP(num_labels=len(label_map), latent_dim=latent_dim).to(args.device)
    predictor.load_state_dict(pckpt["model"])
    predictor.eval()

    label_idx = torch.tensor([label_map[args.primitive_label]], dtype=torch.long, device=args.device)
    with torch.no_grad():
        z_vec = predictor(label_idx)[0]  # [latent_dim]
        z = z_vec[None, None, :].repeat(1, args.num_frames, 1)
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
    print(f"saved primitive-generated npz: {args.output_npz}")


if __name__ == "__main__":
    main()
