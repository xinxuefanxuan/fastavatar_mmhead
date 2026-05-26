#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from motion_model.models import TemporalConvVAE


class TextToLatentMLP(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, num_layers - 1)):
            layers.extend([nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)])
            d = hidden_dim
        layers.append(nn.Linear(d, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def print_stats(motion_raw: np.ndarray) -> None:
    expr = motion_raw[:, :50]
    head = motion_raw[:, 50:53]
    jaw = motion_raw[:, 53:56]
    for n, arr in [("expr", expr), ("head", head), ("jaw", jaw)]:
        norms = np.linalg.norm(arr, axis=1)
        print(f"[{n}] norm mean={norms.mean():.6f} max={norms.max():.6f}")
    yaw = head[:, 1]
    print(f"[head_yaw] min={yaw.min():.6f} max={yaw.max():.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--target_len", type=int, default=64)
    ap.add_argument("--encoder_name", type=str, default="/home/yuanyuhao/models/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise SystemExit(f"sentence-transformers is required: {exc}")

    text_encoder = SentenceTransformer(args.encoder_name, device=args.device)
    emb = text_encoder.encode([args.prompt], convert_to_numpy=True, normalize_embeddings=False)
    emb = np.asarray(emb, dtype=np.float32)[0]

    tckpt = torch.load(args.checkpoint, map_location=args.device)
    emb_dim = int(tckpt.get("embedding_dim", emb.shape[0]))
    latent_dim = int(tckpt.get("latent_dim", 64))
    hidden_dim = int(tckpt.get("args", {}).get("hidden_dim", 512))
    num_layers = int(tckpt.get("args", {}).get("num_layers", 3))
    model = TextToLatentMLP(emb_dim, latent_dim, hidden_dim, num_layers).to(args.device)
    model.load_state_dict(tckpt["model"])
    model.eval()

    vckpt = torch.load(args.vae_checkpoint, map_location=args.device)
    vae_latent = int(vckpt.get("args", {}).get("latent_dim", latent_dim))
    vae = TemporalConvVAE(in_dim=56, latent_dim=vae_latent).to(args.device)
    vae.load_state_dict(vckpt["model"])
    vae.eval()

    x = torch.from_numpy(emb[None, ...]).to(args.device)
    with torch.no_grad():
        z_pred = model(x)  # [1,latent]
        z_seq = z_pred[:, :, None].repeat(1, 1, args.target_len)
        pred_norm = vae.decode(z_seq)[0].cpu().numpy().astype(np.float32)

    st = json.loads(args.norm_stats.read_text(encoding="utf-8"))
    mean = np.asarray(st["mean"], dtype=np.float32)
    std = np.asarray(st["std"], dtype=np.float32)
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
    print(f"saved text-latent npz: {args.output_npz}")


if __name__ == "__main__":
    main()
