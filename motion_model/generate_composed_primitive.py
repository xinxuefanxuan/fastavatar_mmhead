#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_model.models import TemporalConvVAE


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_weights(s: str | None, n: int) -> list[float]:
    if s is None or s.strip() == "":
        return [1.0] * n
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != n:
        raise SystemExit(f"weights count mismatch: got {len(vals)} expected {n}")
    return vals


def print_stats(motion_raw: np.ndarray) -> None:
    expr = motion_raw[:, :50]
    head = motion_raw[:, 50:53]
    jaw = motion_raw[:, 53:56]
    for name, arr in [("expr", expr), ("head", head), ("jaw", jaw)]:
        norms = np.linalg.norm(arr, axis=1)
        print(f"[{name}] norm mean={norms.mean():.6f} max={norms.max():.6f}")
    yaw = head[:, 1]
    print(f"[head_yaw] min={yaw.min():.6f} max={yaw.max():.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primitives", type=str, required=True, help="comma-separated primitives, e.g. turn_left,smile")
    ap.add_argument("--weights", type=str, default=None, help="comma-separated weights; default all 1.0")
    ap.add_argument("--prototype_path", type=Path, required=True)
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--target_len", type=int, default=64)
    ap.add_argument("--latent_scale", type=float, default=1.0)
    ap.add_argument("--noise_scale", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    primitives = parse_csv(args.primitives)
    if not primitives:
        raise SystemExit("--primitives must not be empty")
    weights = parse_weights(args.weights, len(primitives))

    prot = torch.load(args.prototype_path, map_location="cpu")
    label_map = prot["label_map"]
    if "neutral" not in label_map:
        raise SystemExit("neutral prototype is required in prototype file")

    for p in primitives:
        if p not in label_map:
            raise SystemExit(f"unknown primitive={p}, valid={list(label_map.keys())}")

    z_neutral = np.asarray(prot["label_to_mean_mu"]["neutral"], dtype=np.float32)
    z_comp = z_neutral.copy()
    for p, w in zip(primitives, weights):
        z_p = np.asarray(prot["label_to_mean_mu"][p], dtype=np.float32)
        z_comp = z_comp + float(w) * (z_p - z_neutral)

    z_comp = z_comp * float(args.latent_scale)
    std_ref = np.asarray(prot["label_to_std_mu"].get("neutral", np.ones_like(z_comp)), dtype=np.float32)
    if args.noise_scale > 0:
        eps = np.random.randn(*z_comp.shape).astype(np.float32)
        z_comp = z_comp + float(args.noise_scale) * std_ref * eps

    ckpt = torch.load(args.vae_checkpoint, map_location=args.device)
    latent_dim = int(ckpt.get("args", {}).get("latent_dim", 64))
    vae = TemporalConvVAE(in_dim=56, latent_dim=latent_dim).to(args.device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    z = torch.from_numpy(z_comp[None, None, :]).to(args.device).repeat(1, args.target_len, 1)
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

    print(f"primitives={primitives}")
    print(f"weights={weights}")
    print_stats(motion_raw)
    print(f"saved composed primitive npz: {args.output_npz}")


if __name__ == "__main__":
    main()
