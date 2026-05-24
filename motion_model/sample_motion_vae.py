#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_model.models import TemporalConvVAE


def smooth_centered(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xpad = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = xpad[i : i + window].mean(axis=0)
    return out


def clamp_velocity(x: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0 or x.shape[0] <= 1:
        return x
    out = x.copy()
    for i in range(1, out.shape[0]):
        step = out[i] - out[i - 1]
        n = float(np.linalg.norm(step))
        if n > max_step and n > 1e-12:
            out[i] = out[i - 1] + step * (max_step / n)
    return out


def print_stats(sample_idx: int, expr: np.ndarray, head: np.ndarray, jaw: np.ndarray) -> None:
    def norm_mean_max(arr: np.ndarray) -> tuple[float, float]:
        norms = np.linalg.norm(arr, axis=1)
        return float(norms.mean()), float(norms.max())

    expr_mean, expr_max = norm_mean_max(expr)
    head_mean, head_max = norm_mean_max(head)
    jaw_mean, jaw_max = norm_mean_max(jaw)
    head_min, head_max_axis = head.min(axis=0), head.max(axis=0)
    vel = head[1:] - head[:-1] if head.shape[0] > 1 else np.zeros((0, head.shape[1]), dtype=np.float32)
    vel_norm = np.linalg.norm(vel, axis=1) if vel.shape[0] > 0 else np.zeros((1,), dtype=np.float32)
    vel_mean = float(vel_norm.mean())
    vel_max = float(vel_norm.max())
    print(f"[Sample {sample_idx}] expr norm mean={expr_mean:.6f} max={expr_max:.6f}")
    print(f"[Sample {sample_idx}] head norm mean={head_mean:.6f} max={head_max:.6f}")
    print(f"[Sample {sample_idx}] jaw  norm mean={jaw_mean:.6f} max={jaw_max:.6f}")
    print(
        f"[Sample {sample_idx}] head axis min=({head_min[0]:.6f}, {head_min[1]:.6f}, {head_min[2]:.6f}) "
        f"max=({head_max_axis[0]:.6f}, {head_max_axis[1]:.6f}, {head_max_axis[2]:.6f})"
    )
    print(f"[Sample {sample_idx}] head velocity norm mean={vel_mean:.6f} max={vel_max:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--z_scale", type=float, default=0.5)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--smooth_window", type=int, default=5)
    ap.add_argument("--head_scale", type=float, default=1.0)
    ap.add_argument("--head_velocity_clamp", type=float, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    latent_dim = int(ckpt.get("args", {}).get("latent_dim", 64))
    model = TemporalConvVAE(in_dim=56, latent_dim=latent_dim).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = json.loads(args.norm_stats.read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_samples):
        z = torch.randn((1, args.num_frames, latent_dim), device=args.device) * float(args.z_scale)
        with torch.no_grad():
            pred_norm = model.decode(z.transpose(1, 2))[0].cpu().numpy().astype(np.float32)
        motion_raw = pred_norm * std[None, :] + mean[None, :]
        expr_delta = motion_raw[:, 0:50].astype(np.float32)
        head_delta = motion_raw[:, 50:53].astype(np.float32)
        jaw_delta = motion_raw[:, 53:56].astype(np.float32)

        if args.smooth:
            expr_delta = smooth_centered(expr_delta, args.smooth_window)
            head_delta = smooth_centered(head_delta, args.smooth_window)
            jaw_delta = smooth_centered(jaw_delta, args.smooth_window)
        head_delta = head_delta * float(args.head_scale)
        if args.head_velocity_clamp is not None and args.head_velocity_clamp > 0:
            head_delta = clamp_velocity(head_delta, float(args.head_velocity_clamp))

        motion_raw_post = np.concatenate([expr_delta, head_delta, jaw_delta], axis=1).astype(np.float32)
        motion_norm_post = (motion_raw_post - mean[None, :]) / std[None, :]
        out = {
            "motion": motion_raw_post.astype(np.float32),
            "motion_raw": motion_raw_post.astype(np.float32),
            "motion_norm": motion_norm_post.astype(np.float32),
            "expr_delta": expr_delta.astype(np.float32),
            "head_delta": head_delta.astype(np.float32),
            "jaw_delta": jaw_delta.astype(np.float32),
        }

        out_path = args.output_npz if args.num_samples == 1 else args.output_npz.with_name(f"{args.output_npz.stem}_{i:03d}{args.output_npz.suffix}")
        np.savez(out_path, **out)
        print_stats(i, expr_delta, head_delta, jaw_delta)
        print(f"saved sampled motion npz: {out_path}")


if __name__ == "__main__":
    main()
