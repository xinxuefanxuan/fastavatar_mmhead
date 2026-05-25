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


def select_head_index(head: np.ndarray, primitives: list[str]) -> int:
    yaw = head[:, 1]
    if "turn_left" in primitives:
        return int(np.argmax(yaw))
    if "turn_right" in primitives:
        return int(np.argmin(yaw))
    return int(np.argmax(np.linalg.norm(head, axis=1)))


def build_hold_sequence(
    motion_raw: np.ndarray,
    primitives: list[str],
    output_len: int,
    ramp_frames: int,
    hold_frames: int,
    release_frames: int,
    release_ratio: float,
) -> np.ndarray:
    expr = motion_raw[:, :50]
    head = motion_raw[:, 50:53]
    jaw = motion_raw[:, 53:56]

    i_head = select_head_index(head, primitives)
    i_expr = int(np.argmax(np.linalg.norm(expr, axis=1)))
    i_jaw = int(np.argmax(np.linalg.norm(jaw, axis=1)))
    tgt_expr = expr[i_expr]
    tgt_head = head[i_head]
    tgt_jaw = jaw[i_jaw]

    r = max(1, int(ramp_frames))
    h = max(0, int(hold_frames))
    rel = max(0, int(release_frames))
    rel_ratio = float(release_ratio)
    rel_ratio = min(max(rel_ratio, 0.0), 1.0)

    ramp_a = np.linspace(0.0, 1.0, r, dtype=np.float32)
    hold_a = np.ones((h,), dtype=np.float32)
    release_end = 1.0 - rel_ratio
    release_a = np.linspace(1.0, release_end, rel, dtype=np.float32) if rel > 0 else np.zeros((0,), dtype=np.float32)
    alpha = np.concatenate([ramp_a, hold_a, release_a], axis=0)
    if alpha.size == 0:
        alpha = np.zeros((1,), dtype=np.float32)
    if alpha.size < output_len:
        alpha = np.concatenate([alpha, np.full((output_len - alpha.size,), alpha[-1], dtype=np.float32)], axis=0)
    else:
        alpha = alpha[:output_len]

    out_expr = alpha[:, None] * tgt_expr[None, :]
    out_head = alpha[:, None] * tgt_head[None, :]
    out_jaw = alpha[:, None] * tgt_jaw[None, :]
    return np.concatenate([out_expr, out_head, out_jaw], axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primitives", type=str, required=True, help="comma-separated primitives, e.g. turn_left,smile")
    ap.add_argument("--weights", type=str, default=None, help="comma-separated weights; default all 1.0")
    ap.add_argument("--prototype_path", type=Path, required=True)
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--target_len", type=int, default=64)
    ap.add_argument("--temporal_mode", choices=["raw", "hold"], default="raw")
    ap.add_argument("--output_len", type=int, default=None)
    ap.add_argument("--ramp_frames", type=int, default=10)
    ap.add_argument("--hold_frames", type=int, default=18)
    ap.add_argument("--release_frames", type=int, default=4)
    ap.add_argument("--release_ratio", type=float, default=0.75)
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
    print("[Before temporal]")
    print_stats(motion_raw)

    out_len = int(args.output_len) if args.output_len is not None else int(args.target_len)
    if args.temporal_mode == "hold":
        motion_raw = build_hold_sequence(
            motion_raw=motion_raw,
            primitives=primitives,
            output_len=out_len,
            ramp_frames=args.ramp_frames,
            hold_frames=args.hold_frames,
            release_frames=args.release_frames,
            release_ratio=args.release_ratio,
        )
    else:
        motion_raw = motion_raw[:out_len]
        if motion_raw.shape[0] < out_len:
            pad = np.repeat(motion_raw[-1:], out_len - motion_raw.shape[0], axis=0)
            motion_raw = np.concatenate([motion_raw, pad], axis=0)

    pred_norm = (motion_raw - mean[None, :]) / std[None, :]

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
    print(f"temporal_mode={args.temporal_mode}")
    print(f"output_len={out_len}")
    print("[After temporal]")
    print_stats(motion_raw)
    print(f"saved composed primitive npz: {args.output_npz}")


if __name__ == "__main__":
    main()
