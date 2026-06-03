#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_model.channel_vae import ChannelTemporalVAE


NEW_HEAD_PRIMITIVES = ["look_up", "look_down", "tilt_left", "tilt_right"]


def load_cvae(ckpt: Path, device: torch.device):
    c = torch.load(ckpt, map_location=device)
    m = ChannelTemporalVAE(int(c["input_dim"]), int(c["target_len"]), int(c["latent_dim"]), int(c["hidden_dim"]))
    m.load_state_dict(c["model"])
    m.to(device).eval()
    return m


def make_env(target_len: int, ramp: int, hold: int, release: int) -> np.ndarray:
    fixed = ramp + release
    if fixed > target_len:
        release = max(0, target_len - ramp)
    hold = max(0, target_len - ramp - release)
    parts = []
    if ramp > 0:
        parts.append(np.linspace(0.0, 1.0, ramp, endpoint=True, dtype=np.float32))
    if hold > 0:
        parts.append(np.ones((hold,), dtype=np.float32))
    if release > 0:
        parts.append(np.linspace(1.0, 0.0, release, endpoint=True, dtype=np.float32))
    env = np.concatenate(parts, axis=0) if parts else np.zeros((target_len,), dtype=np.float32)
    if env.shape[0] < target_len:
        env = np.concatenate([env, np.zeros((target_len - env.shape[0],), dtype=np.float32)], axis=0)
    return env[:target_len]


def save_teacher_npz(path: Path, motion_raw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        motion=motion_raw.astype(np.float32),
        motion_raw=motion_raw.astype(np.float32),
        expr_delta=motion_raw[:, :50].astype(np.float32),
        head_delta=motion_raw[:, 50:53].astype(np.float32),
        jaw_delta=motion_raw[:, 53:56].astype(np.float32),
    )


def head_stats(raw: np.ndarray) -> dict:
    head = raw[:, 50:53]
    n = np.linalg.norm(head, axis=1)
    return {
        "pitch_min": float(head[:, 0].min()), "pitch_max": float(head[:, 0].max()),
        "yaw_min": float(head[:, 1].min()), "yaw_max": float(head[:, 1].max()),
        "roll_min": float(head[:, 2].min()), "roll_max": float(head[:, 2].max()),
        "head_norm_mean": float(n.mean()), "head_norm_max": float(n.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_channel_prototypes", type=Path, default=Path("outputs/mmhead_debug/channel_prototypes_v2/channel_prototypes.pt"))
    ap.add_argument("--head_checkpoint", type=Path, default=Path("outputs/mmhead_debug/channel_vae_v1/head/best.pt"))
    ap.add_argument("--expr_checkpoint", type=Path, default=Path("outputs/mmhead_debug/channel_vae_v1/expr/best.pt"))
    ap.add_argument("--jaw_checkpoint", type=Path, default=Path("outputs/mmhead_debug/channel_vae_v1/jaw/best.pt"))
    ap.add_argument("--norm_stats", type=Path, default=Path("outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json"))
    ap.add_argument("--output_path", type=Path, default=Path("outputs/mmhead_debug/channel_prototypes_v3_extended/channel_prototypes.pt"))
    ap.add_argument("--target_len", type=int, default=64)
    ap.add_argument("--pitch_mag", type=float, default=0.12)
    ap.add_argument("--roll_mag", type=float, default=0.10)
    ap.add_argument("--ramp_frames", type=int, default=12)
    ap.add_argument("--hold_frames", type=int, default=40)
    ap.add_argument("--release_frames", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    base = torch.load(args.base_channel_prototypes, map_location="cpu")
    proto = base["prototypes"]
    primitive_classes = list(base["primitive_classes"])

    st = json.loads(args.norm_stats.read_text())
    mean = np.asarray(st["mean"], dtype=np.float32)[:56]
    std = np.asarray(st["std"], dtype=np.float32)[:56]

    # load all channel VAEs (expr/jaw only for checkpoint traceability; head used for encoding)
    _expr = load_cvae(args.expr_checkpoint, dev)
    head_vae = load_cvae(args.head_checkpoint, dev)
    _jaw = load_cvae(args.jaw_checkpoint, dev)

    neutral_expr = proto["expr"]["neutral"].clone()
    neutral_jaw = proto["jaw"]["neutral"].clone()

    env = make_env(args.target_len, args.ramp_frames, args.hold_frames, args.release_frames)

    teacher_root = args.output_path.parent / "teacher_npz"

    synth_cfg = {
        "look_down": ("pitch", +args.pitch_mag),
        "look_up": ("pitch", -args.pitch_mag),
        "tilt_left": ("roll", -args.roll_mag),
        "tilt_right": ("roll", +args.roll_mag),
    }

    for name, (axis_name, mag) in synth_cfg.items():
        raw = np.zeros((args.target_len, 56), dtype=np.float32)
        if axis_name == "pitch":
            raw[:, 50] = env * mag
        elif axis_name == "roll":
            raw[:, 52] = env * mag

        save_teacher_npz(teacher_root / f"{name}.npz", raw)

        motion_norm = (raw - mean[None, :]) / np.maximum(std[None, :], 1e-8)
        head_norm = motion_norm[:, 50:53]

        x = torch.from_numpy(head_norm[None].astype(np.float32)).to(dev)
        with torch.no_grad():
            mu, _ = head_vae.encode(x)
        mu = mu[0].cpu()

        proto["head"][name] = mu
        proto["expr"][name] = neutral_expr.clone()
        proto["jaw"][name] = neutral_jaw.clone()

        if name not in primitive_classes:
            primitive_classes.append(name)

        with torch.no_grad():
            dec = head_vae.decode(mu[None].to(dev))[0].cpu().numpy().astype(np.float32)
        rec_raw = np.zeros((args.target_len, 56), dtype=np.float32)
        rec_raw[:, 50:53] = dec * std[None, 50:53] + mean[None, 50:53]
        hs = head_stats(rec_raw)
        print(f"[{name}] {hs}")

    out = {
        **base,
        "primitive_classes": primitive_classes,
        "channels": ["expr", "head", "jaw"],
        "prototypes": proto,
        "target_len": args.target_len,
        "channel_vae_checkpoints": {
            "expr": str(args.expr_checkpoint),
            "head": str(args.head_checkpoint),
            "jaw": str(args.jaw_checkpoint),
        },
        "synthetic_head_primitives": {
            "pitch_mag": args.pitch_mag,
            "roll_mag": args.roll_mag,
            "ramp_frames": args.ramp_frames,
            "hold_frames": args.hold_frames,
            "release_frames": args.release_frames,
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output_path)
    print(f"saved {args.output_path}")
    print(f"teacher npz dir: {teacher_root}")


if __name__ == "__main__":
    main()
