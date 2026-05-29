#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def load_payloads(motion_dir: Path):
    frame_paths = sorted((motion_dir / "flame_param").glob("*.npz"))
    if not frame_paths:
        raise SystemExit(f"No frame files found in: {motion_dir / 'flame_param'}")
    payloads = [dict(np.load(p, allow_pickle=True)) for p in frame_paths]
    return frame_paths, payloads


def save_payloads(frame_paths, payloads):
    for p, payload in zip(frame_paths, payloads):
        np.savez(p, **payload)


def apply_neck_ramp(payloads, axis: int, sign: int, num_frames: int, amplitude: float):
    neck = np.stack([np.asarray(p["neck_pose"], dtype=np.float32) for p in payloads], axis=0)
    orig_shape = neck.shape
    if neck.ndim < 2:
        raise SystemExit(f"neck_pose must have at least 2 dims, got shape={orig_shape}")
    neck_2d = neck.reshape(neck.shape[0], -1)
    if neck_2d.shape[1] <= axis:
        raise SystemExit(f"neck_pose flattened dim={neck_2d.shape[1]} is smaller than axis index {axis}")

    n = min(num_frames, neck_2d.shape[0])
    ramp = np.linspace(-1.0, 1.0, n, dtype=np.float32) * float(amplitude) * float(sign)
    out = neck_2d.copy()
    out[:n, axis] = out[:n, axis] + ramp
    out = out.reshape(orig_shape).astype(np.float32)

    for i, payload in enumerate(payloads):
        payload["neck_pose"] = out[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_motion", type=Path, required=True)
    ap.add_argument("--output_root", type=Path, required=True)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--amplitude", type=float, default=0.2)
    args = ap.parse_args()

    if args.output_root.exists():
        raise SystemExit(f"output_root already exists: {args.output_root}")

    dirs = [
        ("axis0_pos", 0, 1),
        ("axis0_neg", 0, -1),
        ("axis1_pos", 1, 1),
        ("axis1_neg", 1, -1),
        ("axis2_pos", 2, 1),
        ("axis2_neg", 2, -1),
    ]

    for name, axis, sign in dirs:
        dst = args.output_root / name
        shutil.copytree(args.template_motion, dst)
        frame_paths, payloads = load_payloads(dst)
        apply_neck_ramp(payloads, axis=axis, sign=sign, num_frames=args.num_frames, amplitude=args.amplitude)
        save_payloads(frame_paths, payloads)
        print(f"[calib] wrote {dst} axis={axis} sign={sign} frames={min(args.num_frames, len(frame_paths))}")


if __name__ == "__main__":
    main()
