#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def load_template_frames(motion_dir: Path):
    flame_dir = motion_dir / "flame_param"
    frame_paths = sorted(flame_dir.glob("*.npz"))
    if not frame_paths:
        raise SystemExit(f"no frame npz found in {flame_dir}")
    payloads = [dict(np.load(p, allow_pickle=True)) for p in frame_paths]
    return frame_paths, payloads


def to_2d(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    return a.reshape(a.shape[0], -1)


def smooth_centered(x: np.ndarray, window: int = 5) -> np.ndarray:
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


def repeat_or_trim(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    pad = np.repeat(x[-1:], n - x.shape[0], axis=0)
    return np.concatenate([x, pad], axis=0)


def apply_delta(payloads, key: str, delta: np.ndarray):
    base = np.stack([np.asarray(p[key], dtype=np.float32) for p in payloads], axis=0)
    orig_shape = base.shape
    base2d = base.reshape(base.shape[0], -1)
    delta2d = to_2d(delta)
    n = min(base2d.shape[0], delta2d.shape[0])
    edit_dim = min(base2d.shape[1], delta2d.shape[1])
    if base2d.shape[1] != delta2d.shape[1]:
        print(f"[WARN] {key} dim mismatch target={base2d.shape[1]} source={delta2d.shape[1]} edit_dim={edit_dim}")
    out = base2d.copy()
    out[:n, :edit_dim] = out[:n, :edit_dim] + delta2d[:n, :edit_dim]
    out = out.reshape(orig_shape).astype(np.float32)
    for i, p in enumerate(payloads):
        p[key] = out[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_npz", type=Path, required=True)
    ap.add_argument("--neutral_template", type=Path, required=True)
    ap.add_argument("--output_motion_dir", type=Path, required=True)
    ap.add_argument("--motion_key", type=str, default="motion")
    ap.add_argument("--head_target", choices=["neck_pose", "rotation"], default="neck_pose")
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--head_velocity_clamp", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_motion_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {args.output_motion_dir}; use --overwrite")
        shutil.rmtree(args.output_motion_dir)

    shutil.copytree(args.neutral_template, args.output_motion_dir)
    frame_paths, payloads = load_template_frames(args.output_motion_dir)

    data = np.load(args.motion_npz, allow_pickle=True)
    if args.motion_key not in data:
        raise SystemExit(f"motion_key not found: {args.motion_key}")
    motion = np.asarray(data[args.motion_key], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < 56:
        raise SystemExit(f"motion must be [T,56+] got {motion.shape}")

    expr_delta = motion[:, 0:50]
    head_delta = motion[:, 50:53]
    jaw_delta = motion[:, 53:56]

    if args.smooth:
        expr_delta = smooth_centered(expr_delta, 5)
        head_delta = smooth_centered(head_delta, 5)
        jaw_delta = smooth_centered(jaw_delta, 5)
    if args.head_velocity_clamp is not None and args.head_velocity_clamp > 0:
        head_delta = clamp_velocity(head_delta, float(args.head_velocity_clamp))

    n = len(payloads)
    expr_delta = repeat_or_trim(expr_delta, n)
    head_delta = repeat_or_trim(head_delta, n)
    jaw_delta = repeat_or_trim(jaw_delta, n)

    print(f"[Load] motion_npz={args.motion_npz}")
    print(f"[Load] motion shape={motion.shape}")
    print(f"[Stats] expr norm mean={np.linalg.norm(expr_delta, axis=1).mean():.6f} max={np.linalg.norm(expr_delta, axis=1).max():.6f}")
    print(f"[Stats] head norm mean={np.linalg.norm(head_delta, axis=1).mean():.6f} max={np.linalg.norm(head_delta, axis=1).max():.6f}")
    print(f"[Stats] jaw  norm mean={np.linalg.norm(jaw_delta, axis=1).mean():.6f} max={np.linalg.norm(jaw_delta, axis=1).max():.6f}")

    sample_keys = sorted(payloads[0].keys())
    print(f"[Template] keys={sample_keys}")
    for k in ["expr", args.head_target, "jaw_pose", "shape", "rotation", "translation", "eyes_pose"]:
        if k in payloads[0]:
            print(f"[Template] {k} shape={np.asarray(payloads[0][k]).shape}")

    apply_delta(payloads, "expr", expr_delta)
    apply_delta(payloads, args.head_target, head_delta)
    apply_delta(payloads, "jaw_pose", jaw_delta)

    for p, payload in zip(frame_paths, payloads):
        np.savez(p, **payload)
    print(f"[Write] output_motion_dir={args.output_motion_dir}")
    print(f"[Write] frame files={len(frame_paths)}")


if __name__ == "__main__":
    main()
