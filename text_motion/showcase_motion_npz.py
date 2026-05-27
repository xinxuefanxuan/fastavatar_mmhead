#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _stats(motion: np.ndarray) -> dict[str, tuple[float, float]]:
    expr = motion[:, :50]
    head = motion[:, 50:53]
    jaw = motion[:, 53:56]
    expr_n = np.linalg.norm(expr, axis=1)
    head_n = np.linalg.norm(head, axis=1)
    jaw_n = np.linalg.norm(jaw, axis=1)
    yaw = head[:, 1]
    return {
        "expr norm mean/max": (float(expr_n.mean()), float(expr_n.max())),
        "head norm mean/max": (float(head_n.mean()), float(head_n.max())),
        "jaw norm mean/max": (float(jaw_n.mean()), float(jaw_n.max())),
        "yaw min/max": (float(yaw.min()), float(yaw.max())),
    }


def _pick_target(motion: np.ndarray, strategy: str) -> int:
    expr_n = np.linalg.norm(motion[:, :50], axis=1)
    head_n = np.linalg.norm(motion[:, 50:53], axis=1)
    jaw_n = np.linalg.norm(motion[:, 53:56], axis=1)
    if strategy == "last":
        return int(motion.shape[0] - 1)
    if strategy == "max_head":
        return int(np.argmax(head_n))
    if strategy == "max_expr":
        return int(np.argmax(expr_n))
    if strategy == "max_jaw":
        return int(np.argmax(jaw_n))
    score = expr_n + 10.0 * head_n + 5.0 * jaw_n
    return int(np.argmax(score))


def _make_envelope(output_len: int, ramp_frames: int, hold_frames: int, release_frames: int, release_ratio: float) -> np.ndarray:
    if output_len <= 0:
        raise ValueError("output_len must be > 0")
    ramp = max(0, int(ramp_frames))
    release = max(0, int(release_frames))
    hold = max(0, int(hold_frames))
    fixed = ramp + release
    if fixed > output_len:
        overflow = fixed - output_len
        if release >= overflow:
            release -= overflow
        else:
            overflow -= release
            release = 0
            ramp = max(0, ramp - overflow)
    hold = max(0, output_len - ramp - release)

    env_parts: list[np.ndarray] = []
    if ramp > 0:
        env_parts.append(np.linspace(0.0, 1.0, ramp, endpoint=True, dtype=np.float32))
    if hold > 0:
        env_parts.append(np.ones((hold,), dtype=np.float32))
    if release > 0:
        env_parts.append(np.linspace(1.0, float(release_ratio), release, endpoint=True, dtype=np.float32))
    if not env_parts:
        env_parts.append(np.ones((output_len,), dtype=np.float32))

    env = np.concatenate(env_parts, axis=0)
    if env.shape[0] < output_len:
        env = np.concatenate([env, np.full((output_len - env.shape[0],), float(release_ratio), dtype=np.float32)], axis=0)
    elif env.shape[0] > output_len:
        env = env[:output_len]
    return env.astype(np.float32)


def _apply_envelope(target_vec: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    return (envelope[:, None] * target_vec[None, :]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--output_len", type=int, default=64)
    ap.add_argument("--ramp_frames", type=int, default=16)
    ap.add_argument("--hold_frames", type=int, default=36)
    ap.add_argument("--release_frames", type=int, default=12)
    ap.add_argument("--release_ratio", type=float, default=0.75)
    ap.add_argument("--target_strategy", type=str, default="auto", choices=["auto", "last", "max_head", "max_expr", "max_jaw"])
    ap.add_argument("--motion_key", type=str, default="motion")
    ap.add_argument("--amplify_expr", type=float, default=1.0)
    ap.add_argument("--amplify_head", type=float, default=1.0)
    ap.add_argument("--amplify_jaw", type=float, default=1.0)
    args = ap.parse_args()

    src = np.load(args.input_npz, allow_pickle=False)
    if args.motion_key not in src:
        raise SystemExit(f"missing motion_key '{args.motion_key}' in npz")

    motion = np.asarray(src[args.motion_key], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < 56:
        raise SystemExit(f"motion must be [T,56+] got {motion.shape}")
    motion = motion[:, :56]

    before = _stats(motion)
    idx = _pick_target(motion, args.target_strategy)
    target = motion[idx].copy()
    target[:50] *= float(args.amplify_expr)
    target[50:53] *= float(args.amplify_head)
    target[53:56] *= float(args.amplify_jaw)

    env = _make_envelope(
        output_len=args.output_len,
        ramp_frames=args.ramp_frames,
        hold_frames=args.hold_frames,
        release_frames=args.release_frames,
        release_ratio=args.release_ratio,
    )

    out_motion = _apply_envelope(target, env)

    out_motion_norm = out_motion.copy()
    if "motion_norm" in src:
        src_norm = np.asarray(src["motion_norm"], dtype=np.float32)
        if src_norm.ndim == 2 and src_norm.shape[1] >= 56:
            src_norm = src_norm[:, :56]
            idx_norm = min(idx, src_norm.shape[0] - 1)
            target_norm = src_norm[idx_norm].copy()
            target_norm[:50] *= float(args.amplify_expr)
            target_norm[50:53] *= float(args.amplify_head)
            target_norm[53:56] *= float(args.amplify_jaw)
            out_motion_norm = _apply_envelope(target_norm, env)
        else:
            print("[WARN] motion_norm present but shape unexpected; using enveloped motion for motion_norm.")
    else:
        print("[WARN] motion_norm not present; using enveloped motion for motion_norm.")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        motion=out_motion.astype(np.float32),
        motion_raw=out_motion.astype(np.float32),
        motion_norm=out_motion_norm.astype(np.float32),
        expr_delta=out_motion[:, :50].astype(np.float32),
        head_delta=out_motion[:, 50:53].astype(np.float32),
        jaw_delta=out_motion[:, 53:56].astype(np.float32),
    )

    after = _stats(out_motion)
    print(f"selected target frame: {idx}")
    print(f"envelope length/min/max: {env.shape[0]} / {float(env.min()):.6f} / {float(env.max()):.6f}")
    print("[Before]")
    for k, (a, b) in before.items():
        print(f"  {k}: {a:.6f} / {b:.6f}")
    print("[After]")
    for k, (a, b) in after.items():
        print(f"  {k}: {a:.6f} / {b:.6f}")
    print(f"output={args.output_npz}")


if __name__ == "__main__":
    main()
