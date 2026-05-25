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


ROOT_META_FILES = [
    "canonical_flame_param.npz",
    "transforms.json",
    "transforms_train.json",
    "transforms_val.json",
    "transforms_test.json",
    "transforms_backup.json",
    "transforms_backup_flame.json",
]


def prepare_output_layout(
    neutral_template: Path,
    output_motion_dir: Path | None,
    output_motion_root: Path | None,
    sequence_name: str | None,
    overwrite: bool,
    motion_npz: Path,
    fastavatar_pack: bool,
    pack_root: Path | None,
) -> tuple[Path, Path | None, str | None, list[str]]:
    if fastavatar_pack:
        if pack_root is None:
            raise SystemExit("--pack_root is required when --fastavatar_pack is used")
        seq_name = sequence_name or motion_npz.stem
        root_dir = pack_root
        seq_dir = root_dir / seq_name
        if root_dir.exists():
            if not overwrite:
                raise SystemExit(f"pack root exists: {root_dir}; use --overwrite")
            shutil.rmtree(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        copied_root_files: list[str] = []
        for name in ROOT_META_FILES:
            src = neutral_template / name
            if src.exists():
                shutil.copy2(src, root_dir / name)
                copied_root_files.append(name)

        seq_dir.mkdir(parents=True, exist_ok=True)
        flame_src = neutral_template / "flame_param"
        if not flame_src.exists():
            raise SystemExit(f"missing flame_param in neutral_template: {neutral_template}")
        seq_flame = seq_dir / "flame_param"
        if seq_flame.exists() or seq_flame.is_symlink():
            if overwrite and seq_flame.resolve().is_relative_to(root_dir.resolve()):
                if seq_flame.is_symlink():
                    seq_flame.unlink()
                elif seq_flame.is_dir():
                    shutil.rmtree(seq_flame)
                else:
                    seq_flame.unlink()
            elif seq_flame.is_symlink():
                seq_flame.unlink()
            else:
                print(f"[WARN] existing real directory/file at {seq_flame}; keeping it untouched")
        if not seq_flame.exists():
            shutil.copytree(flame_src, seq_flame)

        processed_src = neutral_template / "processed_data"
        seq_processed = seq_dir / "processed_data"
        if processed_src.exists():
            if seq_processed.exists() or seq_processed.is_symlink():
                if seq_processed.is_symlink():
                    seq_processed.unlink()
                else:
                    print(f"[WARN] existing real directory/file at {seq_processed}; keeping it untouched")
            else:
                seq_processed.symlink_to(processed_src.resolve(), target_is_directory=True)

        # Create root-level symlinks expected by some FastAvatar paths:
        # pack_root/flame_param -> pack_root/sequence_name/flame_param
        # pack_root/processed_data -> pack_root/sequence_name/processed_data
        root_flame = root_dir / "flame_param"
        root_processed = root_dir / "processed_data"

        for link_path, target_path, name in [
            (root_flame, seq_flame, "flame_param"),
            (root_processed, seq_processed, "processed_data"),
        ]:
            if link_path.exists() or link_path.is_symlink():
                if link_path.is_symlink():
                    link_path.unlink()
                else:
                    print(f"[WARN] existing real directory/file at {link_path}; keeping it untouched (not replaced)")
                    continue
            if target_path.exists():
                try:
                    link_path.symlink_to(target_path, target_is_directory=True)
                except OSError as e:
                    print(f"[WARN] failed to create root symlink {name}: {e}")
        return seq_dir, root_dir, seq_name, copied_root_files

    if output_motion_root is not None:
        seq_name = sequence_name or motion_npz.stem
        root_dir = output_motion_root
        seq_dir = root_dir / seq_name
        if root_dir.exists():
            if not overwrite:
                raise SystemExit(f"output root exists: {root_dir}; use --overwrite")
            shutil.rmtree(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        copied_root_files: list[str] = []
        for name in ROOT_META_FILES:
            src = neutral_template / name
            if src.exists():
                shutil.copy2(src, root_dir / name)
                copied_root_files.append(name)

        seq_dir.mkdir(parents=True, exist_ok=True)
        for child in neutral_template.iterdir():
            if child.name in copied_root_files:
                continue
            dst = seq_dir / child.name
            if child.is_dir():
                shutil.copytree(child, dst)
            else:
                shutil.copy2(child, dst)
        return seq_dir, root_dir, seq_name, copied_root_files

    if output_motion_dir is None:
        raise SystemExit("either --output_motion_dir or --output_motion_root must be provided")
    if output_motion_dir.exists():
        if not overwrite:
            raise SystemExit(f"output exists: {output_motion_dir}; use --overwrite")
        shutil.rmtree(output_motion_dir)
    shutil.copytree(neutral_template, output_motion_dir)
    return output_motion_dir, None, None, []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_npz", type=Path, required=True)
    ap.add_argument("--neutral_template", type=Path, required=True)
    ap.add_argument("--output_motion_dir", type=Path, default=None)
    ap.add_argument("--output_motion_root", type=Path, default=None)
    ap.add_argument("--sequence_name", type=str, default=None)
    ap.add_argument("--fastavatar_pack", action="store_true")
    ap.add_argument("--pack_root", type=Path, default=None)
    ap.add_argument("--motion_key", type=str, default="motion")
    ap.add_argument("--head_target", choices=["neck_pose", "rotation"], default="neck_pose")
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--head_velocity_clamp", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    target_motion_dir, output_root, sequence_name, root_meta = prepare_output_layout(
        neutral_template=args.neutral_template,
        output_motion_dir=args.output_motion_dir,
        output_motion_root=args.output_motion_root,
        sequence_name=args.sequence_name,
        overwrite=args.overwrite,
        motion_npz=args.motion_npz,
        fastavatar_pack=args.fastavatar_pack,
        pack_root=args.pack_root,
    )
    frame_paths, payloads = load_template_frames(target_motion_dir)

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
    if output_root is not None:
        print(f"[Write] output_motion_root={output_root}")
        print(f"[Write] sequence_name={sequence_name}")
        print(f"[Write] root metadata files={root_meta}")
        print(f"[Write] sequence_dir={target_motion_dir}")
        if args.fastavatar_pack:
            root_flame = output_root / "flame_param"
            resolved_root_flame = root_flame.resolve()
            if "assets/sample_motion" in str(resolved_root_flame):
                raise SystemExit(
                    f"[ERROR] root flame_param resolves outside pack root and points to template: {resolved_root_flame}"
                )
            # post-write sanity stats from pack_root/flame_param
            sanity_frames = sorted((output_root / "flame_param").glob("*.npz"))
            if sanity_frames:
                expr_norms = []
                jaw_norms = []
                yaw_vals = []
                for fp in sanity_frames:
                    d = np.load(fp, allow_pickle=True)
                    expr = np.asarray(d["expr"], dtype=np.float32).reshape(-1)
                    jaw = np.asarray(d["jaw_pose"], dtype=np.float32).reshape(-1)
                    neck = np.asarray(d["neck_pose"], dtype=np.float32).reshape(-1)
                    expr_norms.append(float(np.linalg.norm(expr)))
                    jaw_norms.append(float(np.linalg.norm(jaw)))
                    if neck.shape[0] >= 2:
                        yaw_vals.append(float(neck[1]))
            print(f"[FastAvatarPack] pack_root={output_root}")
            print(f"[FastAvatarPack] sequence_name={sequence_name}")
            print(f"[FastAvatarPack] inference_motion_dir={target_motion_dir}")
            print(f"[FastAvatarPack] root_flame_param={output_root / 'flame_param'}")
            print(f"[FastAvatarPack] root_processed_data={output_root / 'processed_data'}")
            print(f"[FastAvatarPack] root_flame_param_resolved={resolved_root_flame}")
            if sanity_frames:
                print(f"[FastAvatarPack] neck_pose_yaw min/max={min(yaw_vals):.6f}/{max(yaw_vals):.6f}" if yaw_vals else "[FastAvatarPack] neck_pose_yaw min/max=NA/NA")
                print(f"[FastAvatarPack] expr norm mean/max={np.mean(expr_norms):.6f}/{np.max(expr_norms):.6f}")
                print(f"[FastAvatarPack] jaw norm mean/max={np.mean(jaw_norms):.6f}/{np.max(jaw_norms):.6f}")
    else:
        print(f"[Write] output_motion_dir={target_motion_dir}")
    print(f"[Write] flame_param files updated={len(frame_paths)}")


if __name__ == "__main__":
    main()
