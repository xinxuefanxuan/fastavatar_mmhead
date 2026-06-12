#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from motion_model.motion_safety import motion_stats


def load_motion(path: Path, key: str) -> np.ndarray:
    data = np.load(path)
    arr = data[key] if key in data else data["motion_raw"] if "motion_raw" in data else data["motion"]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 56:
        raise ValueError(f"expected (T,>=56), got {arr.shape}")
    return arr[:, :56].astype(np.float32)


def proxy_landmarks(motion: np.ndarray) -> dict[str, np.ndarray]:
    # Lightweight FLAME-proxy geometry for environments where FLAME assets/runtime are absent.
    # It is intentionally deterministic and driven only by the documented 56D layout.
    t = motion.shape[0]
    expr = motion[:, :50]
    head = motion[:, 50:53]
    jaw = motion[:, 53:56]
    smile_axis = expr[:, :10].mean(axis=1) - expr[:, 10:20].mean(axis=1)
    smile = np.tanh(smile_axis)
    jaw_open = np.linalg.norm(jaw, axis=1)
    base_left = np.tile(np.array([-0.32, -0.10, 0.18], dtype=np.float32), (t, 1))
    base_right = np.tile(np.array([0.32, -0.10, 0.18], dtype=np.float32), (t, 1))
    upper = np.tile(np.array([0.0, -0.06, 0.22], dtype=np.float32), (t, 1))
    lower = np.tile(np.array([0.0, -0.14, 0.22], dtype=np.float32), (t, 1))
    left = base_left + np.stack([-0.04 * smile, 0.05 * smile, 0.015 * np.abs(smile)], axis=1)
    right = base_right + np.stack([0.04 * smile, 0.05 * smile, 0.015 * np.abs(smile)], axis=1)
    lower = lower + np.stack([np.zeros(t), -0.10 * jaw_open, 0.02 * jaw_open], axis=1)
    yaw = head[:, 1]
    boundary_left = np.stack([-0.55 - 0.10 * yaw, np.zeros(t), 0.05 + 0.20 * yaw], axis=1)
    boundary_right = np.stack([0.55 - 0.10 * yaw, np.zeros(t), 0.05 - 0.20 * yaw], axis=1)
    return {"mouth_left": left, "mouth_right": right, "upper_lip": upper, "lower_lip": lower, "boundary_left": boundary_left, "boundary_right": boundary_right}


def geometry_metrics(motion: np.ndarray) -> dict[str, list[float] | float]:
    lm = proxy_landmarks(motion)
    corner_width = np.linalg.norm(lm["mouth_right"] - lm["mouth_left"], axis=1)
    corner_disp = 0.5 * (np.linalg.norm(lm["mouth_left"] - lm["mouth_left"][0], axis=1) + np.linalg.norm(lm["mouth_right"] - lm["mouth_right"][0], axis=1))
    lip_dist = np.linalg.norm(lm["upper_lip"] - lm["lower_lip"], axis=1)
    boundary_motion = 0.5 * (np.linalg.norm(lm["boundary_left"] - lm["boundary_left"][0], axis=1) + np.linalg.norm(lm["boundary_right"] - lm["boundary_right"][0], axis=1))
    return {
        "mouth_corner_width": corner_width.astype(float).tolist(),
        "mouth_corner_displacement": corner_disp.astype(float).tolist(),
        "upper_lower_lip_distance": lip_dist.astype(float).tolist(),
        "jaw_opening": np.linalg.norm(motion[:, 53:56], axis=1).astype(float).tolist(),
        "boundary_motion": boundary_motion.astype(float).tolist(),
        "mouth_corner_displacement_max": float(corner_disp.max()) if len(corner_disp) else 0.0,
        "upper_lower_lip_distance_max": float(lip_dist.max()) if len(lip_dist) else 0.0,
        "boundary_motion_max": float(boundary_motion.max()) if len(boundary_motion) else 0.0,
    }


def render_frames(motion: np.ndarray, output_dir: Path, wireframe: bool = False) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "mesh_preview_skipped.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return []
    lm = proxy_landmarks(motion)
    frames_dir = output_dir / ("wire_frames" if wireframe else "mesh_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    theta = np.linspace(0, 2 * np.pi, 80)
    for i in range(motion.shape[0]):
        yaw = float(motion[i, 51])
        fig, ax = plt.subplots(figsize=(5, 5))
        face_x = 0.58 * np.cos(theta) - 0.12 * yaw
        face_y = 0.78 * np.sin(theta)
        ax.plot(face_x, face_y, color="black", lw=1 if wireframe else 2)
        ax.scatter([lm["mouth_left"][i, 0], lm["mouth_right"][i, 0]], [lm["mouth_left"][i, 1], lm["mouth_right"][i, 1]], c="red", s=20)
        ax.plot([lm["mouth_left"][i, 0], lm["mouth_right"][i, 0]], [lm["mouth_left"][i, 1], lm["mouth_right"][i, 1]], c="red", lw=1.5)
        ax.plot([lm["upper_lip"][i, 0], lm["lower_lip"][i, 0]], [lm["upper_lip"][i, 1], lm["lower_lip"][i, 1]], c="blue", lw=2)
        ax.text(0.02, 0.96, f"f={i} yaw={motion[i,51]:.3f}", transform=ax.transAxes, va="top")
        ax.set_aspect("equal")
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-1.0, 0.9)
        ax.grid(True, alpha=0.2)
        path = frames_dir / f"frame_{i:04d}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def write_video(frames: list[Path], out: Path, fps: int) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio
        imgs = [imageio.imread(p) for p in frames]
        imageio.mimsave(out, imgs, fps=fps)
    except Exception as exc:
        out.with_suffix(".video_skipped.txt").write_text(f"imageio video unavailable: {exc}\nframes_dir={frames[0].parent}\n", encoding="utf-8")


def save_curves(metrics: dict, motion: np.ndarray, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        out.with_suffix(".plot_skipped.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(metrics["mouth_corner_displacement"], label="mouth corner displacement")
    axes[0].plot(metrics["upper_lower_lip_distance"], label="lip distance")
    axes[1].plot(metrics["jaw_opening"], label="jaw opening")
    axes[2].plot(motion[:, 50], label="pitch")
    axes[2].plot(motion[:, 51], label="yaw")
    axes[2].plot(motion[:, 52], label="roll")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--motion_key", default="motion")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--wireframe", action="store_true")
    ap.add_argument("--save_views", action="store_true")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motion = load_motion(args.input, args.motion_key)
    metrics = geometry_metrics(motion)
    stats = motion_stats(motion)
    frames = render_frames(motion, args.output_dir, wireframe=False)
    write_video(frames, args.output_dir / "mesh_preview.mp4", args.fps)
    if args.wireframe:
        wire = render_frames(motion, args.output_dir, wireframe=True)
        write_video(wire, args.output_dir / "mesh_preview_wireframe.mp4", args.fps)
    if args.save_views and frames:
        import shutil
        shutil.copy2(frames[0], args.output_dir / "front_view_frame0.png")
        shutil.copy2(frames[len(frames)//2], args.output_dir / "side_view_mid.png")
    save_curves(metrics, motion, args.output_dir / "flame_motion_curves.png")
    report = {"input": str(args.input), "note": "Uses FLAME-proxy geometry when full project FLAME runtime/assets are unavailable.", "motion_stats": stats, "geometry_metrics": metrics}
    (args.output_dir / "flame_mesh_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "flame_mesh_report.md").write_text(
        "# FLAME Mesh Motion Preview\n\n"
        f"Input: `{args.input}`\n\n"
        f"Mouth corner displacement max: {metrics['mouth_corner_displacement_max']:.6f}\n\n"
        f"Upper/lower lip distance max: {metrics['upper_lower_lip_distance_max']:.6f}\n\n"
        f"Boundary motion max: {metrics['boundary_motion_max']:.6f}\n\n"
        "If this proxy/FLAME-level preview looks normal but FastAvatar renders show artifacts, suspect renderer/mask/source-view range rather than motion generation.\n",
        encoding="utf-8",
    )
    print(args.output_dir / "flame_mesh_report.md")


if __name__ == "__main__":
    main()
