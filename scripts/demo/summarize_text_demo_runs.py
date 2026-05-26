#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def find_first_npz(run_dir: Path) -> Path | None:
    npz_files = sorted((run_dir / "generated_npz").glob("*.npz"))
    return npz_files[0] if npz_files else None


def motion_stats(npz_path: Path) -> dict[str, str]:
    data = np.load(npz_path, allow_pickle=True)
    if "motion" not in data:
        raise ValueError(f"missing key 'motion' in {npz_path}")
    motion = np.asarray(data["motion"], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < 56:
        raise ValueError(f"invalid motion shape {motion.shape} in {npz_path}")

    expr = motion[:, :50]
    head = motion[:, 50:53]
    jaw = motion[:, 53:56]
    yaw = head[:, 1]

    expr_norm = np.linalg.norm(expr, axis=1)
    head_norm = np.linalg.norm(head, axis=1)
    jaw_norm = np.linalg.norm(jaw, axis=1)

    return {
        "yaw": f"{float(yaw.min()):.4f}/{float(yaw.max()):.4f}",
        "head": f"{float(head_norm.mean()):.4f}/{float(head_norm.max()):.4f}",
        "expr": f"{float(expr_norm.mean()):.4f}/{float(expr_norm.max()):.4f}",
        "jaw": f"{float(jaw_norm.mean()):.4f}/{float(jaw_norm.max()):.4f}",
    }


def rel_or_na(path: Path | None, root: Path) -> str:
    if path is None:
        return "N/A"
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=Path, default=Path("outputs/mmhead_debug/text_demo_runs"))
    ap.add_argument("--output_md", type=Path, default=Path("outputs/mmhead_debug/text_demo_runs/README.md"))
    args = ap.parse_args()

    runs_root = args.runs_root
    if not runs_root.exists():
        raise SystemExit(f"runs_root not found: {runs_root}")

    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()])

    lines = []
    lines.append("# Text-driven Demo Runs Summary")
    lines.append("")
    lines.append("Pipeline: text prompt -> rule parser -> latent composition -> FastAvatar render")
    lines.append("")
    lines.append("| run | npz | video | yaw min/max | head norm mean/max | expr norm mean/max | jaw norm mean/max |")
    lines.append("|---|---|---|---|---|---|---|")

    for run_dir in run_dirs:
        run_name = run_dir.name
        npz_path = find_first_npz(run_dir)
        video_path = run_dir / "video.mp4"
        if not video_path.exists():
            video_path = None

        yaw = head = expr = jaw = "N/A"
        if npz_path is None:
            note = "missing npz"
        else:
            note = ""
            try:
                st = motion_stats(npz_path)
                yaw, head, expr, jaw = st["yaw"], st["head"], st["expr"], st["jaw"]
            except Exception as e:
                note = f"stats error: {e}".replace("|", "/")

        npz_cell = rel_or_na(npz_path, runs_root)
        video_cell = rel_or_na(video_path, runs_root)
        if note:
            npz_cell = f"{npz_cell} ({note})"

        lines.append(f"| {run_name} | {npz_cell} | {video_cell} | {yaw} | {head} | {expr} | {jaw} |")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved summary: {args.output_md}")


if __name__ == "__main__":
    main()
