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
from motion_model.motion_safety import DEFAULT_SAFETY_CONFIG, motion_stats, safety_flags


def load_motion(path: Path, key: str = "motion") -> np.ndarray:
    data = np.load(path)
    if key in data:
        arr = data[key]
    elif "motion_raw" in data:
        arr = data["motion_raw"]
    else:
        keys = list(data.keys())
        raise KeyError(f"{path} has no '{key}' or 'motion_raw' key; available={keys}")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 56:
        raise ValueError(f"{path} expected >=56 dims, got {arr.shape}")
    return arr[:, :56].astype(np.float32)


def curves(m: np.ndarray) -> dict[str, list[float]]:
    return {
        "expr_mean": m[:, :50].mean(axis=1).astype(float).tolist(),
        "expr_max_abs": np.max(np.abs(m[:, :50]), axis=1).astype(float).tolist(),
        "expr_norm": np.linalg.norm(m[:, :50], axis=1).astype(float).tolist(),
        "pitch": m[:, 50].astype(float).tolist(),
        "yaw": m[:, 51].astype(float).tolist(),
        "roll": m[:, 52].astype(float).tolist(),
        "jaw_mean": m[:, 53:56].mean(axis=1).astype(float).tolist(),
        "jaw_max_abs": np.max(np.abs(m[:, 53:56]), axis=1).astype(float).tolist(),
        "jaw_norm": np.linalg.norm(m[:, 53:56], axis=1).astype(float).tolist(),
        "head_velocity": np.r_[0.0, np.linalg.norm(np.diff(m[:, 50:53], axis=0), axis=1)].astype(float).tolist(),
        "expr_velocity": np.r_[0.0, np.linalg.norm(np.diff(m[:, :50], axis=0), axis=1)].astype(float).tolist(),
        "jaw_velocity": np.r_[0.0, np.linalg.norm(np.diff(m[:, 53:56], axis=0), axis=1)].astype(float).tolist(),
    }


def save_plot(all_curves: dict[str, dict], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        out.with_suffix(".plot_skipped.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    n = len(all_curves)
    fig, axes = plt.subplots(4, 1, figsize=(12, max(9, 3 * n)), sharex=False)
    for label, c in all_curves.items():
        axes[0].plot(c["expr_norm"], label=f"{label} expr_norm")
        axes[1].plot(c["yaw"], label=f"{label} yaw")
        axes[1].plot(c["pitch"], linestyle="--", label=f"{label} pitch")
        axes[1].plot(c["roll"], linestyle=":", label=f"{label} roll")
        axes[2].plot(c["jaw_norm"], label=f"{label} jaw_norm")
        axes[3].plot(c["head_velocity"], label=f"{label} head_vel")
        axes[3].plot(c["expr_velocity"], linestyle="--", label=f"{label} expr_vel")
        axes[3].plot(c["jaw_velocity"], linestyle=":", label=f"{label} jaw_vel")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def report_md(rows: list[dict]) -> str:
    lines = ["# Motion NPZ Diagnosis", "", "| file | frames | yaw min/max | pitch min/max | roll min/max | head max step | expr norm max | jaw max | flags |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        s = r["stats"]
        flags = ", ".join(k for k, v in r["flags"].items() if v) or "ok"
        lines.append(f"| {r['file']} | {s['num_frames']} | {s['yaw_min']:.5f}/{s['yaw_max']:.5f} | {s['pitch_min']:.5f}/{s['pitch_max']:.5f} | {s['roll_min']:.5f}/{s['roll_max']:.5f} | {s['head_step_max']:.5f} | {s['expr_norm_max']:.5f} | {s['jaw_abs_max']:.5f} | {flags} |")
    lines += ["", "## Interpretation", "", "Use this report to identify whether right-turn artifacts correlate with yaw amplitude/velocity, whether smile has weak expression norm, and whether jaw/open-mouth exceeds the configured safe range."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--threshold_abs_yaw", type=float, default=float(DEFAULT_SAFETY_CONFIG["max_abs_yaw"]))
    ap.add_argument("--threshold_head_step", type=float, default=float(DEFAULT_SAFETY_CONFIG["max_head_step"]))
    ap.add_argument("--threshold_expr_norm", type=float, default=float(DEFAULT_SAFETY_CONFIG["max_expr_norm"]))
    ap.add_argument("--threshold_jaw", type=float, default=float(DEFAULT_SAFETY_CONFIG["max_jaw"]))
    ap.add_argument("--motion_key", default="motion")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_SAFETY_CONFIG)
    cfg.update(max_abs_yaw=args.threshold_abs_yaw, max_head_step=args.threshold_head_step, max_expr_norm=args.threshold_expr_norm, max_jaw=args.threshold_jaw)
    rows = []
    all_curves = {}
    for path in args.input:
        try:
            m = load_motion(path, args.motion_key)
            s = motion_stats(m)
            f = safety_flags(s, cfg)
            c = curves(m)
            rows.append({"file": str(path), "stats": s, "flags": f, "curves": c})
            all_curves[path.stem] = c
        except Exception as exc:
            rows.append({"file": str(path), "error": str(exc), "stats": {}, "flags": {"load_error": True}, "curves": {}})
    (args.output_dir / "motion_stats.json").write_text(json.dumps({"thresholds": cfg, "items": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plot(all_curves, args.output_dir / "motion_curves.png")
    (args.output_dir / "motion_report.md").write_text(report_md([r for r in rows if "error" not in r]), encoding="utf-8")
    print(args.output_dir / "motion_report.md")


if __name__ == "__main__":
    main()
