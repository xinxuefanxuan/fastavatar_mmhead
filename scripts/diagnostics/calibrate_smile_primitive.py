#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.diagnostics.render_flame_mesh_sequence import geometry_metrics, load_motion


def candidate_paths(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in inputs:
        if p.is_dir():
            out.extend(sorted(p.rglob("*.npz")))
        elif p.suffix == ".npz":
            out.append(p)
    return out


def score_motion(path: Path, key: str) -> dict:
    m = load_motion(path, key)
    metrics = geometry_metrics(m)
    expr_norm = np.linalg.norm(m[:, :50], axis=1)
    jaw_norm = np.linalg.norm(m[:, 53:56], axis=1)
    lower_face = float(np.std(m[:, 20:50])) if m.shape[0] else 0.0
    corner = float(metrics["mouth_corner_displacement_max"])
    lip = float(metrics["upper_lower_lip_distance_max"])
    score = corner * 4.0 + lower_face * 1.5 + float(expr_norm.max()) * 0.25 - float(jaw_norm.max()) * 0.15
    return {
        "path": str(path),
        "score": score,
        "mouth_corner_displacement_max": corner,
        "upper_lower_lip_distance_max": lip,
        "lower_face_deformation": lower_face,
        "expr_norm_max": float(expr_norm.max()) if len(expr_norm) else 0.0,
        "expr_norm_mean": float(expr_norm.mean()) if len(expr_norm) else 0.0,
        "jaw_norm_max": float(jaw_norm.max()) if len(jaw_norm) else 0.0,
        "jaw_norm_mean": float(jaw_norm.mean()) if len(jaw_norm) else 0.0,
        "num_frames": int(m.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", type=Path, required=True, help="NPZ files or directories containing generated/dataset motions")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--motion_key", default="motion")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in candidate_paths(args.input):
        name = p.name.lower()
        if "smile" not in name and "happy" not in name and "grin" not in name:
            # Keep broad support for generated directories, but prefer smile-like names.
            pass
        try:
            rows.append(score_motion(p, args.motion_key))
        except Exception as exc:
            print(f"[WARN] skip {p}: {exc}")
    rows.sort(key=lambda r: r["score"], reverse=True)
    rows = rows[: max(1, args.topk)]
    csv_path = args.output_dir / "smile_candidates.csv"
    fields = ["rank", "path", "score", "mouth_corner_displacement_max", "upper_lower_lip_distance_max", "lower_face_deformation", "expr_norm_max", "expr_norm_mean", "jaw_norm_max", "jaw_norm_mean", "num_frames"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"rank": i, **r})
    best = rows[0] if rows else None
    rec = {
        "smile_profile": "strong" if best and best["mouth_corner_displacement_max"] < 0.035 else "default",
        "smile_gain": 2.0 if best and best["mouth_corner_displacement_max"] < 0.035 else 1.5,
        "selection_metric": "mouth_corner_displacement + lower_face_deformation, with jaw penalty",
        "best_candidate": best,
        "prototype_config_hint": {"expr": {"smile": "increase localized mouth-expression delta only"}, "jaw": {"smile": "keep light coupling; avoid open-mouth leakage"}},
    }
    (args.output_dir / "smile_preset_recommendation.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Smile Primitive Calibration", "", "Smile ranking is based on mouth-corner displacement, lip geometry, lower-face deformation, expression norm, and a jaw-opening penalty.", "", "| rank | score | mouth corner max | lip distance max | expr max | jaw max | path |", "|---:|---:|---:|---:|---:|---:|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['score']:.6f} | {r['mouth_corner_displacement_max']:.6f} | {r['upper_lower_lip_distance_max']:.6f} | {r['expr_norm_max']:.6f} | {r['jaw_norm_max']:.6f} | `{r['path']}` |")
    lines += ["", "## Recommendation", "", f"```json\n{json.dumps(rec, ensure_ascii=False, indent=2)}\n```"]
    (args.output_dir / "smile_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "smile_calibration_report.md")


if __name__ == "__main__":
    main()
