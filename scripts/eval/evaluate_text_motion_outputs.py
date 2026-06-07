#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

EXPECTED = [
    "turn_left",
    "turn_right",
    "smile",
    "open_mouth",
    "turn_left_and_smile",
    "turn_right_and_smile",
    "turn_left_and_open_mouth",
    "turn_right_and_open_mouth",
]


def load_motion(npz_path: Path) -> np.ndarray:
    arrs = np.load(npz_path, allow_pickle=False)
    if "motion" not in arrs:
        raise ValueError(f"missing 'motion' field: {npz_path}")
    motion = np.asarray(arrs["motion"], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < 56:
        raise ValueError(f"invalid motion shape {motion.shape} in {npz_path}")
    return motion[:, :56]


def compute_stats(motion: np.ndarray) -> dict:
    expr = motion[:, :50]
    head = motion[:, 50:53]
    jaw = motion[:, 53:56]

    expr_n = np.linalg.norm(expr, axis=1)
    head_n = np.linalg.norm(head, axis=1)
    jaw_n = np.linalg.norm(jaw, axis=1)

    pitch = head[:, 0]
    yaw = head[:, 1]
    roll = head[:, 2]

    return {
        "expr_norm_mean": float(expr_n.mean()),
        "expr_norm_max": float(expr_n.max()),
        "head_norm_mean": float(head_n.mean()),
        "head_norm_max": float(head_n.max()),
        "jaw_norm_mean": float(jaw_n.mean()),
        "jaw_norm_max": float(jaw_n.max()),
        "pitch_min": float(pitch.min()),
        "pitch_max": float(pitch.max()),
        "yaw_min": float(yaw.min()),
        "yaw_max": float(yaw.max()),
        "roll_min": float(roll.min()),
        "roll_max": float(roll.max()),
    }


def evaluate_task(task: str, s: dict, th_expr: float, th_jaw: float, th_yaw: float) -> bool:
    checks: dict[str, Callable[[], bool]] = {
        "turn_left": lambda: s["yaw_max"] > th_yaw,
        "turn_right": lambda: s["yaw_min"] < -th_yaw,
        "smile": lambda: s["expr_norm_max"] > th_expr,
        "open_mouth": lambda: s["jaw_norm_max"] > th_jaw,
        "turn_left_and_smile": lambda: s["yaw_max"] > th_yaw and s["expr_norm_max"] > th_expr,
        "turn_right_and_smile": lambda: s["yaw_min"] < -th_yaw and s["expr_norm_max"] > th_expr,
        "turn_left_and_open_mouth": lambda: s["yaw_max"] > th_yaw and s["jaw_norm_max"] > th_jaw,
        "turn_right_and_open_mouth": lambda: s["yaw_min"] < -th_yaw and s["jaw_norm_max"] > th_jaw,
    }
    return checks[task]()


def bucket(task: str) -> str:
    if task in {"turn_left", "turn_right"}:
        return "direction"
    if task in {"smile"}:
        return "expression"
    if task in {"open_mouth"}:
        return "jaw"
    return "composition"


def evaluate_root(root: Path, th_expr: float, th_jaw: float, th_yaw: float) -> dict:
    per_task = {}
    counts = {"direction": [0, 0], "expression": [0, 0], "jaw": [0, 0], "composition": [0, 0], "overall": [0, 0]}

    for task in EXPECTED:
        npz = root / f"{task}.npz"
        rec = {"task": task, "file": str(npz), "exists": npz.exists()}
        if not npz.exists():
            rec["success"] = False
            rec["error"] = "missing file"
        else:
            try:
                motion = load_motion(npz)
                st = compute_stats(motion)
                rec.update(st)
                rec["success"] = evaluate_task(task, st, th_expr, th_jaw, th_yaw)
            except Exception as e:
                rec["success"] = False
                rec["error"] = str(e)

        per_task[task] = rec
        b = bucket(task)
        counts[b][1] += 1
        counts["overall"][1] += 1
        if rec["success"]:
            counts[b][0] += 1
            counts["overall"][0] += 1

    rates = {
        "direction_success_rate": counts["direction"][0] / max(1, counts["direction"][1]),
        "expression_success_rate": counts["expression"][0] / max(1, counts["expression"][1]),
        "jaw_success_rate": counts["jaw"][0] / max(1, counts["jaw"][1]),
        "composition_success_rate": counts["composition"][0] / max(1, counts["composition"][1]),
        "overall_success_rate": counts["overall"][0] / max(1, counts["overall"][1]),
    }

    return {
        "root": str(root),
        "thresholds": {"expr": th_expr, "jaw": th_jaw, "yaw": th_yaw},
        "counts": counts,
        "rates": rates,
        "tasks": per_task,
    }


def build_markdown(single: dict | None, multi: dict | None) -> str:
    lines = ["# Text Motion Evaluation", ""]
    if single is not None:
        lines += [f"Root: `{single['root']}`", "", "## Per-task", "", "| task | success | yaw(min/max) | expr_max | jaw_max |", "|---|---:|---:|---:|---:|"]
        for t in EXPECTED:
            r = single["tasks"][t]
            if r.get("exists") and "error" not in r:
                yaw = f"{r['yaw_min']:.4f}/{r['yaw_max']:.4f}"
                expr = f"{r['expr_norm_max']:.4f}"
                jaw = f"{r['jaw_norm_max']:.4f}"
            else:
                yaw = expr = jaw = "N/A"
            lines.append(f"| {t} | {'✅' if r.get('success') else '❌'} | {yaw} | {expr} | {jaw} |")
        rt = single["rates"]
        lines += ["", "## Aggregate", "", "| metric | value |", "|---|---:|",
                  f"| direction_success_rate | {rt['direction_success_rate']:.3f} |",
                  f"| expression_success_rate | {rt['expression_success_rate']:.3f} |",
                  f"| jaw_success_rate | {rt['jaw_success_rate']:.3f} |",
                  f"| composition_success_rate | {rt['composition_success_rate']:.3f} |",
                  f"| overall_success_rate | {rt['overall_success_rate']:.3f} |"]
    if multi is not None:
        lines += ["", "## Multi-root Comparison", "", "| method | direction | expression | jaw | composition | overall |", "|---|---:|---:|---:|---:|---:|"]
        for name, ev in multi.items():
            r = ev["rates"]
            lines.append(
                f"| {name} | {r['direction_success_rate']:.3f} | {r['expression_success_rate']:.3f} | {r['jaw_success_rate']:.3f} | {r['composition_success_rate']:.3f} | {r['overall_success_rate']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def parse_roots(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --generated_roots item '{item}', expected name=path")
        name, path = item.split("=", 1)
        out[name] = Path(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated_root", type=Path, default=None)
    ap.add_argument("--generated_roots", nargs="*", default=None)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--output_md", type=Path, required=True)
    ap.add_argument("--threshold_expr", type=float, default=1.5)
    ap.add_argument("--threshold_jaw", type=float, default=0.01)
    ap.add_argument("--threshold_yaw", type=float, default=0.03)
    args = ap.parse_args()

    if args.generated_root is None and not args.generated_roots:
        raise SystemExit("provide --generated_root or --generated_roots")

    result = {}
    single = None
    multi = None

    if args.generated_root is not None:
        single = evaluate_root(args.generated_root, args.threshold_expr, args.threshold_jaw, args.threshold_yaw)
        result["single"] = single

    if args.generated_roots:
        roots = parse_roots(args.generated_roots)
        multi = {name: evaluate_root(path, args.threshold_expr, args.threshold_jaw, args.threshold_yaw) for name, path in roots.items()}
        result["multi"] = multi

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(single, multi), encoding="utf-8")

    print(f"[Eval] saved json: {args.output_json}")
    print(f"[Eval] saved md: {args.output_md}")


if __name__ == "__main__":
    main()
