#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CANONICAL_TASKS = [
    "turn_left",
    "turn_right",
    "smile",
    "open_mouth",
    "turn_left_and_smile",
    "turn_right_and_smile",
    "turn_left_and_open_mouth",
    "turn_right_and_open_mouth",
]

ALIASES = {
    "turn_left": ["turn_left"],
    "turn_right": ["turn_right"],
    "smile": ["smile"],
    "open_mouth": ["open_mouth", "mouth_open"],
    "turn_left_and_smile": ["turn_left_and_smile", "turn_left_smile"],
    "turn_right_and_smile": ["turn_right_and_smile", "turn_right_smile"],
    "turn_left_and_open_mouth": ["turn_left_and_open_mouth", "turn_left_open_mouth"],
    "turn_right_and_open_mouth": ["turn_right_and_open_mouth", "turn_right_open_mouth"],
}


def parse_roots(items: list[str]) -> dict[str, Path]:
    roots = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid root spec '{item}', expected name=path")
        name, path = item.split("=", 1)
        roots[name] = Path(path)
    return roots


def find_file(root: Path, task: str) -> Path | None:
    for stem in ALIASES[task]:
        p = root / f"{stem}.npz"
        if p.exists():
            return p
    return None


def load_motion(npz_path: Path, motion_key: str) -> np.ndarray:
    arr = np.load(npz_path, allow_pickle=False)
    if motion_key not in arr:
        raise ValueError(f"missing key '{motion_key}' in {npz_path}")
    m = np.asarray(arr[motion_key], dtype=np.float32)
    if m.ndim != 2 or m.shape[1] < 56:
        raise ValueError(f"invalid motion shape {m.shape} in {npz_path}")
    return m[:, :56]


def metrics(m: np.ndarray) -> dict:
    expr, head, jaw = m[:, :50], m[:, 50:53], m[:, 53:56]
    expr_n = np.linalg.norm(expr, axis=1)
    head_n = np.linalg.norm(head, axis=1)
    jaw_n = np.linalg.norm(jaw, axis=1)
    pitch, yaw, roll = head[:, 0], head[:, 1], head[:, 2]
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


def success(task: str, s: dict, th_expr: float, th_jaw: float, th_yaw: float) -> bool:
    if task == "turn_left":
        return s["yaw_max"] > th_yaw
    if task == "turn_right":
        return s["yaw_min"] < -th_yaw
    if task == "smile":
        return s["expr_norm_max"] > th_expr
    if task == "open_mouth":
        return s["jaw_norm_max"] > th_jaw
    if task == "turn_left_and_smile":
        return s["yaw_max"] > th_yaw and s["expr_norm_max"] > th_expr
    if task == "turn_right_and_smile":
        return s["yaw_min"] < -th_yaw and s["expr_norm_max"] > th_expr
    if task == "turn_left_and_open_mouth":
        return s["yaw_max"] > th_yaw and s["jaw_norm_max"] > th_jaw
    if task == "turn_right_and_open_mouth":
        return s["yaw_min"] < -th_yaw and s["jaw_norm_max"] > th_jaw
    return False


def bucket(task: str) -> str:
    if task in {"turn_left", "turn_right"}:
        return "direction"
    if task == "smile":
        return "expression"
    if task == "open_mouth":
        return "jaw"
    return "composition"


def evaluate_root(root: Path, motion_key: str, th_expr: float, th_jaw: float, th_yaw: float) -> dict:
    tasks = {}
    for task in CANONICAL_TASKS:
        p = find_file(root, task)
        rec = {"task": task, "found": p is not None, "file": str(p) if p else None}
        if p is None:
            rec["success"] = False
            rec["error"] = "missing"
        else:
            try:
                m = load_motion(p, motion_key)
                mm = metrics(m)
                rec.update(mm)
                rec["success"] = success(task, mm, th_expr, th_jaw, th_yaw)
            except Exception as e:
                rec["success"] = False
                rec["error"] = str(e)
        tasks[task] = rec

    c = {"direction": [0, 0], "expression": [0, 0], "jaw": [0, 0], "composition": [0, 0], "overall": [0, 0]}
    expr_max, abs_yaw_max, jaw_max = [], [], []
    found = 0
    for t in CANONICAL_TASKS:
        r = tasks[t]
        b = bucket(t)
        c[b][1] += 1
        c["overall"][1] += 1
        if r["found"] and "error" not in r:
            found += 1
            expr_max.append(r["expr_norm_max"])
            abs_yaw_max.append(max(abs(r["yaw_min"]), abs(r["yaw_max"])))
            jaw_max.append(r["jaw_norm_max"])
        if r["success"]:
            c[b][0] += 1
            c["overall"][0] += 1

    return {
        "num_tasks": len(CANONICAL_TASKS),
        "num_found": found,
        "direction_success_rate": c["direction"][0] / max(1, c["direction"][1]),
        "expression_success_rate": c["expression"][0] / max(1, c["expression"][1]),
        "jaw_success_rate": c["jaw"][0] / max(1, c["jaw"][1]),
        "composition_success_rate": c["composition"][0] / max(1, c["composition"][1]),
        "overall_success_rate": c["overall"][0] / max(1, c["overall"][1]),
        "avg_expr_max": float(np.mean(expr_max)) if expr_max else 0.0,
        "avg_abs_yaw_max": float(np.mean(abs_yaw_max)) if abs_yaw_max else 0.0,
        "avg_jaw_max": float(np.mean(jaw_max)) if jaw_max else 0.0,
        "tasks": tasks,
    }


def write_csv_summary(path: Path, all_res: dict[str, dict]) -> None:
    fields = [
        "method", "num_tasks", "num_found", "direction_success_rate", "expression_success_rate",
        "jaw_success_rate", "composition_success_rate", "overall_success_rate", "avg_expr_max",
        "avg_abs_yaw_max", "avg_jaw_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, r in all_res.items():
            row = {k: r.get(k) for k in fields if k != "method"}
            row["method"] = name
            w.writerow(row)


def write_csv_tasks(path: Path, all_res: dict[str, dict]) -> None:
    fields = ["task", "method", "found", "success", "yaw_min", "yaw_max", "expr_norm_max", "jaw_norm_max", "file", "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for task in CANONICAL_TASKS:
            for name, r in all_res.items():
                t = r["tasks"][task]
                w.writerow({
                    "task": task,
                    "method": name,
                    "found": t.get("found", False),
                    "success": t.get("success", False),
                    "yaw_min": t.get("yaw_min"),
                    "yaw_max": t.get("yaw_max"),
                    "expr_norm_max": t.get("expr_norm_max"),
                    "jaw_norm_max": t.get("jaw_norm_max"),
                    "file": t.get("file"),
                    "error": t.get("error", ""),
                })


def write_md(path: Path, all_res: dict[str, dict]) -> None:
    lines = ["# Text Motion Root Comparison", "", "## Per-method Summary", "", "| method | found | dir | expr | jaw | comp | overall | avg_expr_max | avg_abs_yaw_max | avg_jaw_max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in all_res.items():
        lines.append(
            f"| {name} | {r['num_found']}/{r['num_tasks']} | {r['direction_success_rate']:.3f} | {r['expression_success_rate']:.3f} | {r['jaw_success_rate']:.3f} | {r['composition_success_rate']:.3f} | {r['overall_success_rate']:.3f} | {r['avg_expr_max']:.4f} | {r['avg_abs_yaw_max']:.4f} | {r['avg_jaw_max']:.4f} |"
        )

    lines += ["", "## Per-task Comparison", "", "| task | " + " | ".join(all_res.keys()) + " |", "|---|" + "|".join(["---"] * len(all_res)) + "|"]
    for task in CANONICAL_TASKS:
        cells = []
        for _, r in all_res.items():
            t = r["tasks"][task]
            if not t.get("found"):
                cells.append("❌ missing")
            elif "error" in t:
                cells.append("❌ error")
            else:
                cells.append("✅" if t.get("success") else "❌")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")

    best = max(all_res.items(), key=lambda kv: kv[1]["overall_success_rate"]) if all_res else None
    lines += ["", "## Interpretation", ""]
    if best:
        lines.append(f"Best overall by this metric is **{best[0]}** with overall success rate **{best[1]['overall_success_rate']:.3f}**.")
        lines.append("Direction/expr/jaw/composition rates in the summary table can be used for P6.8 vs P7.2 paper reporting.")
    else:
        lines.append("No valid roots were evaluated.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/mmhead_debug/eval_text_motion_compare"))
    ap.add_argument("--threshold_expr", type=float, default=1.5)
    ap.add_argument("--threshold_jaw", type=float, default=0.01)
    ap.add_argument("--threshold_yaw", type=float, default=0.05)
    ap.add_argument("--motion_key", type=str, default="motion")
    args = ap.parse_args()

    roots = parse_roots(args.roots)
    results = {
        name: evaluate_root(path, args.motion_key, args.threshold_expr, args.threshold_jaw, args.threshold_yaw)
        for name, path in roots.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "detailed_metrics.json").write_text(json.dumps({"roots": {k: str(v) for k, v in roots.items()}, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv_summary(args.output_dir / "per_method_summary.csv", results)
    write_csv_tasks(args.output_dir / "per_task_comparison.csv", results)
    write_md(args.output_dir / "comparison_report.md", results)

    print(f"[Compare] wrote {args.output_dir / 'detailed_metrics.json'}")
    print(f"[Compare] wrote {args.output_dir / 'per_method_summary.csv'}")
    print(f"[Compare] wrote {args.output_dir / 'per_task_comparison.csv'}")
    print(f"[Compare] wrote {args.output_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
