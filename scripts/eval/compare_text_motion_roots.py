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


def to_json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


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
    yaw_min, yaw_max = float(yaw.min()), float(yaw.max())
    return {
        "expr_norm_mean": float(expr_n.mean()),
        "expr_norm_max": float(expr_n.max()),
        "head_norm_mean": float(head_n.mean()),
        "head_norm_max": float(head_n.max()),
        "jaw_norm_mean": float(jaw_n.mean()),
        "jaw_norm_max": float(jaw_n.max()),
        "pitch_min": float(pitch.min()),
        "pitch_max": float(pitch.max()),
        "yaw_min": yaw_min,
        "yaw_max": yaw_max,
        "roll_min": float(roll.min()),
        "roll_max": float(roll.max()),
        "abs_yaw_max": max(abs(yaw_min), abs(yaw_max)),
    }


def eval_task(task: str, s: dict, a) -> tuple[bool, bool, list[str], bool, bool]:
    # activation
    if task == "turn_left":
        activation = s["yaw_max"] > a.threshold_yaw
    elif task == "turn_right":
        activation = s["yaw_min"] < -a.threshold_yaw
    elif task == "smile":
        activation = s["expr_norm_max"] > a.threshold_expr
    elif task == "open_mouth":
        activation = s["jaw_norm_max"] > a.threshold_jaw
    elif task == "turn_left_and_smile":
        activation = s["yaw_max"] > a.threshold_yaw and s["expr_norm_max"] > a.threshold_expr
    elif task == "turn_right_and_smile":
        activation = s["yaw_min"] < -a.threshold_yaw and s["expr_norm_max"] > a.threshold_expr
    elif task == "turn_left_and_open_mouth":
        activation = s["yaw_max"] > a.threshold_yaw and s["jaw_norm_max"] > a.threshold_jaw
    else:
        activation = s["yaw_min"] < -a.threshold_yaw and s["jaw_norm_max"] > a.threshold_jaw

    over_expr = s["expr_norm_max"] > a.max_expr
    over_yaw = s["abs_yaw_max"] > a.max_abs_yaw
    over_jaw = s["jaw_norm_max"] > a.max_jaw
    over_motion = over_expr or over_yaw or over_jaw

    reasons = []
    dir_purity_ok = True
    leakage_ok = True

    if task.startswith("turn_left"):
        if s["yaw_min"] <= -a.direction_purity_yaw:
            dir_purity_ok = False
            reasons.append("direction_impure_left")
    if task.startswith("turn_right"):
        if s["yaw_max"] >= a.direction_purity_yaw:
            dir_purity_ok = False
            reasons.append("direction_impure_right")

    if task in {"turn_left", "turn_right"}:
        if s["expr_norm_max"] > a.leak_expr_for_head_only:
            leakage_ok = False
            reasons.append("expr_leak_head_only")
        if s["jaw_norm_max"] > a.leak_jaw_for_head_only:
            leakage_ok = False
            reasons.append("jaw_leak_head_only")
    elif task == "smile":
        if s["abs_yaw_max"] > a.leak_yaw_for_expr_only:
            leakage_ok = False
            reasons.append("yaw_leak_expr_only")
        if s["jaw_norm_max"] > a.leak_jaw_for_expr_only:
            leakage_ok = False
            reasons.append("jaw_leak_expr_only")
    elif task == "open_mouth":
        if s["abs_yaw_max"] > a.leak_yaw_for_jaw_only:
            leakage_ok = False
            reasons.append("yaw_leak_jaw_only")

    if over_expr:
        reasons.append("over_expr")
    if over_yaw:
        reasons.append("over_yaw")
    if over_jaw:
        reasons.append("over_jaw")

    if not activation:
        reasons.append("activation_failed")

    controlled = activation and dir_purity_ok and leakage_ok and (not over_motion)
    return activation, controlled, reasons, dir_purity_ok, leakage_ok


def bucket(task: str) -> str:
    if task in {"turn_left", "turn_right"}:
        return "direction"
    if task == "smile":
        return "expression"
    if task == "open_mouth":
        return "jaw"
    return "composition"


def evaluate_root(root: Path, motion_key: str, args) -> dict:
    tasks = {}
    act_cnt = {"direction": [0, 0], "expression": [0, 0], "jaw": [0, 0], "composition": [0, 0], "overall": [0, 0]}
    ctl_cnt = {"overall": [0, 0]}
    found = 0
    expr_max, abs_yaw_max, jaw_max = [], [], []
    over_motion_n = 0
    dir_purity_n = 0
    leakage_n = 0

    for task in CANONICAL_TASKS:
        p = find_file(root, task)
        rec = {"task": task, "found": p is not None, "file": str(p) if p else None}
        b = bucket(task)
        act_cnt[b][1] += 1
        act_cnt["overall"][1] += 1
        ctl_cnt["overall"][1] += 1

        if p is None:
            rec.update({"activation_success": False, "controlled_success": False, "failure_reasons": ["missing"]})
        else:
            try:
                m = load_motion(p, motion_key)
                mm = metrics(m)
                rec.update(mm)
                activation, controlled, reasons, dir_ok, leak_ok = eval_task(task, mm, args)
                rec.update({
                    "activation_success": activation,
                    "controlled_success": controlled,
                    "failure_reasons": reasons,
                    "over_expr": mm["expr_norm_max"] > args.max_expr,
                    "over_yaw": mm["abs_yaw_max"] > args.max_abs_yaw,
                    "over_jaw": mm["jaw_norm_max"] > args.max_jaw,
                    "over_motion": (mm["expr_norm_max"] > args.max_expr) or (mm["abs_yaw_max"] > args.max_abs_yaw) or (mm["jaw_norm_max"] > args.max_jaw),
                    "direction_purity_ok": dir_ok,
                    "leakage_ok": leak_ok,
                })
                found += 1
                expr_max.append(mm["expr_norm_max"])
                abs_yaw_max.append(mm["abs_yaw_max"])
                jaw_max.append(mm["jaw_norm_max"])
                if rec["over_motion"]:
                    over_motion_n += 1
                if dir_ok:
                    dir_purity_n += 1
                if leak_ok:
                    leakage_n += 1
            except Exception as e:
                rec.update({"activation_success": False, "controlled_success": False, "failure_reasons": [f"error:{e}"], "error": str(e)})

        if rec["activation_success"]:
            act_cnt[b][0] += 1
            act_cnt["overall"][0] += 1
        if rec["controlled_success"]:
            ctl_cnt["overall"][0] += 1

        tasks[task] = rec

    valid_n = max(1, found)
    return {
        "num_tasks": len(CANONICAL_TASKS),
        "num_found": found,
        "direction_success_rate": act_cnt["direction"][0] / max(1, act_cnt["direction"][1]),
        "expression_success_rate": act_cnt["expression"][0] / max(1, act_cnt["expression"][1]),
        "jaw_success_rate": act_cnt["jaw"][0] / max(1, act_cnt["jaw"][1]),
        "composition_success_rate": act_cnt["composition"][0] / max(1, act_cnt["composition"][1]),
        "overall_success_rate": act_cnt["overall"][0] / max(1, act_cnt["overall"][1]),
        "activation_overall": act_cnt["overall"][0] / max(1, act_cnt["overall"][1]),
        "controlled_overall": ctl_cnt["overall"][0] / max(1, ctl_cnt["overall"][1]),
        "over_motion_rate": over_motion_n / valid_n,
        "direction_purity_rate": dir_purity_n / valid_n,
        "leakage_rate": leakage_n / valid_n,
        "avg_expr_max": float(np.mean(expr_max)) if expr_max else 0.0,
        "avg_abs_yaw_max": float(np.mean(abs_yaw_max)) if abs_yaw_max else 0.0,
        "avg_jaw_max": float(np.mean(jaw_max)) if jaw_max else 0.0,
        "tasks": tasks,
    }


def write_csv_summary(path: Path, all_res: dict[str, dict]) -> None:
    fields = [
        "method", "num_tasks", "num_found", "activation_overall", "controlled_overall", "over_motion_rate",
        "direction_purity_rate", "leakage_rate", "avg_expr_max", "avg_abs_yaw_max", "avg_jaw_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, r in all_res.items():
            row = {k: r.get(k) for k in fields if k != "method"}
            row["method"] = name
            w.writerow(row)


def write_csv_tasks(path: Path, all_res: dict[str, dict]) -> None:
    fields = [
        "task", "method", "found", "activation_success", "controlled_success", "failure_reasons", "yaw_min", "yaw_max",
        "expr_norm_max", "jaw_norm_max", "abs_yaw_max", "over_motion", "direction_purity_ok", "leakage_ok", "file", "error",
    ]
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
                    "activation_success": t.get("activation_success", False),
                    "controlled_success": t.get("controlled_success", False),
                    "failure_reasons": ";".join(t.get("failure_reasons", [])),
                    "yaw_min": t.get("yaw_min"),
                    "yaw_max": t.get("yaw_max"),
                    "expr_norm_max": t.get("expr_norm_max"),
                    "jaw_norm_max": t.get("jaw_norm_max"),
                    "abs_yaw_max": t.get("abs_yaw_max"),
                    "over_motion": t.get("over_motion"),
                    "direction_purity_ok": t.get("direction_purity_ok"),
                    "leakage_ok": t.get("leakage_ok"),
                    "file": t.get("file"),
                    "error": t.get("error", ""),
                })


def write_md(path: Path, all_res: dict[str, dict]) -> None:
    lines = [
        "# Text Motion Root Comparison", "", "## Activation Summary (existing)", "",
        "| method | found | direction | expression | jaw | composition | activation_overall | avg_expr_max | avg_abs_yaw_max | avg_jaw_max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in all_res.items():
        lines.append(
            f"| {name} | {r['num_found']}/{r['num_tasks']} | {r['direction_success_rate']:.3f} | {r['expression_success_rate']:.3f} | {r['jaw_success_rate']:.3f} | {r['composition_success_rate']:.3f} | {r['activation_overall']:.3f} | {r['avg_expr_max']:.4f} | {r['avg_abs_yaw_max']:.4f} | {r['avg_jaw_max']:.4f} |"
        )

    lines += [
        "", "## Controlled Success Summary", "",
        "| method | controlled_overall | over_motion_rate | direction_purity_rate | leakage_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in all_res.items():
        lines.append(
            f"| {name} | {r['controlled_overall']:.3f} | {r['over_motion_rate']:.3f} | {r['direction_purity_rate']:.3f} | {r['leakage_rate']:.3f} |"
        )

    lines += ["", "## Per-task (activation / controlled / reasons)", "", "| task | " + " | ".join(all_res.keys()) + " |", "|---|" + "|".join(["---"] * len(all_res)) + "|"]
    for task in CANONICAL_TASKS:
        cells = []
        for _, r in all_res.items():
            t = r["tasks"][task]
            if not t.get("found"):
                cells.append("❌/❌ missing")
            elif "error" in t:
                cells.append("❌/❌ error")
            else:
                a = "✅" if t.get("activation_success") else "❌"
                c = "✅" if t.get("controlled_success") else "❌"
                rs = ",".join(t.get("failure_reasons", []))
                cells.append(f"{a}/{c} {rs}" if rs else f"{a}/{c}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")

    lines += ["", "## Interpretation", ""]
    if all_res:
        best_act = max(all_res.items(), key=lambda kv: kv[1]["activation_overall"])
        best_ctl = max(all_res.items(), key=lambda kv: kv[1]["controlled_overall"])
        lines.append(f"Best activation is **{best_act[0]}** ({best_act[1]['activation_overall']:.3f}); best controlled success is **{best_ctl[0]}** ({best_ctl[1]['controlled_overall']:.3f}).")
        lines.append("If a method (often NN retrieval) has high activation but low controlled success, it indicates over-amplified or entangled motions (large yaw/expr/jaw or leakage across channels).")
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
    ap.add_argument("--max_expr", type=float, default=6.0)
    ap.add_argument("--max_abs_yaw", type=float, default=0.35)
    ap.add_argument("--max_jaw", type=float, default=0.07)
    ap.add_argument("--leak_expr_for_head_only", type=float, default=1.0)
    ap.add_argument("--leak_jaw_for_head_only", type=float, default=0.02)
    ap.add_argument("--leak_yaw_for_expr_only", type=float, default=0.05)
    ap.add_argument("--leak_jaw_for_expr_only", type=float, default=0.02)
    ap.add_argument("--leak_yaw_for_jaw_only", type=float, default=0.05)
    ap.add_argument("--direction_purity_yaw", type=float, default=0.03)
    args = ap.parse_args()

    roots = parse_roots(args.roots)
    results = {name: evaluate_root(path, args.motion_key, args) for name, path in roots.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = to_json_safe({
        "roots": roots,
        "args": vars(args),
        "results": results,
    })
    print("[Compare] serializing JSON-safe detailed metrics")
    (args.output_dir / "detailed_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv_summary(args.output_dir / "per_method_summary.csv", results)
    write_csv_tasks(args.output_dir / "per_task_comparison.csv", results)
    write_md(args.output_dir / "comparison_report.md", results)

    print(f"[Compare] wrote {args.output_dir / 'detailed_metrics.json'}")
    print(f"[Compare] wrote {args.output_dir / 'per_method_summary.csv'}")
    print(f"[Compare] wrote {args.output_dir / 'per_task_comparison.csv'}")
    print(f"[Compare] wrote {args.output_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
