#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def classify(flame: dict, fastavatar_dir: Path | None, render_issue: str) -> tuple[str, list[str]]:
    metrics = flame.get("geometry_metrics", {})
    motion = flame.get("motion_stats", {})
    corner = float(metrics.get("mouth_corner_displacement_max", 0.0) or 0.0)
    boundary = float(metrics.get("boundary_motion_max", 0.0) or 0.0)
    yaw = max(abs(float(motion.get("yaw_min", 0.0) or 0.0)), abs(float(motion.get("yaw_max", 0.0) or 0.0)))
    has_render = bool(fastavatar_dir and fastavatar_dir.exists() and any(fastavatar_dir.rglob("*")))
    bullets: list[str] = []
    if render_issue:
        bullets.append(f"User/render observation: {render_issue}.")
    if corner < 0.02:
        bullets.append("Smile geometry is weak at the FLAME/proxy layer; prioritize motion generator or smile primitive calibration.")
    elif render_issue and "smile" in render_issue.lower():
        bullets.append("Smile geometry is visible before rendering but weak in FastAvatar; suspect expression response, identity conditioning, or mask/texture suppression.")
    if yaw > 0.12 or boundary > 0.025:
        bullets.append("Yaw/boundary motion is near or beyond conservative range; test lower yaw scale and motion safety smoothing.")
    elif render_issue and "right" in render_issue.lower():
        bullets.append("Yaw looks safe before rendering; right-turn artifact likely belongs to renderer/mask/source-view visibility range.")
    if not has_render:
        bullets.append("No FastAvatar render pack was found; classification is limited to FLAME/proxy and motion statistics.")
    if not bullets:
        bullets.append("No obvious FLAME-level problem detected; compare actual render frames for mask/source-view artifacts.")
    if any("motion generator" in b for b in bullets):
        label = "motion_generator_issue"
    elif any("renderer" in b or "mask" in b or "source-view" in b for b in bullets):
        label = "renderer_or_mask_issue"
    else:
        label = "inconclusive"
    return label, bullets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_npz", type=Path, default=None)
    ap.add_argument("--flame_dir", type=Path, required=True)
    ap.add_argument("--fastavatar_dir", type=Path, default=None)
    ap.add_argument("--output_report", type=Path, required=True)
    ap.add_argument("--render_issue", default="", help="Optional observed issue: right-turn artifact, weak smile, paper-like motion")
    args = ap.parse_args()
    flame = read_json(args.flame_dir / "flame_mesh_metrics.json")
    label, bullets = classify(flame, args.fastavatar_dir, args.render_issue)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# FLAME vs FastAvatar Motion Quality Comparison", "", f"Motion: `{args.motion_npz}`", f"FLAME/proxy directory: `{args.flame_dir}`", f"FastAvatar render directory: `{args.fastavatar_dir}`", "", f"**Layer diagnosis:** `{label}`", "", "## Decision rules", "", "- FLAME mesh normal but FastAvatar render has artifacts → renderer/mask/source-view issue.", "- FLAME mesh action weak/weird → motion generator issue.", "- Smile mesh obvious but render weak → FastAvatar expression response weak.", "- Yaw mesh normal but right-turn render artifact → renderer safe range / mask / visibility issue.", "", "## Findings", ""]
    lines.extend(f"- {b}" for b in bullets)
    lines += ["", "## Raw FLAME/proxy metrics", "", "```json", json.dumps(flame, ensure_ascii=False, indent=2)[:12000], "```", ""]
    args.output_report.write_text("\n".join(lines), encoding="utf-8")
    print(args.output_report)


if __name__ == "__main__":
    main()
