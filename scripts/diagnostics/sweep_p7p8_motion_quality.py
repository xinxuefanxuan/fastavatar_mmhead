#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INSPECT = ROOT / "scripts/diagnostics/inspect_motion_npz.py"
RENDER = ROOT / "scripts/diagnostics/render_flame_mesh_sequence.py"

PRIMITIVES = ["turn_left", "turn_right", "smile", "open_mouth", "turn_left_and_smile", "turn_right_and_smile"]
PROMPTS = {
    "turn_left": "turn left",
    "turn_right": "turn right",
    "smile": "smile",
    "open_mouth": "open mouth",
    "turn_left_and_smile": "turn left and smile",
    "turn_right_and_smile": "turn right and smile",
}
YAW_SCALES = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18]
SMILE_GAINS = [0.5, 1.0, 1.5, 2.0, 3.0]


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode


def scale_motion(src: Path, dst: Path, yaw_scale: float | None, smile_gain: float | None) -> None:
    data = np.load(src)
    m = (data["motion"] if "motion" in data else data["motion_raw"]).astype(np.float32).copy()
    if yaw_scale is not None:
        peak = float(np.max(np.abs(m[:, 51]))) if m.size else 0.0
        if peak > 1e-8:
            m[:, 51] *= float(yaw_scale) / peak
    if smile_gain is not None:
        center = np.median(m[:, :50], axis=0, keepdims=True)
        m[:, :50] = center + (m[:, :50] - center) * float(smile_gain)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, motion=m, motion_raw=m, expr_delta=m[:, :50], head_delta=m[:, 50:53], jaw_delta=m[:, 53:56])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/mmhead_debug/p7p8_quality_sweep"))
    ap.add_argument("--generator", type=Path, default=Path("motion_model/generate_from_text_temporal.py"))
    ap.add_argument("--generator_args", nargs=argparse.REMAINDER, default=[])
    ap.add_argument("--skip_generate", action="store_true")
    ap.add_argument("--fastavatar_command_template", default="", help="Optional command template with {motion} and {out_dir}")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for prim in PRIMITIVES:
        base_dir = args.output_dir / prim
        base_npz = base_dir / "base_motion.npz"
        if not args.skip_generate:
            cmd = [sys.executable, str(args.generator), "--prompt", PROMPTS[prim], "--output_npz", str(base_npz)] + args.generator_args
            rc = run(cmd, base_dir / "generate.log")
            if rc != 0:
                rows.append({"primitive": prim, "status": "generate_failed", "log": str(base_dir / "generate.log")})
                continue
        if not base_npz.exists():
            rows.append({"primitive": prim, "status": "missing_base_motion"})
            continue
        variants: list[tuple[str, float | None, float | None]] = []
        if "turn" in prim:
            variants.extend((f"yaw_{s:.2f}", s, None) for s in YAW_SCALES)
        if "smile" in prim:
            variants.extend((f"smile_{g:.1f}", None, g) for g in SMILE_GAINS)
        if not variants:
            variants = [("base", None, None)]
        for name, yaw_scale, smile_gain in variants:
            out_dir = base_dir / name
            motion = out_dir / "motion.npz"
            scale_motion(base_npz, motion, yaw_scale, smile_gain)
            run([sys.executable, str(INSPECT), "--input", str(motion), "--output_dir", str(out_dir / "motion_stats")], out_dir / "inspect.log")
            run([sys.executable, str(RENDER), "--input", str(motion), "--output_dir", str(out_dir / "flame_preview"), "--wireframe", "--save_views"], out_dir / "flame.log")
            render_status = "not_requested"
            if args.fastavatar_command_template:
                cmd_text = args.fastavatar_command_template.format(motion=motion, out_dir=out_dir / "fastavatar")
                render_status = "ok" if run(["bash", "-lc", cmd_text], out_dir / "fastavatar.log") == 0 else "failed"
            rows.append({"primitive": prim, "variant": name, "motion": str(motion), "yaw_scale": yaw_scale, "smile_gain": smile_gain, "fastavatar": render_status})
    lines = ["# P7/P8 Motion Quality Sweep", "", "| primitive | variant | yaw scale | smile gain | motion | FastAvatar |", "|---|---|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r.get('primitive','')} | {r.get('variant', r.get('status',''))} | {r.get('yaw_scale','')} | {r.get('smile_gain','')} | `{r.get('motion','')}` | {r.get('fastavatar','')} |")
    (args.output_dir / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "sweep_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_dir / "sweep_summary.md")


if __name__ == "__main__":
    main()
