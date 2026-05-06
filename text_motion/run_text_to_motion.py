#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import yaml
except Exception as e:
    raise SystemExit(f"PyYAML is required: {e}")

from text_motion.motion_primitives import (
    PrimitiveResult,
    neutral,
    turn_head_left,
    turn_head_right,
    nod,
    open_mouth,
    smile_from_exemplar_delta,
    sinusoidal_blink_curve,
)


def copy_template(src: Path, dst: Path, dry_run: bool) -> None:
    if src.resolve() == dst.resolve():
        raise SystemExit("output_motion cannot equal template_motion")
    if dst.exists():
        raise SystemExit(f"output_motion already exists: {dst}")
    print(f"Copy template: {src} -> {dst}")
    if not dry_run:
        shutil.copytree(src, dst)


def load_frame_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_frame_npz(npz_path: Path, payload: Dict[str, np.ndarray], dry_run: bool) -> None:
    if dry_run:
        return
    np.savez(npz_path, **payload)


def detect_motion_paths(root: Path) -> Tuple[Path, Path, list[Path]]:
    transforms = root / "transforms.json"
    if not transforms.exists():
        raise SystemExit(f"transforms.json not found under {root}")

    flame_dir = root / "flame_param"
    if not flame_dir.exists():
        raise SystemExit(f"flame_param directory not found under {root}")

    frame_npz = sorted(flame_dir.glob("*.npz"))
    if not frame_npz:
        raise SystemExit(f"no frame npz found under {flame_dir}")

    return transforms, flame_dir, frame_npz


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def apply_primitive_to_sequence(
    prompt: str,
    config: dict,
    frame_payloads: list[Dict[str, np.ndarray]],
    num_frames: int,
    dry_run: bool,
) -> None:
    mapping = config.get("prompt_to_primitive", {})
    primitive = mapping.get(prompt.strip().lower())
    if not primitive:
        raise SystemExit(f"Unsupported prompt: {prompt}. Allowed: {sorted(mapping.keys())}")

    fields = config.get("field_mapping", {})
    params = config.get("primitive_params", {})

    keys = set(frame_payloads[0].keys())
    print(f"Detected frame npz keys: {sorted(keys)}")

    def require_field(field_name: str) -> str:
        k = fields.get(field_name)
        if not k or k not in keys:
            raise SystemExit(f"Required field '{field_name}' -> '{k}' not found in frame npz keys")
        return k

    seq: Dict[str, np.ndarray] = {}
    for k in keys:
        seq[k] = np.stack([p[k] for p in frame_payloads], axis=0)

    n = min(num_frames, len(frame_payloads))
    result: PrimitiveResult

    if primitive == "neutral":
        result = neutral(seq, n)
    elif primitive == "turn_head_left":
        pf = require_field("head_pose_field")
        p = params.get("turn_head_left", {})
        result = turn_head_left(seq, n, pose_field=pf, yaw_index=int(p.get("yaw_index", 1)), amplitude=float(p.get("amplitude", 0.2)))
    elif primitive == "turn_head_right":
        pf = require_field("head_pose_field")
        p = params.get("turn_head_right", {})
        result = turn_head_right(seq, n, pose_field=pf, yaw_index=int(p.get("yaw_index", 1)), amplitude=float(p.get("amplitude", 0.2)))
    elif primitive == "nod":
        pf = require_field("head_pose_field")
        p = params.get("nod", {})
        result = nod(seq, n, pose_field=pf, pitch_index=int(p.get("pitch_index", 0)), amplitude=float(p.get("amplitude", 0.15)), cycles=float(p.get("cycles", 1.5)))
    elif primitive == "open_mouth":
        jf = require_field("jaw_pose_field")
        p = params.get("open_mouth", {})
        result = open_mouth(seq, n, jaw_field=jf, jaw_index=int(p.get("jaw_index", 0)), amplitude=float(p.get("amplitude", 0.25)))
    elif primitive == "smile_from_exemplar_delta":
        ef = require_field("expression_field")
        exemplar = config.get("smile_exemplar", {})
        key = "smile_strongly" if "strong" in prompt.lower() else "smile_slightly"
        alpha = float(params.get(key, {}).get("alpha", 0.5))
        result = smile_from_exemplar_delta(
            seq, n, expr_field=ef,
            neutral_slice=slice(int(exemplar.get("neutral_start", 0)), int(exemplar.get("neutral_end", 10))),
            smile_slice=slice(int(exemplar.get("smile_start", 30)), int(exemplar.get("smile_end", 45))),
            alpha=alpha,
        )
    elif primitive == "blink_once":
        ef = fields.get("expression_field")
        jf = fields.get("jaw_pose_field")
        if ef in keys:
            arr = seq[ef].copy()
            p = params.get("blink_once", {})
            curve = sinusoidal_blink_curve(n, width=int(p.get("width", 8)), amplitude=float(p.get("amplitude", 0.6)))
            arr[:n, 0] += curve  # only configurable index-free fallback; semantics unknown
            result = PrimitiveResult({ef: arr}, f"blink_once fallback applied to {ef}[:,0]")
        elif jf in keys:
            arr = seq[jf].copy()
            p = params.get("blink_once", {})
            curve = sinusoidal_blink_curve(n, width=int(p.get("width", 8)), amplitude=float(p.get("amplitude", 0.6)))
            arr[:n, 0] += curve
            result = PrimitiveResult({jf: arr}, f"blink_once fallback applied to {jf}[:,0]")
        else:
            raise SystemExit("blink_once requested but neither expression_field nor jaw_pose_field exists")
    else:
        raise SystemExit(f"Unknown primitive: {primitive}")

    print(f"Primitive: {primitive}; {result.info}")
    for k, v in result.updates.items():
        seq[k] = v
        print(f"  will modify field={k}, shape={v.shape}, dtype={v.dtype}")

    if dry_run:
        print("[dry_run] no files written")
        return

    for i, payload in enumerate(frame_payloads):
        for k, arr in result.updates.items():
            payload[k] = arr[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--template_motion", required=True, type=Path)
    ap.add_argument("--output_motion", required=True, type=Path)
    ap.add_argument("--num_frames", type=int, default=90)
    ap.add_argument("--primitive_config", type=Path, default=Path("text_motion/primitives.yaml"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    copy_template(args.template_motion, args.output_motion, args.dry_run)

    work_root = args.template_motion if args.dry_run else args.output_motion
    _, flame_dir, frame_npz_paths = detect_motion_paths(work_root)

    cfg = load_config(args.primitive_config)

    frame_payloads = [load_frame_npz(p) for p in frame_npz_paths]
    apply_primitive_to_sequence(args.prompt, cfg, frame_payloads, args.num_frames, args.dry_run)

    if not args.dry_run:
        for p, payload in zip(frame_npz_paths, frame_payloads):
            save_frame_npz(p, payload, dry_run=False)

    print("Done.")


if __name__ == "__main__":
    main()
