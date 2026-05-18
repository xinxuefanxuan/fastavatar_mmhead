#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def copy_template(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        raise SystemExit("output_motion cannot equal template_motion")
    if dst.exists():
        raise SystemExit(f"output_motion already exists: {dst}")
    print(f"[Copy template]\n  from: {src}\n  to:   {dst}")
    shutil.copytree(src, dst)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    np.savez(path, **payload)


def to_vec(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    return a.reshape(-1)


def detect_flame_frames(root: Path) -> List[Path]:
    flame_dir = root / "flame_param"
    if not flame_dir.exists():
        raise SystemExit(f"flame_param directory not found under {root}")
    frames = sorted(flame_dir.glob("*.npz"))
    if not frames:
        raise SystemExit(f"no frame npz found under {flame_dir}")
    return frames


def compute_neutral_score(payload: Dict[str, np.ndarray]) -> float:
    score = 0.0
    if "expr" in payload:
        score += float(np.linalg.norm(to_vec(payload["expr"]), ord=2))
    if "jaw_pose" in payload:
        score += float(np.linalg.norm(to_vec(payload["jaw_pose"]), ord=2))
    if "rotation" in payload:
        score += 0.3 * float(np.linalg.norm(to_vec(payload["rotation"]), ord=2))
    return score


def choose_reference_frame(frame_payloads: List[Dict[str, np.ndarray]], reference_frame: int, auto_neutral: bool) -> int:
    n = len(frame_payloads)
    if auto_neutral:
        scores = [compute_neutral_score(p) for p in frame_payloads]
        idx = int(np.argmin(np.asarray(scores, dtype=np.float32)))
        return idx
    if reference_frame < 0 or reference_frame >= n:
        raise SystemExit(f"reference_frame out of range: {reference_frame}, num_frames={n}")
    return int(reference_frame)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a neutral static FastAvatar motion template from an existing template motion.")
    ap.add_argument("--template_motion", required=True, type=Path)
    ap.add_argument("--output_motion", required=True, type=Path)
    ap.add_argument("--reference_frame", type=int, default=0)
    ap.add_argument("--auto_neutral", action="store_true")
    args = ap.parse_args()

    if not args.template_motion.exists():
        raise SystemExit(f"template_motion not found: {args.template_motion}")

    copy_template(args.template_motion, args.output_motion)

    frame_paths = detect_flame_frames(args.output_motion)
    frame_payloads = [load_npz(p) for p in frame_paths]

    ref_idx = choose_reference_frame(frame_payloads, args.reference_frame, args.auto_neutral)
    ref = frame_payloads[ref_idx]

    keys_to_apply = ["expr", "jaw_pose", "rotation", "neck_pose", "translation", "eyes_pose", "shape"]
    modified_keys = set()
    skipped_keys = set()

    for payload in frame_payloads:
        for k in keys_to_apply:
            if k in ref and k in payload:
                payload[k] = np.asarray(ref[k]).copy()
                modified_keys.add(k)
            elif k not in payload:
                skipped_keys.add(k)

    for p, payload in zip(frame_paths, frame_payloads):
        save_npz(p, payload)

    print(f"[Neutral template] num_frames={len(frame_paths)}")
    print(f"[Neutral template] chosen reference frame={ref_idx}")
    print(f"[Neutral template] modified keys={sorted(modified_keys)}")
    print(f"[Neutral template] skipped keys={sorted(skipped_keys)}")

    for k in sorted(modified_keys):
        refv = np.asarray(ref[k]).astype(np.float32)
        max_diff = 0.0
        for payload in frame_payloads:
            cur = np.asarray(payload[k]).astype(np.float32)
            max_diff = max(max_diff, float(np.max(np.abs(cur - refv))))
        print(f"[Neutral template] max_diff {k}={max_diff:.8f}")

    print(f"[Neutral template] output={args.output_motion}")


if __name__ == "__main__":
    main()
