#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

RELATED = ("flame", "pose", "expr", "camera", "head", "jaw", "eye", "neck", "rotation", "transform")


def print_tree(root: Path, max_depth: int = 3) -> None:
    print(f"[Tree] root={root}")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if len(rel.parts) <= max_depth:
            prefix = "DIR " if p.is_dir() else "FILE"
            print(f"  {prefix:4} {rel}")


def np_stats(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty"
    if np.issubdtype(arr.dtype, np.number):
        return f"min={arr.min():.6g} max={arr.max():.6g} mean={arr.mean():.6g}"
    return "non-numeric"


def find_first_frame_npz(motion_root: Path) -> list[Path]:
    flame_dir = motion_root / "flame_param"
    if flame_dir.is_dir():
        files = sorted(flame_dir.glob("*.npz"))
        if files:
            return [files[0]]
    files = sorted(motion_root.rglob("*.npz"))
    return files[:1]


def inspect_npz(npz_path: Path) -> None:
    print(f"\n[NPZ] {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    for k in data.files:
        arr = data[k]
        print(f"  key={k} shape={arr.shape} dtype={arr.dtype} {np_stats(arr)}")


def inspect_json_schema(json_path: Path) -> None:
    print(f"\n[JSON] {json_path}")
    obj = json.loads(json_path.read_text())
    if isinstance(obj, dict):
        print("  top-level keys:", sorted(obj.keys()))
        frames = obj.get("frames")
        if isinstance(frames, list):
            print(f"  number_of_frames={len(frames)}")
            if frames:
                f0 = frames[0]
                if isinstance(f0, dict):
                    print("  frame[0] keys:", sorted(f0.keys()))
                    for k, v in f0.items():
                        if any(x in k.lower() for x in RELATED):
                            print(f"    related_field: {k} -> type={type(v).__name__}")


def related_fields_in_npz(npz_files: Iterable[Path]) -> None:
    print("\n[Related fields scan]")
    found = False
    for p in npz_files:
        d = np.load(p, allow_pickle=True)
        related = [k for k in d.files if any(x in k.lower() for x in RELATED)]
        if related:
            found = True
            print(f"  {p}: {related}")
    if not found:
        print("  no camera/flame/head-pose-related keys found in npz files")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", required=True, type=Path)
    args = ap.parse_args()

    motion_dir = args.motion_dir
    if not motion_dir.exists():
        raise SystemExit(f"motion_dir not found: {motion_dir}")

    print_tree(motion_dir, max_depth=3)

    npz_all = sorted(motion_dir.rglob("*.npz"))
    first_frame_npz = find_first_frame_npz(motion_dir)
    print(f"\nFirst-frame npz candidates: {[str(p) for p in first_frame_npz]}")
    for p in first_frame_npz:
        inspect_npz(p)

    transforms = motion_dir / "transforms.json"
    if transforms.exists():
        inspect_json_schema(transforms)
    else:
        # also search nested transforms
        nested = sorted(motion_dir.rglob("transforms.json"))
        if nested:
            inspect_json_schema(nested[0])
        else:
            print("\n[JSON] transforms.json not found")

    # number of frames by flame_param files if available
    flame_dir = motion_dir / "flame_param"
    if flame_dir.is_dir():
        n_frames = len(list(flame_dir.glob("*.npz")))
        print(f"\nframe_count_by_flame_param={n_frames}")

    related_fields_in_npz(npz_all)


if __name__ == "__main__":
    main()
