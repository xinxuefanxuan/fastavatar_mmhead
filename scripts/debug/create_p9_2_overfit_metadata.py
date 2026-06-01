#!/usr/bin/env python3
"""Create a small project-local FastAvatar metadata JSON for P9.2 overfit runs.

The script reuses the existing processed data root and copies only metadata entries; it never
copies processed images/FLAME files.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected metadata dict, got {type(data).__name__}: {path}")
    return data


def filter_nersemble(all_frame_groups: dict[str, Any]) -> dict[str, Any]:
    filtered = {}
    for path, frame_data in all_frame_groups.items():
        if path.startswith("nersemble/"):
            filtered[path[len("nersemble/"):]] = frame_data
    return filtered


def extract_id(path: str) -> str:
    return path.split("/")[0] if path else path


def candidate_paths(root_dir: Path, clean_key: str, pair: dict[str, Any]) -> list[tuple[Path, Path]]:
    frame_idx = int(pair["frame"])
    camera_id = str(pair["camera"])
    sequence_name = pair.get("seq")
    parts = clean_key.split("/")
    base_path = "/".join(parts[:-1]) if len(parts) > 1 else clean_key
    frame = f"{frame_idx:05d}"
    out = []
    if sequence_name:
        out.append((
            root_dir / base_path / camera_id / str(sequence_name) / "processed_data" / frame,
            root_dir / base_path / camera_id / str(sequence_name) / "flame_param" / f"{frame}.npz",
        ))
    out.append((
        root_dir / base_path / "processed_data" / frame,
        root_dir / base_path / "flame_param" / f"{frame}.npz",
    ))
    out.append((
        root_dir / base_path / camera_id / "processed_data" / frame,
        root_dir / base_path / camera_id / "flame_param" / f"{frame}.npz",
    ))
    return out


def item_has_required_files(root_dir: Path, clean_key: str, frame_data: dict[str, Any], max_pairs_to_check: int) -> bool:
    pairs = frame_data.get("data") or []
    if not pairs:
        return False
    pairs_to_check = pairs if max_pairs_to_check <= 0 else pairs[:max_pairs_to_check]
    for pair in pairs_to_check:
        if "camera" not in pair or "frame" not in pair:
            return False
        if not any(frame_dir.exists() and flame.exists() for frame_dir, flame in candidate_paths(root_dir, clean_key, pair)):
            return False
    return True


def prefixed_key(clean_key: str) -> str:
    return clean_key if clean_key.startswith("nersemble/") else f"nersemble/{clean_key}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small P9.2 overfit mixed_uids metadata file from existing FastAvatar data.")
    parser.add_argument("--source_meta", type=Path, default=Path("/home/yuanyuhao/FastAvatar/datasets/mixed_uids.json"))
    parser.add_argument("--root_dir", type=Path, default=Path("/home/yuanyuhao/FastAvatar/data/nersemble_fastavatar_unified_full"))
    parser.add_argument("--output", type=Path, default=Path("datasets/p9_2_overfit_mixed_uids.json"))
    parser.add_argument("--num_ids", type=int, default=6, help="Select this many valid IDs when available. Empty val_id selects 5 for validation, so 6 leaves train samples.")
    parser.add_argument("--max_items_per_id", type=int, default=4)
    parser.add_argument("--max_pairs_to_check", type=int, default=0, help="Validate this many frame pairs per item; <=0 validates every referenced frame pair.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.source_meta.exists():
        raise FileNotFoundError(f"source metadata not found: {args.source_meta}")
    if not args.root_dir.exists():
        raise FileNotFoundError(f"processed data root not found: {args.root_dir}")

    all_meta = load_json(args.source_meta)
    filtered = filter_nersemble(all_meta)
    by_id: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    skipped_missing = 0
    for clean_key, frame_data in filtered.items():
        if item_has_required_files(args.root_dir, clean_key, frame_data, args.max_pairs_to_check):
            by_id[extract_id(clean_key)].append((clean_key, frame_data))
        else:
            skipped_missing += 1

    ids = sorted(uid for uid, items in by_id.items() if items)
    if not ids:
        raise RuntimeError("No valid nersemble IDs with required processed_data/flame_param files were found.")
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    selected_ids = sorted(ids[: min(args.num_ids, len(ids))])

    out: dict[str, Any] = {}
    for uid in selected_ids:
        items = list(by_id[uid])
        rng.shuffle(items)
        for clean_key, frame_data in items[: args.max_items_per_id]:
            out[prefixed_key(clean_key)] = frame_data

    if not out:
        raise RuntimeError("Selected IDs produced no metadata rows.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[CreateP9.2Meta] source_meta={args.source_meta}")
    print(f"[CreateP9.2Meta] root_dir={args.root_dir}")
    print(f"[CreateP9.2Meta] output={args.output}")
    print(f"[CreateP9.2Meta] skipped_missing_or_invalid={skipped_missing}")
    print(f"[CreateP9.2Meta] selected IDs ({len(selected_ids)}): {selected_ids}")
    print(f"[CreateP9.2Meta] item count={len(out)}")
    print(f"[CreateP9.2Meta] per-ID counts={dict(Counter(extract_id(k[len('nersemble/'):]) for k in out))}")
    print("[CreateP9.2Meta] No processed data was copied; metadata references the existing root_dir.")


if __name__ == "__main__":
    main()
