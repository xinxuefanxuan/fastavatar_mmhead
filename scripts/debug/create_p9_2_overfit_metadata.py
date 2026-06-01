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
from typing import Any, Iterable


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected metadata dict, got {type(data).__name__}: {path}")
    return data


def load_config(path: Path) -> Any:
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def resolve_path(value: str | Path, base: Path) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def required_pairs_from_config(config_path: Path | None) -> tuple[int, int, int]:
    if config_path is None:
        return 27, 0, 27
    cfg = load_config(config_path)
    input_frames = int(cfg.dataset.input_frames)
    target_frames = int(cfg.dataset.target_frames)
    return input_frames + target_frames, input_frames, target_frames


def item_required_pairs(frame_data: dict[str, Any], default_input_frames: int, target_frames: int) -> int:
    return int(frame_data.get("input_frames", default_input_frames)) + int(target_frames)


def filter_nersemble(all_frame_groups: dict[str, Any]) -> dict[str, Any]:
    filtered = {}
    for path, frame_data in all_frame_groups.items():
        if path.startswith("nersemble/"):
            filtered[path[len("nersemble/"):]] = frame_data
    return filtered


def extract_id(path: str) -> str:
    return path.split("/")[0] if path else path


def pair_count(frame_data: dict[str, Any]) -> int:
    pairs = frame_data.get("data") or []
    return len(pairs) if isinstance(pairs, list) else 0


def percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def print_pair_distribution(title: str, counts: Iterable[int], min_pairs: int) -> None:
    values = sorted(int(v) for v in counts)
    if not values:
        print(f"[CreateP9.2Meta] {title}: no frame groups")
        return
    mean = sum(values) / len(values)
    ge_min = sum(1 for v in values if v >= min_pairs)
    print(f"[CreateP9.2Meta] {title}: n={len(values)} min={values[0]} max={values[-1]} mean={mean:.2f}")
    print(
        "[CreateP9.2Meta] pair-count percentiles: "
        f"p10={percentile(values, 10):.1f} p25={percentile(values, 25):.1f} "
        f"p50={percentile(values, 50):.1f} p75={percentile(values, 75):.1f} "
        f"p90={percentile(values, 90):.1f}"
    )
    print(f"[CreateP9.2Meta] count >= min_pairs({min_pairs}) = {ge_min}; below = {len(values) - ge_min}")


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


def parse_prefer_ids(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small P9.2 overfit mixed_uids metadata file from existing FastAvatar data.")
    parser.add_argument("--config", type=Path, default=Path("configs/train/fastavatar_motion_zero_token_overfit.yaml"))
    parser.add_argument("--source_meta", type=Path, default=Path("/home/yuanyuhao/FastAvatar/datasets/mixed_uids.json"))
    parser.add_argument("--root_dir", type=Path, default=Path("/home/yuanyuhao/FastAvatar/data/nersemble_fastavatar_unified_full"))
    parser.add_argument("--output", type=Path, default=Path("datasets/p9_2_overfit_mixed_uids.json"))
    parser.add_argument("--min_pairs", default="auto", help="Required camera-frame pairs, or 'auto' to infer input_frames + target_frames from --config.")
    parser.add_argument("--max_ids", type=int, default=6, help="Select at most this many valid IDs. Empty val_id selects 5 for validation, so 6 leaves train samples.")
    parser.add_argument("--num_ids", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max_items_per_id", type=int, default=4)
    parser.add_argument("--prefer_ids", default=None, help="Optional comma-separated IDs to prefer before random selection.")
    parser.add_argument("--max_pairs_to_check", type=int, default=0, help="Validate this many frame pairs per item; <=0 validates every referenced frame pair.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_ids is not None:
        args.max_ids = args.num_ids

    config_path = resolve_path(args.config, Path.cwd()) if args.config else None
    if str(args.min_pairs).lower() == "auto":
        inferred_required_pairs, default_input_frames, target_frames = required_pairs_from_config(config_path)
    else:
        inferred_required_pairs = int(args.min_pairs)
        if config_path and config_path.exists():
            _, default_input_frames, target_frames = required_pairs_from_config(config_path)
        else:
            default_input_frames = max(inferred_required_pairs, 0)
            target_frames = 0

    print(f"[CreateP9.2Meta] config={config_path}")
    print(f"[CreateP9.2Meta] input_frames={default_input_frames}")
    print(f"[CreateP9.2Meta] target_frames={target_frames}")
    print(f"[CreateP9.2Meta] inferred_required_pairs={inferred_required_pairs}")

    if not args.source_meta.exists():
        raise FileNotFoundError(f"source metadata not found: {args.source_meta}")
    if not args.root_dir.exists():
        raise FileNotFoundError(f"processed data root not found: {args.root_dir}")

    all_meta = load_json(args.source_meta)
    filtered = filter_nersemble(all_meta)
    print_pair_distribution("pair-count distribution before filtering", [pair_count(v) for v in filtered.values()], inferred_required_pairs)

    by_id: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    skipped_short = 0
    skipped_missing = 0
    for clean_key, frame_data in filtered.items():
        required_pairs = item_required_pairs(frame_data, default_input_frames, target_frames)
        if pair_count(frame_data) < required_pairs:
            skipped_short += 1
            continue
        if item_has_required_files(args.root_dir, clean_key, frame_data, args.max_pairs_to_check):
            by_id[extract_id(clean_key)].append((clean_key, frame_data))
        else:
            skipped_missing += 1

    eligible_count = sum(len(items) for items in by_id.values())
    if eligible_count == 0:
        raise RuntimeError(
            "No eligible frame groups satisfy min_pairs and file validation. "
            f"required_pairs={inferred_required_pairs}, skipped_short={skipped_short}, skipped_missing={skipped_missing}. "
            "Try lowering input_frames/target_frames in the config or overriding --min_pairs."
        )

    preferred = parse_prefer_ids(args.prefer_ids)
    rng = random.Random(args.seed)
    remaining_ids = sorted(uid for uid, items in by_id.items() if items and uid not in set(preferred))
    rng.shuffle(remaining_ids)
    ordered_ids = [uid for uid in preferred if uid in by_id and by_id[uid]] + remaining_ids
    selected_ids = sorted(ordered_ids[: min(args.max_ids, len(ordered_ids))])

    out: dict[str, Any] = {}
    selected_groups: list[tuple[str, int, int]] = []
    for uid in selected_ids:
        items = list(by_id[uid])
        rng.shuffle(items)
        for clean_key, frame_data in items[: args.max_items_per_id]:
            out[prefixed_key(clean_key)] = frame_data
            selected_groups.append((prefixed_key(clean_key), pair_count(frame_data), item_required_pairs(frame_data, default_input_frames, target_frames)))

    if not out:
        raise RuntimeError(
            "Selected IDs produced no metadata rows. "
            f"Eligible IDs={sorted(by_id)}; try increasing --max_ids or lowering --min_pairs."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[CreateP9.2Meta] source_meta={args.source_meta}")
    print(f"[CreateP9.2Meta] root_dir={args.root_dir}")
    print(f"[CreateP9.2Meta] output={args.output}")
    print(f"[CreateP9.2Meta] min_pairs={args.min_pairs}")
    print(f"[CreateP9.2Meta] effective_required_pairs={inferred_required_pairs}")
    print(f"[CreateP9.2Meta] skipped_short={skipped_short}")
    print(f"[CreateP9.2Meta] skipped_missing_or_invalid={skipped_missing}")
    print(f"[CreateP9.2Meta] eligible frame groups={eligible_count}")
    print(f"[CreateP9.2Meta] selected IDs ({len(selected_ids)}): {selected_ids}")
    print(f"[CreateP9.2Meta] item count={len(out)}")
    print(f"[CreateP9.2Meta] per-ID counts={dict(Counter(extract_id(k[len('nersemble/'):]) for k in out))}")
    print("[CreateP9.2Meta] selected frame groups with pair counts:")
    for key, count, required in selected_groups:
        print(f"  - {key}: pairs={count} required={required}")
    print("[CreateP9.2Meta] No processed data was copied; metadata references the existing root_dir.")


if __name__ == "__main__":
    main()
