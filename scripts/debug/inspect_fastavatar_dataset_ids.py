#!/usr/bin/env python3
"""Inspect FastAvatar train config dataset IDs before launching P9.2 smoke training."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def load_config(path: Path) -> Any:
    try:
        from omegaconf import OmegaConf
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("OmegaConf is required to read FastAvatar yaml configs") from exc
    return OmegaConf.load(path)


def resolve_path(value: str | Path, base: Path) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def required_pairs_from_config(cfg: Any) -> tuple[int, int, int]:
    input_frames = int(cfg.dataset.input_frames)
    target_frames = int(cfg.dataset.target_frames)
    return input_frames + target_frames, input_frames, target_frames


def item_required_pairs(frame_data: dict[str, Any], default_input_frames: int, target_frames: int) -> int:
    return int(frame_data.get("input_frames", default_input_frames)) + int(target_frames)


def load_meta(meta_path: Path) -> dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected metadata JSON object/dict, got {type(data).__name__}: {meta_path}")
    return data


def filter_for_dataset(all_frame_groups: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    """Mirror MixerDataset._filter_data_for_subset for inspection without importing torch."""
    filtered: dict[str, Any] = {}
    prefix = f"{dataset_name}/"
    for path, frame_data in all_frame_groups.items():
        if path.startswith(prefix):
            filtered[path[len(prefix):]] = frame_data
            continue
        if "/" in path and path.split("/")[0] not in ["nersemble", "vfhq"]:
            if dataset_name == "vfhq":
                parts = path.split("/")
                if (len(parts) >= 2 and "EXP-" in parts[1]) or (len(parts) == 2 and parts[0].isdigit() and "EXP-" in path):
                    filtered[path] = frame_data
            # Nersemble legacy fallback intentionally matches current MixerDataset: no-op.
    return filtered


def extract_nersemble_id(path: str) -> str:
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


def pair_stats(values: Iterable[int], min_pairs: int) -> dict[str, Any]:
    counts = sorted(int(v) for v in values)
    if not counts:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "below_min_pairs": 0,
            "ge_min_pairs": 0,
        }
    return {
        "count": len(counts),
        "min": counts[0],
        "max": counts[-1],
        "mean": sum(counts) / len(counts),
        "p10": percentile(counts, 10),
        "p25": percentile(counts, 25),
        "p50": percentile(counts, 50),
        "p75": percentile(counts, 75),
        "p90": percentile(counts, 90),
        "below_min_pairs": sum(1 for v in counts if v < min_pairs),
        "ge_min_pairs": sum(1 for v in counts if v >= min_pairs),
    }


def selected_val_ids(all_ids: list[str], configured_val_id: list[str] | None) -> tuple[list[str], list[str]]:
    configured_val_id = configured_val_id or []
    if not configured_val_id:
        if len(all_ids) < 5:
            return list(all_ids), []
        rng = random.Random(42)
        return rng.sample(list(all_ids), 5), []
    valid = [uid for uid in configured_val_id if uid in set(all_ids)]
    invalid = [uid for uid in configured_val_id if uid not in set(all_ids)]
    if not valid:
        # Mirrors NersembleDataset fallback that makes all IDs validation and therefore can make train empty.
        return list(all_ids), invalid
    return valid, invalid


def infer_counts(filtered: dict[str, Any], configured_val_id: list[str] | None) -> dict[str, Any]:
    id_to_count = Counter(extract_nersemble_id(path) for path in filtered)
    all_ids = sorted(id_to_count)
    val_ids, invalid = selected_val_ids(all_ids, configured_val_id)
    val_set = set(val_ids)
    train_items = {k: v for k, v in filtered.items() if extract_nersemble_id(k) not in val_set}
    val_items = {k: v for k, v in filtered.items() if extract_nersemble_id(k) in val_set}
    return {
        "available_ids": all_ids,
        "id_to_count": dict(id_to_count),
        "valid_val_id": val_ids,
        "invalid_val_id": invalid,
        "expected_train_count": len(train_items),
        "expected_val_candidate_count": len(val_items),
        "train_items": train_items,
        "val_items": val_items,
    }


def print_pair_stats(prefix: str, stats: dict[str, Any], min_pairs: int) -> None:
    print(
        f"[InspectDataset] {prefix} pair-count stats: "
        f"n={stats['count']} min={stats['min']} max={stats['max']} mean={stats['mean']:.2f} "
        f"p10={stats['p10']:.1f} p25={stats['p25']:.1f} p50={stats['p50']:.1f} "
        f"p75={stats['p75']:.1f} p90={stats['p90']:.1f}"
    )
    print(f"[InspectDataset] {prefix} count below {min_pairs}={stats['below_min_pairs']} count >= {min_pairs}={stats['ge_min_pairs']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect FastAvatar dataset IDs and train/val split for a config.")
    parser.add_argument("--config", type=Path, default=Path("configs/train/fastavatar_motion_zero_token_overfit.yaml"))
    parser.add_argument("--dataset", default="nersemble")
    parser.add_argument("--min_pairs", default="auto", help="Required camera-frame pairs, or auto to infer input_frames + target_frames from config.")
    parser.add_argument("--require_train", action="store_true", help="Exit non-zero if expected train count is zero, metadata is missing, or train frame groups are too short.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    cfg_path = resolve_path(args.config, repo_root)
    print(f"[InspectDataset] config={cfg_path}")
    cfg = load_config(cfg_path)
    inferred_required_pairs, default_input_frames, target_frames = required_pairs_from_config(cfg)
    if str(args.min_pairs).lower() == "auto":
        min_pairs = inferred_required_pairs
    else:
        min_pairs = int(args.min_pairs)

    meta_path = resolve_path(cfg.dataset.meta_path, repo_root)
    dataset_cfg = cfg.dataset.datasets[args.dataset]
    root_dir = resolve_path(dataset_cfg.root_dir, repo_root)
    configured_val_id = list(getattr(dataset_cfg, "val_id", []) or [])

    print(f"[InspectDataset] resolved meta_path={meta_path}")
    print(f"[InspectDataset] meta_path exists={meta_path.exists()}")
    print(f"[InspectDataset] resolved {args.dataset} root_dir={root_dir}")
    print(f"[InspectDataset] root_dir exists={root_dir.exists()}")
    print(f"[InspectDataset] configured val_id={configured_val_id}")
    print(f"[InspectDataset] input_frames={default_input_frames}")
    print(f"[InspectDataset] target_frames={target_frames}")
    print(f"[InspectDataset] inferred_required_pairs={inferred_required_pairs}")
    print(f"[InspectDataset] min_pairs={args.min_pairs} effective_min_pairs={min_pairs}")

    if not meta_path.exists():
        print(f"[InspectDataset][ERROR] metadata does not exist: {meta_path}")
        return 2 if args.require_train else 0

    all_meta = load_meta(meta_path)
    filtered = filter_for_dataset(all_meta, args.dataset)
    counts = infer_counts(filtered, configured_val_id)

    print(f"[InspectDataset] loaded frame groups/items total={len(all_meta)}")
    print(f"[InspectDataset] filtered frame groups/items for {args.dataset}={len(filtered)}")
    print_pair_stats("all", pair_stats((pair_count(v) for v in filtered.values()), min_pairs), min_pairs)
    print_pair_stats("train", pair_stats((pair_count(v) for v in counts["train_items"].values()), min_pairs), min_pairs)
    print(f"[InspectDataset] available IDs ({len(counts['available_ids'])}): {counts['available_ids'][:50]}")
    if len(counts["available_ids"]) > 50:
        print(f"[InspectDataset] ... {len(counts['available_ids']) - 50} more IDs omitted")
    print(f"[InspectDataset] valid_val_id={counts['valid_val_id']}")
    print(f"[InspectDataset] invalid_val_id={counts['invalid_val_id']}")
    print(f"[InspectDataset] expected train count={counts['expected_train_count']}")
    print(f"[InspectDataset] expected val candidate count={counts['expected_val_candidate_count']}")
    print(f"[InspectDataset] top ID counts={Counter(counts['id_to_count']).most_common(10)}")

    print("[InspectDataset] train item pair checks:")
    offending = []
    for key, value in counts["train_items"].items():
        raw_count = pair_count(value)
        required = item_required_pairs(value, default_input_frames, target_frames) if str(args.min_pairs).lower() == "auto" else min_pairs
        ok = raw_count >= required
        uid = extract_nersemble_id(key)
        status = "pass" if ok else "fail"
        print(f"  - key={key} uid={uid} raw_pair_count={raw_count} effective_pair_count={required} {status}")
        if not ok:
            offending.append((key, raw_count, required))
    if offending:
        print(f"[InspectDataset][ERROR] Train frame groups below required pair count:")
        for key, count, required in offending[:50]:
            print(f"  - {key}: pairs={count} required={required}")
        if len(offending) > 50:
            print(f"  ... {len(offending) - 50} more offending groups omitted")

    if args.require_train and counts["expected_train_count"] <= 0:
        print("[InspectDataset][ERROR] expected train count is 0. Create overfit metadata or choose a valid val_id split first.")
        return 3
    if args.require_train and offending:
        print("[InspectDataset][ERROR] Some train groups are too short for the configured input_frames + target_frames.")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
