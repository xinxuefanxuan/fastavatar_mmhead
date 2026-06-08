#!/usr/bin/env python3
"""Inspect MotionTokenAdapter sensitivity to correct, wrong, and zero tokens.

This is a no-grad diagnostic: it reuses the same dataset/batch construction as
`eval_p9_2_abc_same_batch.py`, builds three motion-token inputs per batch, and
passes only those tokens through the trained MotionTokenAdapter.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.debug.eval_p9_2_abc_same_batch import (  # noqa: E402
    build_flame_dicts,
    build_loader,
    choose_val_id,
    extract_batch_identity,
    evaluated_subject_uids_from_identities,
    filter_metadata_by_ids,
    generate_metadata,
    json_safe,
    load_yaml,
    mean_std,
    move_to_device,
    parse_id_list,
    resolve_local_paths,
    resolve_path,
    select_shuffled_token_donors,
    write_resolved_config,
    write_runtime_config_with_meta,
)


def load_trained_model(cfg: Any, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from FastAvatar.models.modeling_FastAvatar import ModelFastAvatar

    cfg = cfg.copy() if hasattr(cfg, "copy") else cfg
    cfg.model.use_motion_token = True
    cfg.model.motion_token_source = "frame_flame_gt"
    cfg.model.zero_flame_motion = True
    cfg.model.debug_skip_renderer = False
    cfg.model.debug_latent_smoke_loss = False
    cfg.model.debug_max_query_points = None
    model = ModelFastAvatar(**cfg.model)
    if not checkpoint.exists():
        raise FileNotFoundError(f"P9.2 checkpoint does not exist: {checkpoint}")
    import safetensors.torch

    missing, unexpected = safetensors.torch.load_model(model, str(checkpoint), strict=False)
    print(
        f"[P9.2ADAPTER-SENS] loaded checkpoint={checkpoint} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    model.to(device)
    model.eval()
    return model


def token_norm(token: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(token.detach().float().reshape(token.shape[0], -1), ord=2, dim=1).mean().cpu())


def output_norm(output: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(output.detach().float().reshape(output.shape[0], -1), ord=2, dim=1).mean().cpu())


def cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.detach().float().reshape(left.shape[0], -1)
    right_flat = right.detach().float().reshape(right.shape[0], -1)
    return float(torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=1, eps=1e-8).mean().cpu())


def l2_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    diff = left.detach().float() - right.detach().float()
    return float(torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), ord=2, dim=1).mean().cpu())


def match_token_batch_size(token: torch.Tensor, batch_size: int) -> torch.Tensor:
    if token.shape[0] == batch_size:
        return token
    if token.shape[0] == 1:
        return token.expand(batch_size, -1).contiguous()
    if token.shape[0] > batch_size:
        return token[:batch_size].contiguous()
    repeats = (batch_size + token.shape[0] - 1) // token.shape[0]
    return token.repeat(repeats, 1)[:batch_size].contiguous()


def build_tokens_for_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    donor_batch: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    batch = move_to_device(batch, device)
    donor_batch = move_to_device(donor_batch, device)
    _, target_flame = build_flame_dicts(batch)
    _, donor_target_flame = build_flame_dicts(donor_batch)

    correct = model.build_motion_token_input_from_flame(target_flame)
    wrong = model.build_motion_token_input_from_flame(donor_target_flame)
    wrong = match_token_batch_size(wrong, correct.shape[0]).to(device=device, dtype=correct.dtype)
    zero = torch.zeros_like(correct)
    return {"correct": correct, "wrong": wrong, "zero": zero}, correct.shape[0]


def adapter_metrics_for_batch(
    model: torch.nn.Module,
    batch_idx: int,
    batch: dict[str, Any],
    donor_idx: int,
    donor_batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
        tokens, batch_size = build_tokens_for_batch(model, batch, donor_batch, device)
        outputs = {name: model.motion_token_adapter(token) for name, token in tokens.items()}

    row = {
        "batch_idx": batch_idx,
        "token_source_batch_idx": donor_idx,
        "batch_size": batch_size,
        "correct_input_norm": token_norm(tokens["correct"]),
        "wrong_input_norm": token_norm(tokens["wrong"]),
        "zero_input_norm": token_norm(tokens["zero"]),
        "correct_output_norm": output_norm(outputs["correct"]),
        "wrong_output_norm": output_norm(outputs["wrong"]),
        "zero_output_norm": output_norm(outputs["zero"]),
        "cosine_correct_wrong": cosine_mean(outputs["correct"], outputs["wrong"]),
        "cosine_correct_zero": cosine_mean(outputs["correct"], outputs["zero"]),
        "l2_correct_wrong": l2_mean(outputs["correct"], outputs["wrong"]),
        "l2_correct_zero": l2_mean(outputs["correct"], outputs["zero"]),
    }
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "correct_input_norm",
        "wrong_input_norm",
        "zero_input_norm",
        "correct_output_norm",
        "wrong_output_norm",
        "zero_output_norm",
        "cosine_correct_wrong",
        "cosine_correct_zero",
        "l2_correct_wrong",
        "l2_correct_zero",
    ]
    return {key: mean_std([row.get(key) for row in rows]) for key in metric_keys}


def write_per_batch_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "per_batch.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_idx",
        "token_source_batch_idx",
        "batch_size",
        "subject_uid",
        "metadata_key",
        "frame_id",
        "camera_ids",
        "correct_input_norm",
        "wrong_input_norm",
        "zero_input_norm",
        "correct_output_norm",
        "wrong_output_norm",
        "zero_output_norm",
        "cosine_correct_wrong",
        "cosine_correct_zero",
        "l2_correct_wrong",
        "l2_correct_zero",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(json_safe(row.get(key)), ensure_ascii=False) for key in fieldnames})
    return path


def write_report(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate = metrics.get("aggregate", {})

    def fmt(value: float | None, digits: int = 6) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"

    lines = [
        "# P9.2 MotionTokenAdapter Token Sensitivity",
        "",
        "## Run Selection",
        f"* checkpoint: `{metrics.get('checkpoint')}`",
        f"* base_config: `{metrics.get('base_config')}`",
        f"* split: `{metrics.get('split')}`",
        f"* eval_protocol: `{metrics.get('eval_protocol')}`",
        f"* train_ids: `{metrics.get('train_ids')}`",
        f"* holdout_ids: `{metrics.get('holdout_ids')}`",
        f"* sacrificial_val_id: `{metrics.get('sacrificial_val_id')}`",
        f"* evaluated_subject_uids: `{metrics.get('evaluated_subject_uids')}`",
        f"* evaluated_batch_indices: `{metrics.get('evaluated_batch_indices')}`",
        "",
        "## Aggregate Metrics",
        "| metric | mean | std |",
        "|---|---:|---:|",
    ]
    for key, stats in aggregate.items():
        lines.append(f"| {key} | {fmt(stats.get('mean'))} | {fmt(stats.get('std'))} |")

    lines += [
        "",
        "## Interpretation",
        "* If cosine(correct_out, wrong_out) and cosine(correct_out, zero_out) are close to 1 while L2 distances are near 0, the adapter may ignore token content.",
        "* If output cosines decrease and L2 distances increase, the adapter responds to motion-token content; renderer/loss behavior should then be diagnosed separately.",
        "",
        "## Per-Batch Metrics",
        "| batch_idx | donor_batch | subject_uid | cos C/W | cos C/Z | L2 C/W | L2 C/Z | correct_out_norm | wrong_out_norm | zero_out_norm |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.get("per_batch", []):
        lines.append(
            f"| {row.get('batch_idx')} | {row.get('token_source_batch_idx')} | {row.get('subject_uid')} | "
            f"{fmt(row.get('cosine_correct_wrong'))} | {fmt(row.get('cosine_correct_zero'))} | "
            f"{fmt(row.get('l2_correct_wrong'))} | {fmt(row.get('l2_correct_zero'))} | "
            f"{fmt(row.get('correct_output_norm'))} | {fmt(row.get('wrong_output_norm'))} | {fmt(row.get('zero_output_norm'))} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect trained MotionTokenAdapter sensitivity to correct/wrong/zero tokens.")
    parser.add_argument("--base_config", type=Path, default=Path("configs/train/fastavatar_motion_zero_token_overfit_micro_render.yaml"))
    parser.add_argument("--local_paths_config", type=Path, default=Path("configs/local/p9_2_local_paths.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("exps/checkpoints/fastavatar/fastavatar_motion_zero_token_overfit_micro_render_500step/000500/model.safetensors"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/mmhead_debug/p9_2_adapter_sensitivity"))
    parser.add_argument("--split", choices=("train", "holdout"), default="holdout")
    parser.add_argument("--train_ids", default="030,037,038,069,070")
    parser.add_argument("--holdout_ids", default="083")
    parser.add_argument("--sacrificial_val_id", default=None)
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_eval_loader(args: argparse.Namespace, base_cfg: Any, paths: dict[str, Path | str], output_dir: Path) -> tuple[Any, Any, str, list[str], str, list[str], dict[str, int]]:
    train_ids = parse_id_list(args.train_ids)
    holdout_ids = parse_id_list(args.holdout_ids)
    overlap = sorted(set(train_ids) & set(holdout_ids))
    if args.split == "holdout" and overlap:
        raise RuntimeError(f"holdout_ids must be disjoint from train_ids; overlap={overlap}")

    if args.split == "holdout":
        sacrificial_val_id = str(args.sacrificial_val_id or "")
        if not sacrificial_val_id:
            candidates = [uid for uid in train_ids if uid not in set(holdout_ids)]
            if not candidates:
                raise RuntimeError("Could not choose sacrificial_val_id from train_ids")
            sacrificial_val_id = candidates[0]
        metadata_requested_ids = list(dict.fromkeys(holdout_ids + [sacrificial_val_id]))
        prefer_ids = train_ids + [uid for uid in metadata_requested_ids if uid not in set(train_ids)]
        generate_metadata(args.base_config, paths, prefer_ids=prefer_ids, max_ids=max(6, len(prefer_ids)))
        generated_meta = Path(paths["generated_meta"])
        holdout_meta = output_dir / "runtime_configs" / "adapter_sensitivity_holdout_mixed_uids.json"
        holdout_meta, selected_ids, holdout_item_count = filter_metadata_by_ids(generated_meta, holdout_meta, metadata_requested_ids)
        runtime_config = write_runtime_config_with_meta(
            args.base_config,
            base_cfg,
            Path(paths["root_dir"]),
            holdout_meta,
            val_ids=[sacrificial_val_id],
            output_dir=output_dir,
            suffix="adapter_sensitivity_holdout_resolved",
        )
        cfg = load_yaml(runtime_config)
        loader = build_loader(cfg, "train")
        if holdout_item_count <= 0 or len(loader.dataset) <= 0:
            raise RuntimeError(
                f"Holdout adapter sensitivity dataset is empty: holdout_item_count={holdout_item_count}, "
                f"dataset_len={len(loader.dataset)}, holdout_ids={holdout_ids}, sacrificial_val_id={sacrificial_val_id}"
            )
        split_counts = {"holdout": len(loader.dataset), "train_style_loader": len(loader.dataset)}
        return cfg, loader, "holdout_trainstyle_sacrificial_val", selected_ids, sacrificial_val_id, holdout_ids, split_counts

    generate_metadata(args.base_config, paths)
    val_id, selected_ids = choose_val_id(Path(paths["generated_meta"]), str(paths["preferred_val_id"]))
    runtime_config = write_resolved_config(args.base_config, base_cfg, paths, val_id, output_dir)
    cfg = load_yaml(runtime_config)
    loader = build_loader(cfg, "train")
    if len(loader.dataset) <= 0:
        raise RuntimeError("Train split produced zero samples for adapter sensitivity inspection.")
    split_counts = {"train": len(loader.dataset)}
    return cfg, loader, "train", selected_ids, "", holdout_ids, split_counts


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    base_config = resolve_path(args.base_config, repo)
    local_paths_config = resolve_path(args.local_paths_config, repo)
    checkpoint = resolve_path(args.checkpoint, repo)
    output_dir = resolve_path(args.output_dir, repo)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    args.base_config = base_config
    base_cfg = load_yaml(base_config)
    paths = resolve_local_paths(base_cfg, repo, local_paths_config)
    cfg, loader, eval_protocol, selected_ids, sacrificial_val_id, holdout_ids, split_counts = build_eval_loader(args, base_cfg, paths, output_dir)

    selected_batches: list[tuple[int, dict[str, Any]]] = []
    stop_after = args.batch_idx + args.num_batches
    for idx, item in enumerate(loader):
        if idx < args.batch_idx:
            continue
        if idx >= stop_after:
            break
        selected_batches.append((idx, item))
    if not selected_batches:
        raise RuntimeError(
            f"Could not fetch batches for batch_idx={args.batch_idx}, num_batches={args.num_batches}; "
            f"dataset_len={len(loader.dataset)}"
        )
    donor_map = select_shuffled_token_donors(loader, selected_batches)
    model = load_trained_model(cfg, checkpoint, device)

    known_subject_ids = set(parse_id_list(args.train_ids)) | set(holdout_ids) | set(selected_ids)
    identities = [extract_batch_identity(batch, args.split, eval_protocol, idx, known_subject_ids) for idx, batch in selected_batches]
    evaluated_subject_uids = evaluated_subject_uids_from_identities(identities)

    rows: list[dict[str, Any]] = []
    for batch_idx, batch in selected_batches:
        donor_idx, donor_batch = donor_map[batch_idx]
        row = adapter_metrics_for_batch(model, batch_idx, batch, donor_idx, donor_batch, device)
        identity = next((item for item in identities if item.get("batch_idx") == batch_idx), {})
        row.update({
            "subject_uid": identity.get("subject_uid"),
            "metadata_key": identity.get("metadata_key"),
            "frame_id": identity.get("frame_id"),
            "camera_ids": identity.get("camera_ids"),
        })
        rows.append(row)
        print(f"[P9.2ADAPTER-SENS] batch={batch_idx} donor={donor_idx} row={row}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_per_batch_csv(rows, output_dir)
    metrics = {
        "checkpoint": checkpoint,
        "base_config": base_config,
        "output_dir": output_dir,
        "split": args.split,
        "eval_protocol": eval_protocol,
        "train_ids": parse_id_list(args.train_ids),
        "holdout_ids": holdout_ids,
        "sacrificial_val_id": sacrificial_val_id,
        "selected_ids": selected_ids,
        "split_counts": split_counts,
        "batch_idx": args.batch_idx,
        "num_batches": args.num_batches,
        "evaluated_batch_indices": [idx for idx, _ in selected_batches],
        "evaluated_subject_uids": evaluated_subject_uids,
        "per_batch_csv": csv_path,
        "per_batch": rows,
        "aggregate": aggregate_rows(rows),
    }
    write_report(metrics, output_dir)
    print(f"[P9.2ADAPTER-SENS] wrote metrics: {output_dir / 'metrics.json'}")
    print(f"[P9.2ADAPTER-SENS] wrote report: {output_dir / 'report.md'}")
    print(f"[P9.2ADAPTER-SENS] wrote per-batch CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
