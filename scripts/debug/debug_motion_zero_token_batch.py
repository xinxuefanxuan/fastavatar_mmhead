#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from FastAvatar.models.motion_token_adapter import MotionTokenAdapter


MOTION_FIELDS = ["expr", "neck_pose", "jaw_pose"]
ZERO_FIELDS = ["expr", "jaw_pose", "neck_pose", "rotation"]


def frame_average(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        return value.mean(dim=1)
    if value.ndim == 2:
        return value
    raise ValueError(f"Expected [B,N,D] or [B,D], got {list(value.shape)}")


def pad_or_truncate(value: torch.Tensor, target_dim: int) -> torch.Tensor:
    if value.shape[-1] > target_dim:
        return value[..., :target_dim]
    if value.shape[-1] < target_dim:
        pad = torch.zeros(*value.shape[:-1], target_dim - value.shape[-1], device=value.device, dtype=value.dtype)
        return torch.cat([value, pad], dim=-1)
    return value


def build_motion_token_input(flame_params: dict[str, torch.Tensor], norm_stats: Path | None, expr_dim: int, pad_to_dim: int) -> torch.Tensor:
    missing = [k for k in MOTION_FIELDS if k not in flame_params]
    if missing:
        raise ValueError(f"Missing required FLAME motion fields: {missing}")
    expr = pad_or_truncate(frame_average(flame_params["expr"])[..., :expr_dim], 50)
    neck = pad_or_truncate(frame_average(flame_params["neck_pose"]), 3)
    jaw = pad_or_truncate(frame_average(flame_params["jaw_pose"]), 3)
    motion56 = torch.cat([expr, neck, jaw], dim=-1)
    if norm_stats is not None:
        stats = json.loads(norm_stats.read_text())
        mean = torch.tensor(stats["mean"][:56], dtype=motion56.dtype, device=motion56.device)
        std = torch.tensor(stats["std"][:56], dtype=motion56.dtype, device=motion56.device).clamp_min(1e-8)
        motion56 = (motion56 - mean[None, :]) / std[None, :]
    return pad_or_truncate(motion56, pad_to_dim)


def zero_flame_motion(flame_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in flame_params.items():
        out[k] = torch.zeros_like(v) if k in ZERO_FIELDS else v
    return out


def field_norms(flame_params: dict[str, torch.Tensor]) -> dict[str, float]:
    out = {}
    for k in MOTION_FIELDS:
        v = flame_params[k]
        out[k] = float(torch.linalg.norm(v).detach().cpu())
    return out


def make_dummy_flame(batch_size: int, frames: int, expr_dim: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "expr": torch.randn(batch_size, frames, max(expr_dim, 100), device=device) * 0.05,
        "neck_pose": torch.randn(batch_size, frames, 3, device=device) * 0.03,
        "jaw_pose": torch.randn(batch_size, frames, 3, device=device) * 0.01,
        "rotation": torch.randn(batch_size, frames, 3, device=device) * 0.02,
        "translation": torch.randn(batch_size, frames, 3, device=device) * 0.001,
        "betas": torch.zeros(batch_size, frames, 300, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug P9.2 motion-zero + GT motion-token batch construction.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--expr_dim", type=int, default=50)
    parser.add_argument("--pad_to_dim", type=int, default=96)
    parser.add_argument("--transformer_dim", type=int, default=1024)
    parser.add_argument("--norm_stats", type=Path, default=Path("outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    norm_stats = args.norm_stats if args.norm_stats and args.norm_stats.exists() else None
    if args.norm_stats and norm_stats is None:
        print(f"[P9.2Debug] norm_stats not found, using raw motion56: {args.norm_stats}")

    flame_before = make_dummy_flame(args.batch_size, args.frames, args.expr_dim, device)
    token = build_motion_token_input(flame_before, norm_stats, args.expr_dim, args.pad_to_dim)
    flame_after = zero_flame_motion(flame_before)

    print(f"[P9.2Debug] motion_token_input shape: {list(token.shape)}")
    print(f"[P9.2Debug] motion_token_input mean/std/norm: {float(token.mean()):.6f}/{float(token.std()):.6f}/{float(torch.linalg.norm(token)):.6f}")
    print(f"[P9.2Debug] finite check: {bool(torch.isfinite(token).all())}")
    print(f"[P9.2Debug] expr/neck/jaw norms before zero: {field_norms(flame_before)}")
    print(f"[P9.2Debug] expr/neck/jaw norms after zero: {field_norms(flame_after)}")

    adapter = MotionTokenAdapter(input_dim=args.pad_to_dim, hidden_dim=args.transformer_dim, output_dim=args.transformer_dim).to(device)
    trainable = [(n, p.numel()) for n, p in adapter.named_parameters() if p.requires_grad]
    projected = adapter(token)
    print(f"[P9.2Debug] trainable parameter count: {sum(c for _, c in trainable)}")
    for name, count in trainable:
        print(f"[P9.2Debug] MotionTokenAdapter trainable: {name} ({count})")
    print(f"[P9.2Debug] projected token shape: {list(projected.shape)}")
    print("[P9.2Debug] Forward modes to compare in full training/inference: A normal/no-token, B zero/no-token, C zero+GT-token")


if __name__ == "__main__":
    main()
