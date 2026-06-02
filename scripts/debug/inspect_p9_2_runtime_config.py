#!/usr/bin/env python3
"""Inspect P9.2 runtime config knobs that control memory for micro smoke runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def load_config(path: Path) -> Any:
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def resolve_path(value: str | Path, base: Path) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def get_nested(cfg: Any, dotted: str, default: Any = None) -> Any:
    cur = cfg
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect P9.2 micro-smoke runtime config.")
    parser.add_argument("--config", type=Path, default=Path("configs/train/fastavatar_motion_zero_token_overfit_micro.yaml"))
    parser.add_argument("--require_micro", action="store_true", help="Fail if config is too large for a micro smoke run.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    cfg_path = resolve_path(args.config, repo_root)
    cfg = load_config(cfg_path)

    input_frames = int(cfg.dataset.input_frames)
    target_frames = int(cfg.dataset.target_frames)
    source_res = int(cfg.dataset.source_image_res)
    render_res = int(cfg.dataset.render_image_res)
    model_source_res = int(cfg.model.source_image_res)
    debug_max_query_points = get_nested(cfg, "model.debug_max_query_points", None)
    debug_max_query_points_int = None if debug_max_query_points is None else int(debug_max_query_points)

    print(f"[P9.2 Runtime] config={cfg_path}")
    print(f"[P9.2 Runtime] dataset.input_frames={input_frames}")
    print(f"[P9.2 Runtime] dataset.target_frames={target_frames}")
    print(f"[P9.2 Runtime] dataset.source_image_res={source_res}")
    print(f"[P9.2 Runtime] dataset.render_image_res={render_res}")
    print(f"[P9.2 Runtime] model.source_image_res={model_source_res}")
    print(f"[P9.2 Runtime] model.rendering_chunk_size_train={get_nested(cfg, 'model.rendering_chunk_size_train')}")
    print(f"[P9.2 Runtime] model.rendering_chunk_size_infer={get_nested(cfg, 'model.rendering_chunk_size_infer')}")
    print(f"[P9.2 Runtime] model.debug_max_query_points={debug_max_query_points}")
    print(f"[P9.2 Runtime] model.use_motion_token={get_nested(cfg, 'model.use_motion_token')}")
    print(f"[P9.2 Runtime] model.zero_flame_motion={get_nested(cfg, 'model.zero_flame_motion')}")
    print(f"[P9.2 Runtime] model.motion_token_source={get_nested(cfg, 'model.motion_token_source')}")
    print(f"[P9.2 Runtime] model.freeze_backbone_for_motion_token={get_nested(cfg, 'model.freeze_backbone_for_motion_token')}")
    print(f"[P9.2 Runtime] model.motion_token_train_adapter_only_strict={get_nested(cfg, 'model.motion_token_train_adapter_only_strict')}")

    loss = cfg.train.loss
    for key in ["pixel_weight", "perceptual_weight", "ssim_weight", "identity_weight", "offset_weight", "tracking_weight", "pruning_weight"]:
        print(f"[P9.2 Runtime] train.loss.{key}={getattr(loss, key, '<missing>')}")

    failures: list[str] = []
    if args.require_micro:
        if input_frames > 8:
            failures.append(f"dataset.input_frames={input_frames} > 8")
        if target_frames > 1:
            failures.append(f"dataset.target_frames={target_frames} > 1")
        if source_res > 128 or render_res > 128 or model_source_res > 128:
            failures.append(f"resolution too large: dataset source/render={source_res}/{render_res}, model source={model_source_res}")
        if debug_max_query_points_int is None:
            failures.append("model.debug_max_query_points is missing/null")
        elif debug_max_query_points_int > 4096:
            failures.append(f"model.debug_max_query_points={debug_max_query_points_int} > 4096")
        if failures:
            print("[P9.2 Runtime][ERROR] Config is not micro-safe:")
            for failure in failures:
                print(f"  - {failure}")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
