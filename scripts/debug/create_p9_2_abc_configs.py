#!/usr/bin/env python3
"""Create P9.2 A/B/C micro-render comparison configs.

The generated configs intentionally keep dataset paths generic.  Use them through
scripts/debug/run_motion_zero_token_overfit.sh so the local path override flow
writes resolved runtime configs with machine-local root/meta/val_id values.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def load_cfg(path: Path) -> Any:
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def save_cfg(cfg: Any, path: Path) -> None:
    from omegaconf import OmegaConf
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def find_latest_model_safetensors(checkpoint_root: Path, parent: str, child: str) -> Path | None:
    ckpt_dir = checkpoint_root / parent / child
    if not ckpt_dir.exists():
        return None
    numeric_dirs = sorted([p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    for step_dir in reversed(numeric_dirs):
        candidate = step_dir / "model.safetensors"
        if candidate.exists():
            return candidate.resolve()
    return None


def set_common_render_smoke(cfg: Any) -> None:
    cfg.model.debug_skip_renderer = False
    cfg.model.debug_latent_smoke_loss = False
    cfg.model.debug_max_query_points = None
    cfg.train.debug_global_steps = 1
    cfg.saver.checkpoint_global_steps = 1
    cfg.val.skip_eval = True
    # Avoid accidentally resuming A/B/C from previous smoke attempts.
    cfg.saver.auto_resume = False
    cfg.saver.load_model = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create P9.2 A/B/C micro-render comparison configs.")
    parser.add_argument("--base_config", type=Path, default=Path("configs/train/fastavatar_motion_zero_token_overfit_micro_render.yaml"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/mmhead_debug/p9_2_abc_configs"))
    parser.add_argument("--adapter_checkpoint", type=Path, default=None, help="Optional model.safetensors from the P9.2c 20-step adapter run.")
    parser.add_argument("--p9_2c_child", default="fastavatar_motion_zero_token_overfit_micro_render_20step")
    args = parser.parse_args()

    cfg = load_cfg(args.base_config)
    parent = str(cfg.experiment.parent)
    checkpoint_root = Path(str(cfg.saver.checkpoint_root)).expanduser()
    if not checkpoint_root.is_absolute():
        checkpoint_root = (Path.cwd() / checkpoint_root).resolve()

    adapter_checkpoint = args.adapter_checkpoint
    if adapter_checkpoint is not None:
        adapter_checkpoint = adapter_checkpoint.expanduser().resolve()
    else:
        adapter_checkpoint = find_latest_model_safetensors(checkpoint_root, parent, args.p9_2c_child)

    variants = {
        "A_normal_no_token": {
            "use_motion_token": False,
            "zero_flame_motion": False,
            "motion_token_source": "none",
            "load_model": None,
        },
        "B_zero_no_token": {
            "use_motion_token": False,
            "zero_flame_motion": True,
            "motion_token_source": "none",
            "load_model": None,
        },
        "C_zero_gt_token": {
            "use_motion_token": True,
            "zero_flame_motion": True,
            "motion_token_source": "frame_flame_gt",
            "load_model": str(adapter_checkpoint) if adapter_checkpoint and adapter_checkpoint.exists() else None,
        },
    }

    print(f"[P9.2ABC] base_config={args.base_config}")
    print(f"[P9.2ABC] output_dir={args.output_dir}")
    if adapter_checkpoint and adapter_checkpoint.exists():
        print(f"[P9.2ABC] C_zero_gt_token load_model={adapter_checkpoint}")
    else:
        print("[P9.2ABC][WARN] No P9.2c model.safetensors found; C_zero_gt_token will run with randomly initialized adapter unless a checkpoint is provided.")

    for name, opts in variants.items():
        variant_cfg = load_cfg(args.base_config)
        set_common_render_smoke(variant_cfg)
        variant_cfg.experiment.child = f"p9_2_abc_{name}"
        variant_cfg.model.use_motion_token = bool(opts["use_motion_token"])
        variant_cfg.model.zero_flame_motion = bool(opts["zero_flame_motion"])
        variant_cfg.model.motion_token_source = str(opts["motion_token_source"])
        # Only the token condition should freeze the backbone.  A/B remain normal model
        # smoke configs but still use the same dataset split and renderer settings.
        variant_cfg.model.freeze_backbone_for_motion_token = bool(opts["use_motion_token"])
        variant_cfg.model.motion_token_train_adapter_only_strict = bool(opts["use_motion_token"])
        if opts["load_model"]:
            variant_cfg.saver.load_model = opts["load_model"]
        out_path = args.output_dir / f"{name}.yaml"
        save_cfg(variant_cfg, out_path)
        print(f"[P9.2ABC] wrote {out_path}")


if __name__ == "__main__":
    main()
