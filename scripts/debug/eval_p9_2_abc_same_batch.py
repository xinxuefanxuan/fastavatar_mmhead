#!/usr/bin/env python3
"""Strict same-batch no-grad P9.2 A/B/C/D evaluation.

This script resolves the same local metadata/root overrides as the P9.2 smoke
runner, loads fixed FastAvatar batches, and evaluates four model variants on
exactly the same data without optimizer/backward/checkpointing.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def load_yaml(path: Path) -> Any:
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def save_yaml(cfg: Any, path: Path) -> None:
    from omegaconf import OmegaConf
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    return getattr(obj, key) if hasattr(obj, key) else default


def resolve_path(value: str | Path, repo: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo / path).resolve()


def load_local_overrides(path: Path) -> Any:
    if path.exists():
        return load_yaml(path)
    from omegaconf import OmegaConf
    return OmegaConf.create({})


def resolve_local_paths(base_cfg: Any, repo: Path, local_cfg_path: Path) -> dict[str, Path | str]:
    local_cfg = load_local_overrides(local_cfg_path)
    root = (
        os.environ.get("P9_NERSEMBLE_ROOT")
        or cfg_get(local_cfg, "nersemble_root_dir", None)
        or base_cfg.dataset.datasets.nersemble.root_dir
    )
    source = (
        os.environ.get("P9_SOURCE_META")
        or cfg_get(local_cfg, "source_mixed_uids", None)
        or "datasets/mixed_uids.json"
    )
    generated = (
        os.environ.get("P9_GENERATED_META")
        or cfg_get(local_cfg, "generated_meta_path", None)
        or base_cfg.dataset.meta_path
    )
    preferred = os.environ.get("P9_VAL_ID") or cfg_get(local_cfg, "preferred_val_id", None) or ""
    return {
        "root_dir": resolve_path(root, repo),
        "source_meta": resolve_path(source, repo),
        "generated_meta": resolve_path(generated, repo),
        "preferred_val_id": str(preferred or ""),
    }


def extract_uid(key: str) -> str:
    if key.startswith("nersemble/"):
        key = key[len("nersemble/"):]
    return key.split("/", 1)[0]


def generate_metadata(base_config: Path, paths: dict[str, Path | str], prefer_ids: list[str] | None = None, max_ids: int = 6) -> None:
    cmd = [
        sys.executable,
        "scripts/debug/create_p9_2_overfit_metadata.py",
        "--config",
        str(base_config),
        "--src_meta",
        str(paths["source_meta"]),
        "--root_dir",
        str(paths["root_dir"]),
        "--output",
        str(paths["generated_meta"]),
        "--min_pairs",
        "auto",
        "--max_ids",
        str(max_ids),
        "--max_items_per_id",
        "4",
    ]
    if prefer_ids:
        cmd.extend(["--prefer_ids", ",".join(prefer_ids)])
    print("[P9.2ABC-EVAL] generate metadata:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def choose_val_id(meta_path: Path, preferred: str) -> tuple[str, list[str]]:
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    ids = sorted({extract_uid(k) for k in meta})
    if not ids:
        raise RuntimeError(f"Generated metadata contains no IDs: {meta_path}")
    if preferred and preferred in ids:
        val_id = preferred
    elif preferred:
        print(f"[P9.2ABC-EVAL][WARN] preferred val_id={preferred} not in selected IDs {ids}; choosing {ids[-1]}")
        val_id = ids[-1]
    else:
        val_id = ids[-1]
    train_ids = [uid for uid in ids if uid != val_id]
    if not train_ids:
        raise RuntimeError(f"Chosen val_id={val_id} leaves zero train IDs from selected IDs={ids}")
    return val_id, ids


def selected_ids_from_meta(meta_path: Path) -> list[str]:
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return sorted({extract_uid(k) for k in meta})


def parse_id_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def filter_metadata_by_ids(src_meta: Path, output_meta: Path, include_ids: list[str]) -> tuple[Path, list[str], int]:
    include = set(include_ids)
    with src_meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    filtered = {key: value for key, value in meta.items() if extract_uid(key) in include}
    if not filtered:
        raise RuntimeError(f"No metadata rows found for requested holdout_ids={include_ids} in {src_meta}")
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    output_meta.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    present_ids = sorted({extract_uid(key) for key in filtered})
    return output_meta, present_ids, len(filtered)


def write_resolved_config(base_config: Path, cfg: Any, paths: dict[str, Path | str], val_id: str, output_dir: Path) -> Path:
    runtime_path = output_dir / "runtime_configs" / f"{base_config.stem}_same_batch_eval_resolved.yaml"
    cfg = copy.deepcopy(cfg)
    cfg.dataset.meta_path = str(paths["generated_meta"])
    cfg.dataset.datasets.nersemble.root_dir = str(paths["root_dir"])
    cfg.dataset.datasets.nersemble.val_id = [val_id]
    cfg.model.debug_skip_renderer = False
    cfg.model.debug_latent_smoke_loss = False
    cfg.model.debug_max_query_points = None
    cfg.val.skip_eval = True
    cfg.saver.auto_resume = False
    cfg.saver.load_model = None
    save_yaml(cfg, runtime_path)
    return runtime_path


def write_runtime_config_with_meta(
    base_config: Path,
    cfg: Any,
    root_dir: Path,
    meta_path: Path,
    val_ids: list[str],
    output_dir: Path,
    suffix: str,
) -> Path:
    runtime_path = output_dir / "runtime_configs" / f"{base_config.stem}_{suffix}.yaml"
    cfg = copy.deepcopy(cfg)
    cfg.dataset.meta_path = str(meta_path)
    cfg.dataset.datasets.nersemble.root_dir = str(root_dir)
    cfg.dataset.datasets.nersemble.val_id = list(val_ids)
    cfg.dataset.val_num = 0
    cfg.model.debug_skip_renderer = False
    cfg.model.debug_latent_smoke_loss = False
    cfg.model.debug_max_query_points = None
    cfg.val.skip_eval = True
    cfg.saver.auto_resume = False
    cfg.saver.load_model = None
    save_yaml(cfg, runtime_path)
    return runtime_path


VARIANTS = (
    "A_normal_no_token",
    "B_zero_no_token",
    "C_zero_gt_token_trained_adapter",
    "D_zero_gt_token_random_adapter",
)

VARIANT_LABELS = {
    "A_normal_no_token": "A-normal",
    "B_zero_no_token": "B-zero",
    "C_zero_gt_token_trained_adapter": "C-trained-token",
    "D_zero_gt_token_random_adapter": "D-random-token",
}

VARIANT_SHORT = {
    "A_normal_no_token": "A",
    "B_zero_no_token": "B",
    "C_zero_gt_token_trained_adapter": "C",
    "D_zero_gt_token_random_adapter": "D",
}


def make_variant_config(cfg: Any, name: str, adapter_ckpt: Path | None) -> Any:
    out = copy.deepcopy(cfg)
    out.model.debug_skip_renderer = False
    out.model.debug_latent_smoke_loss = False
    out.model.debug_max_query_points = None
    out.saver.auto_resume = False
    out.saver.load_model = None
    if name == "A_normal_no_token":
        out.model.use_motion_token = False
        out.model.zero_flame_motion = False
        out.model.motion_token_source = "none"
        out.model.freeze_backbone_for_motion_token = False
        out.model.motion_token_train_adapter_only_strict = False
    elif name == "B_zero_no_token":
        out.model.use_motion_token = False
        out.model.zero_flame_motion = True
        out.model.motion_token_source = "none"
        out.model.freeze_backbone_for_motion_token = False
        out.model.motion_token_train_adapter_only_strict = False
    elif name in ("C_zero_gt_token", "C_zero_gt_token_trained_adapter"):
        out.model.use_motion_token = True
        out.model.zero_flame_motion = True
        out.model.motion_token_source = "frame_flame_gt"
        out.model.freeze_backbone_for_motion_token = True
        out.model.motion_token_train_adapter_only_strict = True
        if adapter_ckpt is not None:
            out.saver.load_model = str(adapter_ckpt)
    elif name == "D_zero_gt_token_random_adapter":
        out.model.use_motion_token = True
        out.model.zero_flame_motion = True
        out.model.motion_token_source = "frame_flame_gt"
        out.model.freeze_backbone_for_motion_token = True
        out.model.motion_token_train_adapter_only_strict = True
        out.saver.load_model = None
    else:
        raise ValueError(f"Unknown variant {name}")
    return out


def build_loader(cfg: Any, split: str) -> torch.utils.data.DataLoader:
    from FastAvatar.datasets.mixer import MixerDataset

    subsets = []
    for name, dataset_cfg in cfg.dataset.datasets.items():
        subsets.append({
            "name": name,
            "root_dirs": dataset_cfg.root_dir,
            "meta_path": cfg.dataset.meta_path,
            "val_id": cfg_get(dataset_cfg, "val_id", None),
        })
    is_val = split == "val"
    dataset = MixerDataset(
        split=split,
        subsets=subsets,
        input_frames=cfg.dataset.input_frames,
        frames_per_sample=cfg.dataset.input_frames + cfg.dataset.target_frames,
        render_image_res=cfg.dataset.render_image_res,
        source_image_res=cfg.dataset.source_image_res,
        disorder=False,
        val_num=cfg.dataset.val_num,
        is_val=is_val,
        use_teeth=cfg_get(cfg.model, "add_teeth", True),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def tensor_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    try:
        return float(value)
    except Exception:
        return None


def flame_motion_summary(batch: dict[str, Any]) -> dict[str, float | None]:
    summary = {}
    for key in ("target_expr", "target_neck_pose", "target_jaw_pose", "target_rotation"):
        value = batch.get(key)
        if torch.is_tensor(value):
            summary[f"{key}_norm_mean"] = float(torch.linalg.norm(value.detach().float().reshape(value.shape[0], -1), dim=-1).mean().cpu())
            summary[f"{key}_norm_max"] = float(torch.linalg.norm(value.detach().float().reshape(value.shape[0], -1), dim=-1).max().cpu())
        else:
            summary[f"{key}_norm_mean"] = None
            summary[f"{key}_norm_max"] = None
    return summary


def build_flame_dicts(data: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    flame_keys = ['expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'teeth_bs', 'translation', 'shape', 'betas']
    input_flame_params = {
        k.replace('input_', ''): v for k, v in data.items()
        if k.startswith('input_') and k.replace('input_', '') in flame_keys
    }
    target_flame_params = {
        k.replace('target_', ''): v for k, v in data.items()
        if k.startswith('target_') and k.replace('target_', '') in flame_keys
    }
    return input_flame_params, target_flame_params


def forward_and_loss(model: torch.nn.Module, cfg: Any, batch: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, float | None]]:
    from FastAvatar.losses import PixelLoss

    input_flame_params, target_flame_params = build_flame_dicts(batch)
    landmarks = torch.cat([batch['landmarks'], batch['target_landmarks']], dim=1)
    outputs = model(
        input_image=batch['rgbs'],
        target_image=batch['target_rgbs'],
        input_c2ws=batch['c2ws'],
        target_c2ws=batch['target_c2ws'],
        input_intrs=batch['intrs'],
        target_intrs=batch['target_intrs'],
        input_bg_colors=batch['bg_colors'],
        target_bg_colors=batch['target_bg_colors'],
        landmarks=landmarks,
        input_flame_params=input_flame_params,
        inf_flame_params=target_flame_params,
        uid=batch['uid'],
    )
    losses: dict[str, float | None] = {
        "total_loss": None,
        "r_pixel": None,
        "r_perceptual": None,
        "r_ssim": None,
        "r_id": None,
    }
    if "comp_rgb" in outputs:
        pixel_loss = PixelLoss(option=cfg.train.loss.pixel_loss_type)(outputs["comp_rgb"], batch["target_rgbs"])
        total = pixel_loss * float(cfg.train.loss.pixel_weight)
        losses["r_pixel"] = tensor_scalar(pixel_loss)
        losses["total_loss"] = tensor_scalar(total)
    elif "latent_smoke_loss" in outputs:
        losses["total_loss"] = tensor_scalar(outputs["latent_smoke_loss"])
    return outputs, losses


def load_model_for_variant(cfg: Any, adapter_ckpt: Path | None, device: torch.device) -> tuple[torch.nn.Module, bool]:
    from FastAvatar.models.modeling_FastAvatar import ModelFastAvatar

    model = ModelFastAvatar(**cfg.model)
    loaded_adapter = False
    if cfg_get(cfg.saver, "load_model", None):
        ckpt = Path(str(cfg.saver.load_model)).expanduser()
        if ckpt.exists():
            import safetensors.torch
            missing, unexpected = safetensors.torch.load_model(model, str(ckpt), strict=False)
            loaded_adapter = True
            print(f"[P9.2ABC-EVAL] loaded checkpoint={ckpt} missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print(f"[P9.2ABC-EVAL][WARN] requested checkpoint does not exist: {ckpt}")
    model.to(device)
    model.eval()
    return model, loaded_adapter


IMAGE_KEY_PRIORITY = (
    "comp_rgb",
    "render_rgb",
    "rgb",
    "image",
    "pred_rgb",
    "pred_image",
    "rendered_image",
)


def _tensor_min_max(tensor: torch.Tensor) -> tuple[float | None, float | None]:
    if tensor.numel() == 0:
        return None, None
    tensor = tensor.detach()
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return None, None
    return float(finite.min().cpu()), float(finite.max().cpu())


def print_output_debug_shapes(outputs: Any, prefix: str = "outputs") -> None:
    """Print nested output types/shapes for debugging image selection."""
    def visit(obj: Any, path: str) -> None:
        if torch.is_tensor(obj):
            t_min, t_max = _tensor_min_max(obj)
            print(
                f"[P9.2ABC-EVAL][OutputShape] {path}: "
                f"type=tensor shape={list(obj.shape)} dtype={obj.dtype} min={t_min} max={t_max}"
            )
        elif isinstance(obj, dict):
            print(f"[P9.2ABC-EVAL][OutputShape] {path}: type=dict keys={list(obj.keys())}")
            for key, value in obj.items():
                visit(value, f"{path}.{key}")
        elif isinstance(obj, (list, tuple)):
            print(f"[P9.2ABC-EVAL][OutputShape] {path}: type={type(obj).__name__} len={len(obj)}")
            for idx, value in enumerate(obj):
                visit(value, f"{path}[{idx}]")
        else:
            print(f"[P9.2ABC-EVAL][OutputShape] {path}: type={type(obj).__name__}")

    visit(outputs, prefix)


def _is_image_channel_count(channels: int) -> bool:
    return int(channels) in (1, 3, 4)


def _normalize_image_tensor(img: torch.Tensor) -> torch.Tensor:
    img = img.detach().float().cpu()
    if img.numel() > 0:
        img_min = float(img.min())
        img_max = float(img.max())
        if img_min < 0.0 and img_max <= 1.0:
            img = (img + 1.0) * 0.5
        elif img_max > 2.0:
            img = img / 255.0
    return img.clamp(0.0, 1.0)


def _image_frames_from_tensor(tensor: torch.Tensor, max_frames: int) -> list[torch.Tensor]:
    """Convert supported image-like tensors into a list of CHW tensors.

    Supported shapes include [B,T,C,H,W], [B,T,H,W,C], [B,C,H,W],
    [T,C,H,W], [C,H,W], [H,W,C], and [H,W]. Tensors whose image
    channel dimension is not 1/3/4 are rejected to avoid saving latent
    features such as [512,1,1].
    """
    if not torch.is_tensor(tensor):
        return []
    shape = list(tensor.shape)
    frames: list[torch.Tensor] = []

    if tensor.ndim == 5:
        # [B,T,C,H,W]
        if _is_image_channel_count(shape[2]):
            n = min(max_frames, shape[1])
            frames = [tensor[0, idx] for idx in range(n)]
        # FastAvatar commonly returns [B,T,H,W,C].
        elif _is_image_channel_count(shape[-1]):
            n = min(max_frames, shape[1])
            frames = [tensor[0, idx].permute(2, 0, 1) for idx in range(n)]
        else:
            return []
    elif tensor.ndim == 4:
        # [N,C,H,W], where N may be batch or time.
        if _is_image_channel_count(shape[1]):
            n = min(max_frames, shape[0])
            frames = [tensor[idx] for idx in range(n)]
        # [N,H,W,C]
        elif _is_image_channel_count(shape[-1]):
            n = min(max_frames, shape[0])
            frames = [tensor[idx].permute(2, 0, 1) for idx in range(n)]
        else:
            return []
    elif tensor.ndim == 3:
        # [C,H,W]
        if _is_image_channel_count(shape[0]):
            frames = [tensor]
        # [H,W,C]
        elif _is_image_channel_count(shape[-1]):
            frames = [tensor.permute(2, 0, 1)]
        else:
            return []
    elif tensor.ndim == 2:
        frames = [tensor.unsqueeze(0)]
    else:
        return []

    return [_normalize_image_tensor(frame) for frame in frames if frame.ndim == 3 and _is_image_channel_count(frame.shape[0])]


def _iter_key_matches(obj: Any, target_key: str, path: str = "outputs"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}"
            if key == target_key:
                yield child_path, value
            yield from _iter_key_matches(value, target_key, child_path)
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from _iter_key_matches(value, target_key, f"{path}[{idx}]")


def find_first_image_tensor(outputs: Any, max_frames: int) -> tuple[str, list[torch.Tensor]] | None:
    for key in IMAGE_KEY_PRIORITY:
        for path, value in _iter_key_matches(outputs, key):
            frames = _image_frames_from_tensor(value, max_frames=max_frames)
            if frames:
                return path, frames
            if torch.is_tensor(value):
                print(
                    f"[P9.2ABC-EVAL][WARN] Skip non-image tensor at {path}: "
                    f"shape={list(value.shape)} dtype={value.dtype}"
                )
    return None


def save_render_images(outputs: dict[str, Any], out_dir: Path, prefix: str, max_frames: int = 4) -> list[str]:
    saved: list[str] = []
    found = find_first_image_tensor(outputs, max_frames=max_frames)
    if found is None:
        print("[P9.2ABC-EVAL][WARN] No valid image-like output tensor found; skip image saving.")
        return saved

    image_path, frames = found
    from torchvision.utils import save_image

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[P9.2ABC-EVAL] Saving images from {image_path}; frames={len(frames)}")
    for idx, img in enumerate(frames[:max_frames]):
        path = out_dir / f"{prefix}_{idx:03d}.png"
        save_image(img, path)
        saved.append(str(path))
    return saved


def save_first_render_image(outputs: dict[str, Any], path: Path) -> list[str]:
    frame = first_image_frame_from_outputs(outputs)
    if frame is None:
        print(f"[P9.2ABC-EVAL][WARN] No valid image tensor for {path.name}; skip image saving.")
        return []
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(frame, path)
    return [str(path)]


def first_image_frame_from_outputs(outputs: Any) -> torch.Tensor | None:
    found = find_first_image_tensor(outputs, max_frames=1)
    if found is None:
        return None
    _, frames = found
    return frames[0] if frames else None


def save_variant_grid(frames_by_variant: dict[str, torch.Tensor], path: Path) -> str | None:
    if not frames_by_variant:
        return None
    from torchvision.utils import make_grid, save_image

    ordered = [name for name in VARIANTS if name in frames_by_variant]
    if not ordered:
        return None
    tensors = [frames_by_variant[name] for name in ordered]
    # All tensors should be CHW with image channels; resize is intentionally not
    # performed here because A/B/C use the same batch/config and should match.
    try:
        grid = make_grid(torch.stack(tensors, dim=0), nrow=len(tensors))
    except Exception as exc:
        print(f"[P9.2ABC-EVAL][WARN] Could not build image grid {path}: {exc}")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)
    return str(path)


def save_labeled_eval_grid(frames_by_key: dict[str, torch.Tensor], path: Path) -> str | None:
    ordered_keys = ["GT"] + [name for name in VARIANTS if name in frames_by_key]
    ordered = [(key, frames_by_key[key]) for key in ordered_keys if key in frames_by_key]
    if not ordered:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"[P9.2ABC-EVAL][WARN] PIL unavailable; cannot build labeled grid {path}: {exc}")
        return save_variant_grid({key: frame for key, frame in ordered if key != "GT"}, path)

    pil_images = []
    labels = []
    for key, frame in ordered:
        img = _normalize_image_tensor(frame)
        if img.ndim != 3 or not _is_image_channel_count(img.shape[0]):
            continue
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        array = (img.permute(1, 2, 0).clamp(0, 1) * 255.0).byte().numpy()
        pil_images.append(Image.fromarray(array, mode="RGB"))
        labels.append("GT" if key == "GT" else VARIANT_LABELS.get(key, key))
    if not pil_images:
        return None
    width = max(img.width for img in pil_images)
    height = max(img.height for img in pil_images)
    label_h = 36
    canvas = Image.new("RGB", (width * len(pil_images), height + label_h), color="white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for idx, (label, img) in enumerate(zip(labels, pil_images)):
        if img.size != (width, height):
            img = img.resize((width, height))
        x0 = idx * width
        draw.text((x0 + 6, 7), label, fill=(0, 0, 0), font=font)
        canvas.paste(img, (x0, label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def print_batch_debug_shapes(batch: dict[str, Any], prefix: str = "batch") -> None:
    for key, value in batch.items():
        if torch.is_tensor(value):
            t_min, t_max = _tensor_min_max(value)
            print(
                f"[P9.2ABC-EVAL][BatchShape] {prefix}.{key}: "
                f"type=tensor shape={list(value.shape)} dtype={value.dtype} min={t_min} max={t_max}"
            )
        else:
            print(f"[P9.2ABC-EVAL][BatchShape] {prefix}.{key}: type={type(value).__name__}")


def first_gt_image_frame_from_batch(batch: dict[str, Any]) -> torch.Tensor | None:
    for key in ("target_rgbs", "target_rgb", "target_image", "image", "rgbs"):
        if key in batch:
            frames = _image_frames_from_tensor(batch[key], max_frames=1)
            if frames:
                return frames[0]
    return None


def _summarize_value(value: Any, max_items: int = 4) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.numel() == 0:
            return []
        if tensor.numel() <= max_items:
            return tensor.reshape(-1).tolist()
        return tensor.reshape(-1)[:max_items].tolist()
    if isinstance(value, (list, tuple)):
        return [_summarize_value(v, max_items=max_items) for v in value[:max_items]]
    if isinstance(value, dict):
        return {str(k): _summarize_value(v, max_items=max_items) for k, v in list(value.items())[:max_items]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _first_existing(batch: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in batch:
            return _summarize_value(batch[key])
    return None


def extract_batch_identity(batch: dict[str, Any], split: str, eval_protocol: str, batch_idx: int) -> dict[str, Any]:
    uid = _first_existing(batch, ("uid", "uids", "subject_id", "identity_id", "id"))
    return {
        "split": split,
        "eval_protocol": eval_protocol,
        "batch_idx": batch_idx,
        "uid": uid,
        "subject_id": _first_existing(batch, ("subject_id", "identity_id", "id")) or uid,
        "frame_id": _first_existing(batch, ("frame_id", "frame_ids", "target_frame_id", "target_frame_ids", "frame_idx", "target_frame_idx")),
        "metadata_key": _first_existing(batch, ("key", "keys", "meta_key", "metadata_key", "sample_key", "path")),
        "metadata_path": _first_existing(batch, ("metadata_path", "meta_path")),
        "source_image_path": _first_existing(batch, ("source_image_path", "input_image_path", "image_path", "rgb_path", "rgbs_path")),
        "target_image_path": _first_existing(batch, ("target_image_path", "target_rgb_path", "target_rgbs_path")),
        "available_batch_keys": sorted(str(k) for k in batch.keys()),
    }


def write_batch_identity_files(identities: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "batch_identity.json"
    csv_path = output_dir / "batch_identity.csv"
    json_path.write_text(json.dumps(json_safe(identities), ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "split",
        "eval_protocol",
        "batch_idx",
        "uid",
        "subject_id",
        "frame_id",
        "metadata_key",
        "metadata_path",
        "source_image_path",
        "target_image_path",
        "available_batch_keys",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in identities:
            writer.writerow({key: json.dumps(json_safe(row.get(key)), ensure_ascii=False) for key in fieldnames})
    return json_path, csv_path


def mean_std(values: list[float]) -> dict[str, float | None]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"mean": None, "std": None}
    mean = sum(finite) / len(finite)
    if len(finite) == 1:
        std = 0.0
    else:
        std = (sum((v - mean) ** 2 for v in finite) / len(finite)) ** 0.5
    return {"mean": mean, "std": std}


def fmt_float(value: float | None, digits: int = 6) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def compute_aggregate(per_batch_results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"variants": {}, "improvement": {}}
    for variant in VARIANTS:
        losses = [rows.get(variant, {}).get("losses", {}) for rows in per_batch_results.values()]
        aggregate["variants"][variant] = {
            "total_loss": mean_std([loss.get("total_loss") for loss in losses]),
            "r_pixel": mean_std([loss.get("r_pixel") for loss in losses]),
        }

    b_minus_c: list[float] = []
    b_minus_d: list[float] = []
    d_minus_c: list[float] = []
    rel_b_minus_c: list[float] = []
    rel_b_minus_d: list[float] = []
    c_lt_b_count = 0
    d_lt_b_count = 0
    c_lt_d_count = 0
    compared_bc = 0
    compared_bd = 0
    compared_cd = 0
    for rows in per_batch_results.values():
        b_val = rows.get("B_zero_no_token", {}).get("losses", {}).get("r_pixel")
        c_val = rows.get("C_zero_gt_token_trained_adapter", {}).get("losses", {}).get("r_pixel")
        d_val = rows.get("D_zero_gt_token_random_adapter", {}).get("losses", {}).get("r_pixel")
        b = None if b_val is None else float(b_val)
        c = None if c_val is None else float(c_val)
        d = None if d_val is None else float(d_val)
        if b is not None and c is not None:
            diff = b - c
            b_minus_c.append(diff)
            if b != 0.0:
                rel_b_minus_c.append(diff / b)
            c_lt_b_count += int(c < b)
            compared_bc += 1
        if b is not None and d is not None:
            diff = b - d
            b_minus_d.append(diff)
            if b != 0.0:
                rel_b_minus_d.append(diff / b)
            d_lt_b_count += int(d < b)
            compared_bd += 1
        if c is not None and d is not None:
            d_minus_c.append(d - c)
            c_lt_d_count += int(c < d)
            compared_cd += 1
    aggregate["improvement"] = {
        "b_minus_c_r_pixel": mean_std(b_minus_c),
        "relative_b_minus_c_over_b": mean_std(rel_b_minus_c),
        "b_minus_d_r_pixel": mean_std(b_minus_d),
        "relative_b_minus_d_over_b": mean_std(rel_b_minus_d),
        "d_minus_c_r_pixel": mean_std(d_minus_c),
        "percent_batches_c_lt_b": None if compared_bc == 0 else c_lt_b_count / compared_bc,
        "percent_batches_d_lt_b": None if compared_bd == 0 else d_lt_b_count / compared_bd,
        "percent_batches_c_lt_d": None if compared_cd == 0 else c_lt_d_count / compared_cd,
        "num_compared_batches_bc": compared_bc,
        "num_compared_batches_bd": compared_bd,
        "num_compared_batches_cd": compared_cd,
    }
    return aggregate


def write_per_batch_csv(per_batch_results: dict[int, dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "per_batch_metrics.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_idx",
        "variant",
        "total_loss",
        "r_pixel",
        "r_perceptual",
        "r_ssim",
        "r_id",
        "adapter_loaded",
        "random_adapter",
        "images",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for batch_idx in sorted(per_batch_results):
            for variant, row in per_batch_results[batch_idx].items():
                losses = row.get("losses", {})
                writer.writerow({
                    "batch_idx": batch_idx,
                    "variant": variant,
                    "total_loss": losses.get("total_loss"),
                    "r_pixel": losses.get("r_pixel"),
                    "r_perceptual": losses.get("r_perceptual"),
                    "r_ssim": losses.get("r_ssim"),
                    "r_id": losses.get("r_id"),
                    "adapter_loaded": row.get("adapter_loaded", False),
                    "random_adapter": row.get("random_adapter", False),
                    "images": ";".join(row.get("images", [])),
                })
    return path


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def write_report(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")

    aggregate = metrics.get("aggregate", {})
    per_batch_results = metrics.get("per_batch_results", {})
    variant_configs = metrics.get("variant_configs", {})
    improvement = aggregate.get("improvement", {})
    b_minus_c = improvement.get("b_minus_c_r_pixel", {})
    rel_bc = improvement.get("relative_b_minus_c_over_b", {})
    b_minus_d = improvement.get("b_minus_d_r_pixel", {})
    rel_bd = improvement.get("relative_b_minus_d_over_b", {})
    d_minus_c = improvement.get("d_minus_c_r_pixel", {})
    c_lt_b_pct = improvement.get("percent_batches_c_lt_b")
    d_lt_b_pct = improvement.get("percent_batches_d_lt_b")
    c_lt_d_pct = improvement.get("percent_batches_c_lt_d")

    lines = [
        "# P9.2 Same-Batch A/B/C/D Evaluation",
        "",
        "## Run Selection",
        f"* evaluation protocol: `{metrics.get('eval_protocol')}`",
        f"* requested split: `{metrics.get('requested_split')}`",
        f"* evaluated split: `{metrics.get('split')}`",
        f"* allow_fallback_to_train: `{metrics.get('allow_fallback_to_train')}`",
        f"* split counts: `{metrics.get('split_counts')}`",
        f"* train IDs used for adapter: `{metrics.get('train_ids_used_for_adapter')}`",
        f"* holdout IDs: `{metrics.get('holdout_ids')}`",
        f"* holdout/train disjoint: `{metrics.get('holdout_train_ids_disjoint')}`",
        f"* selected_ids: `{metrics.get('selected_ids')}`",
        f"* val_id: `{metrics.get('val_id')}`",
        f"* batch_idx start: `{metrics.get('batch_idx')}`",
        f"* requested num_batches: `{metrics.get('num_batches')}`",
        f"* evaluated batch indices: `{metrics.get('evaluated_batch_indices')}`",
        "",
        "## Configs",
        "| Variant | label | use_motion_token | zero_flame_motion | motion_token_source | adapter_loaded | random_adapter |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for name, cfg in variant_configs.items():
        lines.append(
            f"| {name} | {cfg.get('label', '')} | {cfg.get('use_motion_token')} | {cfg.get('zero_flame_motion')} | "
            f"{cfg.get('motion_token_source')} | {cfg.get('adapter_loaded', False)} | {cfg.get('random_adapter', False)} |"
        )

    identities = metrics.get("batch_identities", [])
    lines += [
        "",
        "## Evaluated Samples",
        "| split | eval_protocol | batch_idx | uid | subject_id | frame_id | metadata_key | source_image_path | target_image_path |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for identity in identities:
        lines.append(
            f"| {identity.get('split')} | {identity.get('eval_protocol')} | {identity.get('batch_idx')} | {identity.get('uid')} | "
            f"{identity.get('subject_id')} | {identity.get('frame_id')} | {identity.get('metadata_key')} | "
            f"{identity.get('source_image_path')} | {identity.get('target_image_path')} |"
        )

    lines += [
        "",
        "## Image Grids",
        "| batch_idx | grid_path |",
        "|---:|---|",
    ]
    for batch_idx, grid_path in sorted((metrics.get("image_grids") or {}).items(), key=lambda item: int(item[0])):
        lines.append(f"| {batch_idx} | {grid_path or 'N/A'} |")

    lines += [
        "",
        "## Aggregate Losses",
        "| Variant | total_loss mean | total_loss std | r_pixel mean | r_pixel std |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in aggregate.get("variants", {}).items():
        total = stats.get("total_loss", {})
        pixel = stats.get("r_pixel", {})
        lines.append(
            f"| {name} | {fmt_float(total.get('mean'))} | {fmt_float(total.get('std'))} | "
            f"{fmt_float(pixel.get('mean'))} | {fmt_float(pixel.get('std'))} |"
        )

    lines += [
        "",
        "## Relative Improvement",
        f"* B - C r_pixel mean/std: {fmt_float(b_minus_c.get('mean'))} / {fmt_float(b_minus_c.get('std'))}",
        f"* (B - C) / B mean/std: {fmt_float(rel_bc.get('mean'), 4)} / {fmt_float(rel_bc.get('std'), 4)}",
        f"* B - D r_pixel mean/std: {fmt_float(b_minus_d.get('mean'))} / {fmt_float(b_minus_d.get('std'))}",
        f"* (B - D) / B mean/std: {fmt_float(rel_bd.get('mean'), 4)} / {fmt_float(rel_bd.get('std'), 4)}",
        f"* D - C r_pixel gap mean/std: {fmt_float(d_minus_c.get('mean'))} / {fmt_float(d_minus_c.get('std'))}",
        f"* Percentage of batches where C < B: {'N/A' if c_lt_b_pct is None else f'{c_lt_b_pct:.2%}'}",
        f"* Percentage of batches where D < B: {'N/A' if d_lt_b_pct is None else f'{d_lt_b_pct:.2%}'}",
        f"* Percentage of batches where C < D: {'N/A' if c_lt_d_pct is None else f'{c_lt_d_pct:.2%}'}",
        "",
        "## Per-Batch Losses",
        "| batch_idx | A r_pixel | B r_pixel | C r_pixel | D r_pixel | B-C | B-D | D-C | C < B | D < B | C < D |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for batch_idx in sorted(int(k) for k in per_batch_results.keys()):
        rows = per_batch_results[batch_idx]
        a = rows.get("A_normal_no_token", {}).get("losses", {}).get("r_pixel")
        b = rows.get("B_zero_no_token", {}).get("losses", {}).get("r_pixel")
        c = rows.get("C_zero_gt_token_trained_adapter", {}).get("losses", {}).get("r_pixel")
        d = rows.get("D_zero_gt_token_random_adapter", {}).get("losses", {}).get("r_pixel")
        bc = None if b is None or c is None else b - c
        bd = None if b is None or d is None else b - d
        dc = None if d is None or c is None else d - c
        c_lt_b = c is not None and b is not None and c < b
        d_lt_b = d is not None and b is not None and d < b
        c_lt_d = c is not None and d is not None and c < d
        lines.append(
            f"| {batch_idx} | {fmt_float(a)} | {fmt_float(b)} | {fmt_float(c)} | {fmt_float(d)} | "
            f"{fmt_float(bc)} | {fmt_float(bd)} | {fmt_float(dc)} | {c_lt_b} | {d_lt_b} | {c_lt_d} |"
        )

    lines += [
        "",
        "## Conclusion",
    ]
    if c_lt_b_pct is None:
        lines.append("* No comparable B/C r_pixel batches were evaluated.")
    elif c_lt_b_pct > 0.5:
        lines.append("* C improves over B on most evaluated batches, suggesting the trained GT motion token helps under zeroed FLAME motion.")
    else:
        lines.append("* C does not improve over B on most evaluated batches; this may indicate adapter overfit to a different batch or weak token conditioning.")
    if c_lt_d_pct is not None:
        if c_lt_d_pct > 0.5:
            lines.append("* C is better than random-adapter D on most batches, supporting that adapter training matters.")
        else:
            lines.append("* D is competitive with or better than C on many batches; diagnose whether token conditioning is too weak or the batch is out-of-distribution.")
    if metrics.get("holdout_ids"):
        lines.append(f"* Holdout/train ID disjoint check: {metrics.get('holdout_train_ids_disjoint')}.")
    (output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict same-batch P9.2 A/B/C/D no-grad evaluation.")
    parser.add_argument("--base_config", type=Path, default=Path("configs/train/fastavatar_motion_zero_token_overfit_micro_render.yaml"))
    parser.add_argument("--local_paths_config", type=Path, default=Path("configs/local/p9_2_local_paths.yaml"))
    parser.add_argument("--adapter_ckpt", type=Path, default=Path("exps/checkpoints/fastavatar/fastavatar_motion_zero_token_overfit_micro_render_20step/000020/model.safetensors"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/mmhead_debug/p9_2_abc_eval"))
    parser.add_argument("--split", choices=("val", "train", "holdout"), default="val")
    parser.add_argument("--holdout_ids", default="083", help="Comma-separated IDs for holdout train-style no-grad evaluation.")
    parser.add_argument("--train_ids", default="030,037,038,069,070", help="Comma-separated adapter-training IDs for disjointness/reporting checks.")
    parser.add_argument("--eval_protocol", choices=("native_val", "holdout_trainstyle", "holdout_trainstyle_no_val_exclusion"), default=None)
    parser.add_argument("--batch_idx", "--batch_index", dest="batch_idx", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--save_images", dest="save_images", action="store_true", default=True)
    parser.add_argument("--no_save_images", dest="save_images", action="store_false")
    parser.add_argument("--max_save_frames", type=int, default=4)
    parser.add_argument("--debug_output_shapes", action="store_true")
    parser.add_argument("--allow_fallback_to_train", action="store_true", help="Allow an explicit val->train fallback when the requested split is empty.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo = Path.cwd()
    output_dir = resolve_path(args.output_dir, repo)
    base_config = resolve_path(args.base_config, repo)
    local_paths_config = resolve_path(args.local_paths_config, repo)
    adapter_ckpt = resolve_path(args.adapter_ckpt, repo) if args.adapter_ckpt else None
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    base_cfg = load_yaml(base_config)
    paths = resolve_local_paths(base_cfg, repo, local_paths_config)
    train_ids = parse_id_list(args.train_ids)
    holdout_ids = parse_id_list(args.holdout_ids)
    overlap = sorted(set(train_ids) & set(holdout_ids))
    if overlap:
        raise RuntimeError(f"holdout_ids must be disjoint from train_ids; overlap={overlap}")
    holdout_train_ids_disjoint = True
    requested_protocol = args.eval_protocol or ("holdout_trainstyle_no_val_exclusion" if args.split == "holdout" else "native_val")
    if args.split == "holdout" and requested_protocol not in ("holdout_trainstyle", "holdout_trainstyle_no_val_exclusion"):
        raise RuntimeError("--split holdout requires --eval_protocol holdout_trainstyle_no_val_exclusion or an omitted --eval_protocol")
    if requested_protocol in ("holdout_trainstyle", "holdout_trainstyle_no_val_exclusion") and args.split != "holdout":
        raise RuntimeError("holdout train-style eval protocols require --split holdout")
    if requested_protocol == "holdout_trainstyle":
        print("[P9.2ABC-EVAL][WARN] eval_protocol=holdout_trainstyle is an alias; using holdout_trainstyle_no_val_exclusion.")
        requested_protocol = "holdout_trainstyle_no_val_exclusion"

    val_id_diagnostics: dict[str, dict[str, int]] = {}
    selected_ids: list[str]
    val_id = ""
    if requested_protocol == "holdout_trainstyle_no_val_exclusion":
        if not holdout_ids:
            raise RuntimeError("--split holdout requires at least one --holdout_ids value")
        prefer_ids = train_ids + [uid for uid in holdout_ids if uid not in set(train_ids)]
        generate_metadata(base_config, paths, prefer_ids=prefer_ids, max_ids=max(6, len(prefer_ids)))
        generated_meta = Path(paths["generated_meta"])
        holdout_meta = output_dir / "runtime_configs" / "holdout_eval_mixed_uids.json"
        holdout_meta, selected_ids, holdout_item_count = filter_metadata_by_ids(generated_meta, holdout_meta, holdout_ids)
        runtime_config = write_runtime_config_with_meta(
            base_config,
            base_cfg,
            Path(paths["root_dir"]),
            holdout_meta,
            val_ids=[],
            output_dir=output_dir,
            suffix="holdout_trainstyle_no_val_exclusion_resolved",
        )
        cfg = load_yaml(runtime_config)
        loader = build_loader(cfg, "train")
        actual_split = "holdout"
        holdout_dataset_len = len(loader.dataset)
        nersemble_val_id = list(cfg.dataset.datasets.nersemble.val_id)
        val_num = int(cfg.dataset.val_num)
        validation_exclusion_disabled = val_num == 0 and len(nersemble_val_id) == 0
        split_counts = {"holdout": holdout_dataset_len, "train_style_loader": holdout_dataset_len, "native_val": 0}
        if not validation_exclusion_disabled:
            raise RuntimeError(
                f"Holdout train-style evaluation must disable validation exclusion, but got "
                f"dataset.val_num={val_num}, nersemble.val_id={nersemble_val_id}"
            )
        if holdout_item_count <= 0:
            raise RuntimeError(f"Holdout metadata is empty. holdout_ids={holdout_ids}, holdout_meta={holdout_meta}")
        if holdout_dataset_len <= 0:
            raise RuntimeError(
                f"Holdout train-style evaluation produced zero samples. holdout_ids={holdout_ids}, "
                f"holdout_meta={holdout_meta}, holdout_item_count={holdout_item_count}, "
                f"dataset.val_num={val_num}, nersemble.val_id={nersemble_val_id}, "
                f"validation_exclusion_disabled={validation_exclusion_disabled}"
            )
        print("[P9.2ABC-EVAL] eval_protocol=holdout_trainstyle_no_val_exclusion")
        print(f"[P9.2ABC-EVAL] holdout_ids={holdout_ids}")
        print(f"[P9.2ABC-EVAL] holdout_metadata_path={holdout_meta}")
        print(f"[P9.2ABC-EVAL] holdout_item_count={holdout_item_count}")
        print(f"[P9.2ABC-EVAL] dataset.val_num={val_num}")
        print(f"[P9.2ABC-EVAL] nersemble.val_id={nersemble_val_id}")
        print(f"[P9.2ABC-EVAL] holdout_dataset_len={holdout_dataset_len}")
        print(f"[P9.2ABC-EVAL] validation_exclusion_disabled={validation_exclusion_disabled}")
        print(f"[P9.2ABC-EVAL] train_ids_used_for_adapter={train_ids}")
        print(f"[P9.2ABC-EVAL] eval_count={holdout_dataset_len}")
    else:
        generate_metadata(base_config, paths)
        val_id, selected_ids = choose_val_id(Path(paths["generated_meta"]), str(paths["preferred_val_id"]))
        runtime_config = write_resolved_config(base_config, base_cfg, paths, val_id, output_dir)
        cfg = load_yaml(runtime_config)
        train_loader = build_loader(cfg, "train")
        val_loader = build_loader(cfg, "val")
        loaders = {"train": train_loader, "val": val_loader}
        split_counts = {"train": len(train_loader.dataset), "val": len(val_loader.dataset)}
        print("[P9.2ABC-EVAL] eval_protocol=native_val")
        print(f"[P9.2ABC-EVAL] split_counts={split_counts} requested_split={args.split}")
        actual_split = args.split
        loader = loaders[actual_split]
        if len(loader.dataset) == 0:
            error = (
                f"Requested split={args.split} has zero samples. selected_ids={selected_ids}, "
                f"val_id={val_id}, train_count={split_counts['train']}, val_count={split_counts['val']}."
            )
            if args.split == "val" and args.allow_fallback_to_train:
                print(f"[P9.2ABC-EVAL][WARN] {error} Explicit --allow_fallback_to_train enabled; using train split.")
                actual_split = "train"
                loader = train_loader
            else:
                raise RuntimeError(error + " Refusing silent fallback; pass --allow_fallback_to_train to evaluate train intentionally.")
        if len(loader.dataset) == 0:
            raise RuntimeError(
                f"No samples available for same-batch evaluation after split resolution. "
                f"selected_ids={selected_ids}, val_id={val_id}, split_counts={split_counts}"
            )

    print(f"[P9.2ABC-EVAL] runtime_config={runtime_config}")
    print(f"[P9.2ABC-EVAL] selected_ids={selected_ids} val_id={val_id}")
    print(f"[P9.2ABC-EVAL] device={device}")

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
            f"Could not fetch any batches for batch_idx={args.batch_idx}, "
            f"num_batches={args.num_batches}; split={actual_split}; dataset size={len(loader.dataset)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    grid_dir = output_dir / "grids"
    variants = list(VARIANTS)
    per_batch_results: dict[int, dict[str, Any]] = {idx: {} for idx, _ in selected_batches}
    batch_motion_summaries: dict[int, dict[str, Any]] = {}
    grid_frames_by_batch: dict[int, dict[str, torch.Tensor]] = {idx: {} for idx, _ in selected_batches}
    variant_configs: dict[str, Any] = {}

    if args.save_images:
        from torchvision.utils import save_image

        for batch_idx, batch_cpu in selected_batches:
            gt_frame = first_gt_image_frame_from_batch(batch_cpu)
            if gt_frame is None:
                print(f"[P9.2ABC-EVAL][WARN] No GT/target RGB image found for batch={batch_idx}; available batch shapes follow.")
                print_batch_debug_shapes(batch_cpu, prefix=f"batch{batch_idx}")
            else:
                gt_path = image_dir / f"batch_{batch_idx:03d}_GT.png"
                gt_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(gt_frame, gt_path)
                grid_frames_by_batch[batch_idx]["GT"] = gt_frame

    for name in variants:
        ckpt_for_variant = adapter_ckpt if name == "C_zero_gt_token_trained_adapter" and adapter_ckpt and adapter_ckpt.exists() else None
        variant_cfg = make_variant_config(cfg, name, ckpt_for_variant)
        model, adapter_loaded = load_model_for_variant(variant_cfg, ckpt_for_variant, device)
        variant_configs[name] = {
            "use_motion_token": bool(variant_cfg.model.use_motion_token),
            "zero_flame_motion": bool(variant_cfg.model.zero_flame_motion),
            "motion_token_source": str(variant_cfg.model.motion_token_source),
            "adapter_loaded": adapter_loaded,
            "random_adapter": name == "D_zero_gt_token_random_adapter",
            "label": VARIANT_LABELS.get(name, name),
        }
        for batch_idx, batch_cpu in selected_batches:
            batch = move_to_device(batch_cpu, device)
            if batch_idx not in batch_motion_summaries:
                batch_motion_summaries[batch_idx] = flame_motion_summary(batch)
            with torch.no_grad():
                outputs, losses = forward_and_loss(model, variant_cfg, batch, device)
            if args.debug_output_shapes or os.environ.get("FASTAVATAR_TOKEN_DEBUG", "0") == "1":
                print_output_debug_shapes(outputs, prefix=f"outputs.{name}.batch{batch_idx}")
            images: list[str] = []
            if args.save_images:
                short = VARIANT_SHORT.get(name, name[:1])
                images = save_first_render_image(outputs, image_dir / f"batch_{batch_idx:03d}_{short}.png")
                first_frame = first_image_frame_from_outputs(outputs)
                if first_frame is not None:
                    grid_frames_by_batch[batch_idx][name] = first_frame
            per_batch_results[batch_idx][name] = {
                "config": variant_configs[name],
                "adapter_loaded": adapter_loaded,
                "random_adapter": name == "D_zero_gt_token_random_adapter",
                "losses": losses,
                "images": images,
            }
            print(
                f"[P9.2ABC-EVAL] batch={batch_idx} {name}: "
                f"losses={losses} adapter_loaded={adapter_loaded} images={images}"
            )
            del outputs, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    image_grids: dict[int, str | None] = {}
    if args.save_images:
        for batch_idx, frames in grid_frames_by_batch.items():
            image_grids[batch_idx] = save_labeled_eval_grid(frames, grid_dir / f"batch_{batch_idx:03d}_grid.png")

    batch_identities = [extract_batch_identity(batch, actual_split, requested_protocol, idx) for idx, batch in selected_batches]
    identity_json_path, identity_csv_path = write_batch_identity_files(batch_identities, output_dir)
    aggregate = compute_aggregate(per_batch_results)
    csv_path = write_per_batch_csv(per_batch_results, output_dir)

    metrics = {
        "base_config": base_config,
        "runtime_config": runtime_config,
        "output_dir": output_dir,
        "selected_ids": selected_ids,
        "val_id": val_id,
        "eval_protocol": requested_protocol,
        "requested_split": args.split,
        "split": actual_split,
        "train_ids_used_for_adapter": train_ids,
        "holdout_ids": holdout_ids,
        "holdout_train_ids_disjoint": holdout_train_ids_disjoint,
        "allow_fallback_to_train": bool(args.allow_fallback_to_train),
        "split_counts": split_counts,
        "validation_exclusion_disabled": locals().get("validation_exclusion_disabled", None),
        "val_id_diagnostics": val_id_diagnostics,
        "batch_idx": args.batch_idx,
        "num_batches": args.num_batches,
        "evaluated_batch_indices": [idx for idx, _ in selected_batches],
        "adapter_ckpt": adapter_ckpt,
        "adapter_ckpt_exists": bool(adapter_ckpt and adapter_ckpt.exists()),
        "batch_motion_summaries": batch_motion_summaries,
        "batch_identities": batch_identities,
        "batch_identity_json": identity_json_path,
        "batch_identity_csv": identity_csv_path,
        "variant_configs": variant_configs,
        "per_batch_results": per_batch_results,
        "aggregate": aggregate,
        "per_batch_csv": csv_path,
        "image_grids": image_grids,
    }
    write_report(metrics, output_dir)
    print(f"[P9.2ABC-EVAL] wrote metrics: {output_dir / 'metrics.json'}")
    print(f"[P9.2ABC-EVAL] wrote per-batch CSV: {csv_path}")
    print(f"[P9.2ABC-EVAL] wrote report: {output_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
