#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_model.models import TemporalConvVAE
from motion_model.generate_composed_primitive import build_hold_sequence, print_stats

DEFAULT_WEIGHTS = {
    "turn_left": 1.8,
    "turn_right": 1.8,
    "smile": 2.5,
    "mouth_open": 2.0,
    "nod": 1.5,
    "neutral": 1.0,
}
DEFAULT_INTENSITY_MULTIPLIERS = {
    "slight": 0.7,
    "subtle": 0.7,
    "a little": 0.7,
    "very": 1.3,
    "strong": 1.5,
    "big": 1.5,
    "exaggerated": 1.5,
}

def parse_prompt_to_primitives(
    prompt: str,
    default_weights: dict[str, float],
    intensity_multipliers: dict[str, float],
) -> tuple[list[str], list[float], float]:
    p = prompt.lower()
    primitives: list[str] = []
    if any(x in p for x in ["turn left", "look left", " left"]):
        primitives.append("turn_left")
    if any(x in p for x in ["turn right", "look right", " right"]):
        primitives.append("turn_right")
    if any(x in p for x in ["nod", "nodding"]):
        primitives.append("nod")
    if any(x in p for x in ["smile", "happy", "grin"]):
        primitives.append("smile")
    if any(x in p for x in ["open mouth", "mouth open", "jaw"]):
        primitives.append("mouth_open")
    if any(x in p for x in ["neutral", "still"]):
        primitives.append("neutral")

    if not primitives:
        primitives = ["neutral"]

    mult = 1.0
    for token, factor in intensity_multipliers.items():
        if token in p:
            mult *= float(factor)

    weights = [float(default_weights.get(k, 1.0)) * mult for k in primitives]
    return primitives, weights, mult


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--prototype_path", type=Path, required=True)
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--target_len", type=int, default=64)
    ap.add_argument("--output_len", type=int, default=32)
    ap.add_argument("--temporal_mode", choices=["raw", "hold"], default="hold")
    ap.add_argument("--ramp_frames", type=int, default=10)
    ap.add_argument("--hold_frames", type=int, default=18)
    ap.add_argument("--release_frames", type=int, default=4)
    ap.add_argument("--release_ratio", type=float, default=0.75)
    ap.add_argument("--latent_scale", type=float, default=1.0)
    ap.add_argument("--noise_scale", type=float, default=0.0)
    ap.add_argument("--preset_config", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    default_weights = dict(DEFAULT_WEIGHTS)
    intensity_multipliers = dict(DEFAULT_INTENSITY_MULTIPLIERS)
    if args.preset_config is not None:
        cfg = json.loads(args.preset_config.read_text(encoding="utf-8"))
        default_weights.update(cfg.get("default_weights", {}))
        intensity_multipliers.update(cfg.get("intensity_multipliers", {}))
        print(f"loaded preset config: {args.preset_config}")

    primitives, weights, mult = parse_prompt_to_primitives(
        args.prompt,
        default_weights=default_weights,
        intensity_multipliers=intensity_multipliers,
    )

    prot = torch.load(args.prototype_path, map_location="cpu")
    label_map = prot["label_map"]
    if "neutral" not in label_map:
        raise SystemExit("neutral prototype is required")

    z_neutral = np.asarray(prot["label_to_mean_mu"]["neutral"], dtype=np.float32)
    z_comp = z_neutral.copy()
    for p, w in zip(primitives, weights):
        if p not in label_map:
            continue
        z_p = np.asarray(prot["label_to_mean_mu"][p], dtype=np.float32)
        z_comp = z_comp + float(w) * (z_p - z_neutral)

    z_comp = z_comp * float(args.latent_scale)
    std_ref = np.asarray(prot["label_to_std_mu"].get("neutral", np.ones_like(z_comp)), dtype=np.float32)
    if args.noise_scale > 0:
        z_comp = z_comp + float(args.noise_scale) * std_ref * np.random.randn(*z_comp.shape).astype(np.float32)

    ckpt = torch.load(args.vae_checkpoint, map_location=args.device)
    latent_dim = int(ckpt.get("args", {}).get("latent_dim", 64))
    vae = TemporalConvVAE(in_dim=56, latent_dim=latent_dim).to(args.device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()

    z = torch.from_numpy(z_comp[None, None, :]).to(args.device).repeat(1, args.target_len, 1)
    with torch.no_grad():
        pred_norm = vae.decode(z.transpose(1, 2))[0].cpu().numpy().astype(np.float32)

    stats = json.loads(args.norm_stats.read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    motion_raw = pred_norm * std[None, :] + mean[None, :]

    print("[Before temporal]")
    print_stats(motion_raw)
    if args.temporal_mode == "hold":
        motion_raw = build_hold_sequence(
            motion_raw, primitives, args.output_len, args.ramp_frames, args.hold_frames, args.release_frames, args.release_ratio
        )
    else:
        motion_raw = motion_raw[: args.output_len]
    pred_norm = (motion_raw - mean[None, :]) / std[None, :]

    out = {
        "motion": motion_raw.astype(np.float32),
        "motion_raw": motion_raw.astype(np.float32),
        "motion_norm": pred_norm.astype(np.float32),
        "expr_delta": motion_raw[:, 0:50].astype(np.float32),
        "head_delta": motion_raw[:, 50:53].astype(np.float32),
        "jaw_delta": motion_raw[:, 53:56].astype(np.float32),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **out)

    print(f"prompt={args.prompt}")
    print(f"parsed_primitives={primitives}")
    print(f"weights={weights}")
    print(f"intensity_multiplier={mult}")
    print("[After temporal]")
    print_stats(motion_raw)
    print(f"saved text-rule npz: {args.output_npz}")


if __name__ == "__main__":
    main()
