#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from motion_model.channel_vae import ChannelTemporalVAE

HEAD_PRIMITIVES = {"turn_left", "turn_right", "look_up", "look_down", "tilt_left", "tilt_right", "nod"}
PRIMITIVE_SYNONYMS = {
    "turn_left": ["turn left", "look left", "face left"],
    "turn_right": ["turn right", "look right", "face right"],
    "look_up": ["look up", "head up", "raise head", "lift head"],
    "look_down": ["look down", "head down", "lower head"],
    "tilt_left": ["tilt left", "lean left"],
    "tilt_right": ["tilt right", "lean right"],
    "smile": ["smile", "happy", "grin"],
    "mouth_open": ["open mouth", "mouth open", "jaw open"],
}
SEQUENTIAL_SEPARATORS = ["and then", "then", "after", "next", "然后", "接着", "再"]
SLOW_WORDS = ["slowly", "gradually", "slow", "慢慢", "逐渐"]
QUICK_WORDS = ["quickly", "fast", "quick", "快速", "立刻"]
SLIGHT_WORDS = ["slightly", "subtle", "a little", "轻微"]
STRONG_WORDS = ["very", "strong", "exaggerated", "大幅"]
CHANNEL_SCALES = {
    "expr": {"smile": 1.0, "mouth_open": 0.5},
    "head": {"turn_left": 1.0, "turn_right": 1.0, "look_up": 1.0, "look_down": 1.0, "tilt_left": 1.0, "tilt_right": 1.0, "nod": 1.0},
    "jaw": {"mouth_open": 1.5, "smile": 0.3},
}


def load_cvae(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = ChannelTemporalVAE(int(ckpt["input_dim"]), int(ckpt["target_len"]), int(ckpt["latent_dim"]), int(ckpt["hidden_dim"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def contains_any(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def split_temporal(prompt: str) -> list[str]:
    text = prompt.lower().strip()
    pattern = "|".join(re.escape(x) for x in sorted(SEQUENTIAL_SEPARATORS, key=len, reverse=True))
    parts = [p.strip(" ,.;") for p in re.split(pattern, text) if p.strip(" ,.;")]
    return parts or [text]


def labels_from_text(text: str) -> list[str]:
    labels: list[str] = []
    for label, syns in PRIMITIVE_SYNONYMS.items():
        if any(s in text for s in syns):
            labels.append(label)
    if "neutral" in text or "still" in text:
        labels.append("neutral")
    return labels or ["neutral"]


def parse_prompt(prompt: str, args) -> list[dict]:
    segments: list[dict] = []
    carried: list[str] = []
    for part in split_temporal(prompt):
        labels = labels_from_text(part)
        if "neutral" in labels:
            active = ["neutral"]
            carried = []
        elif args.carry_mode == "accumulate":
            active = list(dict.fromkeys(carried + labels))
            carried = active
        else:
            active = labels

        if contains_any(part, SLOW_WORDS):
            ramp = args.slow_ramp_frames
        elif contains_any(part, QUICK_WORDS):
            ramp = args.quick_ramp_frames
        else:
            ramp = args.default_ramp_frames

        intensity = 1.0
        if contains_any(part, SLIGHT_WORDS):
            intensity = 0.7
        if contains_any(part, STRONG_WORDS):
            intensity = 1.3

        segments.append(
            {
                "text": part,
                "labels": active,
                "new_labels": labels,
                "intensity": intensity,
                "ramp_frames": int(ramp),
                "hold_frames": int(args.default_hold_frames),
                "release_frames": int(args.default_release_frames),
            }
        )
    return segments


def compose_latents(labels: list[str], intensity: float, protos: dict, device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for ch in ["expr", "head", "jaw"]:
        z0 = protos[ch]["neutral"].to(device)
        z = z0.clone()
        for lab in labels:
            if lab == "neutral":
                continue
            if lab not in protos[ch]:
                print(f"[WARN] label {lab} missing in {ch} prototypes; skipped")
                continue
            scale = float(CHANNEL_SCALES.get(ch, {}).get(lab, 0.0))
            z = z + scale * float(intensity) * (protos[ch][lab].to(device) - z0)
        out[ch] = z[None, :]
    return out


def motion_stats(m: np.ndarray) -> dict:
    expr = m[:, :50]
    head = m[:, 50:53]
    jaw = m[:, 53:56]
    expr_n = np.linalg.norm(expr, axis=1)
    head_n = np.linalg.norm(head, axis=1)
    jaw_n = np.linalg.norm(jaw, axis=1)
    return {
        "expr_norm_mean": float(expr_n.mean()),
        "expr_norm_max": float(expr_n.max()),
        "head_norm_mean": float(head_n.mean()),
        "head_norm_max": float(head_n.max()),
        "jaw_norm_mean": float(jaw_n.mean()),
        "jaw_norm_max": float(jaw_n.max()),
        "pitch_min": float(head[:, 0].min()),
        "pitch_max": float(head[:, 0].max()),
        "yaw_min": float(head[:, 1].min()),
        "yaw_max": float(head[:, 1].max()),
        "roll_min": float(head[:, 2].min()),
        "roll_max": float(head[:, 2].max()),
    }


def target_index(motion: np.ndarray, labels: list[str]) -> tuple[int, str]:
    if "mouth_open" in labels:
        scores = np.linalg.norm(motion[:, 53:56], axis=1)
        return int(np.argmax(scores)), "max_jaw"
    if "smile" in labels:
        scores = np.linalg.norm(motion[:, :50], axis=1)
        return int(np.argmax(scores)), "max_expr"
    if any(x in HEAD_PRIMITIVES for x in labels):
        scores = np.linalg.norm(motion[:, 50:53], axis=1)
        return int(np.argmax(scores)), "max_head"
    return 0, "neutral"


def envelope(ramp: int, hold: int, release: int, release_ratio: float) -> np.ndarray:
    parts: list[np.ndarray] = []
    if ramp > 0:
        parts.append(np.linspace(0.0, 1.0, ramp, endpoint=True, dtype=np.float32))
    if hold > 0:
        parts.append(np.ones((hold,), dtype=np.float32))
    if release > 0:
        parts.append(np.linspace(1.0, release_ratio, release, endpoint=True, dtype=np.float32))
    return np.concatenate(parts, axis=0) if parts else np.ones((1,), dtype=np.float32)


def fit_output_len(m: np.ndarray, output_len: int) -> np.ndarray:
    if m.shape[0] == output_len:
        return m
    if m.shape[0] > output_len:
        idx = np.linspace(0, m.shape[0] - 1, output_len).round().astype(np.int64)
        return m[idx]
    pad = np.repeat(m[-1:], output_len - m.shape[0], axis=0)
    return np.concatenate([m, pad], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--channel_prototypes", type=Path, default=Path("outputs/mmhead_debug/channel_prototypes_v3_extended/channel_prototypes.pt"))
    ap.add_argument("--expr_checkpoint", type=Path, default=Path("outputs/mmhead_debug/channel_vae_v1/expr/best.pt"))
    ap.add_argument("--head_checkpoint", type=Path, default=Path("outputs/mmhead_debug/channel_vae_v1/head/best.pt"))
    ap.add_argument("--jaw_checkpoint", type=Path, default=Path("outputs/mmhead_debug/channel_vae_v1/jaw/best.pt"))
    ap.add_argument("--norm_stats", type=Path, default=Path("outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json"))
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--output_plan_json", type=Path, default=None)
    ap.add_argument("--output_len", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--carry_mode", choices=["none", "accumulate"], default="accumulate")
    ap.add_argument("--default_ramp_frames", type=int, default=6)
    ap.add_argument("--default_hold_frames", type=int, default=10)
    ap.add_argument("--default_release_frames", type=int, default=4)
    ap.add_argument("--slow_ramp_frames", type=int, default=12)
    ap.add_argument("--quick_ramp_frames", type=int, default=3)
    ap.add_argument("--release_ratio", type=float, default=0.8)
    ap.add_argument("--motion_key", default="motion")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cp = torch.load(args.channel_prototypes, map_location="cpu")
    protos = {ch: {k: v.float() for k, v in cp["prototypes"][ch].items()} for ch in ["expr", "head", "jaw"]}
    expr_dec = load_cvae(args.expr_checkpoint, device)
    head_dec = load_cvae(args.head_checkpoint, device)
    jaw_dec = load_cvae(args.jaw_checkpoint, device)
    st = json.loads(args.norm_stats.read_text())
    mean = np.asarray(st["mean"], dtype=np.float32)[:56]
    std = np.asarray(st["std"], dtype=np.float32)[:56]

    segments = parse_prompt(args.prompt, args)
    segment_motions: list[np.ndarray] = []
    plan_segments: list[dict] = []
    print("prompt:", args.prompt)
    print("carry_mode:", args.carry_mode)
    print("parsed segments:", json.dumps(segments, ensure_ascii=False, indent=2))

    for si, seg in enumerate(segments):
        z = compose_latents(seg["labels"], seg["intensity"], protos, device)
        with torch.no_grad():
            expr = expr_dec.decode(z["expr"])[0].cpu().numpy().astype(np.float32)
            head = head_dec.decode(z["head"])[0].cpu().numpy().astype(np.float32)
            jaw = jaw_dec.decode(z["jaw"])[0].cpu().numpy().astype(np.float32)
        motion_norm_full = np.concatenate([expr, head, jaw], axis=1)
        motion_raw_full = motion_norm_full * std[None, :] + mean[None, :]
        idx, reason = target_index(motion_raw_full, seg["labels"])
        target_raw = motion_raw_full[idx]
        target_norm = motion_norm_full[idx]
        env = envelope(seg["ramp_frames"], seg["hold_frames"], seg["release_frames"], args.release_ratio)
        seg_raw = env[:, None] * target_raw[None, :]
        seg_norm = env[:, None] * target_norm[None, :]
        segment_motions.append(seg_raw.astype(np.float32))
        stats = motion_stats(seg_raw)
        plan_seg = {**seg, "segment_index": si, "target_frame": idx, "target_strategy": reason, "stats": stats, "num_frames": int(seg_raw.shape[0])}
        plan_segments.append(plan_seg)
        print(f"segment {si}: labels={seg['labels']} target_frame={idx} strategy={reason} stats={stats}")

    out_raw = np.concatenate(segment_motions, axis=0) if segment_motions else np.zeros((1, 56), dtype=np.float32)
    out_raw = fit_output_len(out_raw, args.output_len).astype(np.float32)
    out_norm = (out_raw - mean[None, :]) / np.maximum(std[None, :], 1e-8)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        motion=out_raw,
        motion_raw=out_raw,
        motion_norm=out_norm.astype(np.float32),
        expr_delta=out_raw[:, :50].astype(np.float32),
        head_delta=out_raw[:, 50:53].astype(np.float32),
        jaw_delta=out_raw[:, 53:56].astype(np.float32),
    )

    final_stats = motion_stats(out_raw)
    print("final motion stats:", final_stats)
    print("output:", args.output_npz)

    if args.output_plan_json:
        args.output_plan_json.parent.mkdir(parents=True, exist_ok=True)
        plan = {"prompt": args.prompt, "carry_mode": args.carry_mode, "segments": plan_segments, "output_npz": str(args.output_npz), "final_stats": final_stats}
        args.output_plan_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print("plan:", args.output_plan_json)


if __name__ == "__main__":
    main()
