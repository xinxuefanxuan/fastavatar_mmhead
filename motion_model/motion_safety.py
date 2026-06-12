#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SAFETY_CONFIG: dict[str, float | int | None] = {
    "max_abs_yaw": 0.12,
    "max_abs_pitch": 0.10,
    "max_abs_roll": 0.10,
    "max_head_step": 0.025,
    "head_smooth_window": 3,
    "max_expr_norm": 4.0,
    "max_jaw": 0.35,
    "max_jaw_step": 0.05,
    "turn_left_scale": 1.0,
    "turn_right_scale": 0.8,
    "smile_gain": 1.0,
    "open_mouth_gain": 1.0,
}


def load_safety_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(DEFAULT_SAFETY_CONFIG)
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        print(f"[WARN] motion safety config not found: {p}; using defaults")
        return cfg
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = _parse_simple_yaml(text)
    for k, v in data.items():
        if k in cfg:
            cfg[k] = v
    return cfg


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not v:
            continue
        if v.lower() in {"null", "none"}:
            data[k] = None
        elif v.lower() in {"true", "false"}:
            data[k] = v.lower() == "true"
        else:
            try:
                data[k] = int(v) if v.isdigit() or (v.startswith("-") and v[1:].isdigit()) else float(v)
            except ValueError:
                data[k] = v.strip('"\'')
    return data


def motion_stats(motion: np.ndarray) -> dict[str, Any]:
    m = ensure_motion56(motion)
    expr = m[:, :50]
    head = m[:, 50:53]
    jaw = m[:, 53:56]
    expr_norm = np.linalg.norm(expr, axis=1)
    head_norm = np.linalg.norm(head, axis=1)
    jaw_norm = np.linalg.norm(jaw, axis=1)
    head_step = np.linalg.norm(np.diff(head, axis=0), axis=1) if len(head) > 1 else np.zeros((0,), dtype=np.float32)
    expr_step = np.linalg.norm(np.diff(expr, axis=0), axis=1) if len(expr) > 1 else np.zeros((0,), dtype=np.float32)
    jaw_step = np.linalg.norm(np.diff(jaw, axis=0), axis=1) if len(jaw) > 1 else np.zeros((0,), dtype=np.float32)
    return {
        "num_frames": int(m.shape[0]),
        "expr_norm_mean": float(expr_norm.mean()) if len(expr_norm) else 0.0,
        "expr_norm_max": float(expr_norm.max()) if len(expr_norm) else 0.0,
        "head_norm_mean": float(head_norm.mean()) if len(head_norm) else 0.0,
        "head_norm_max": float(head_norm.max()) if len(head_norm) else 0.0,
        "jaw_norm_mean": float(jaw_norm.mean()) if len(jaw_norm) else 0.0,
        "jaw_norm_max": float(jaw_norm.max()) if len(jaw_norm) else 0.0,
        "jaw_abs_max": float(np.max(np.abs(jaw))) if jaw.size else 0.0,
        "pitch_min": float(head[:, 0].min()) if len(head) else 0.0,
        "pitch_max": float(head[:, 0].max()) if len(head) else 0.0,
        "yaw_min": float(head[:, 1].min()) if len(head) else 0.0,
        "yaw_max": float(head[:, 1].max()) if len(head) else 0.0,
        "roll_min": float(head[:, 2].min()) if len(head) else 0.0,
        "roll_max": float(head[:, 2].max()) if len(head) else 0.0,
        "head_step_mean": float(head_step.mean()) if len(head_step) else 0.0,
        "head_step_max": float(head_step.max()) if len(head_step) else 0.0,
        "expr_step_mean": float(expr_step.mean()) if len(expr_step) else 0.0,
        "expr_step_max": float(expr_step.max()) if len(expr_step) else 0.0,
        "jaw_step_mean": float(jaw_step.mean()) if len(jaw_step) else 0.0,
        "jaw_step_max": float(jaw_step.max()) if len(jaw_step) else 0.0,
    }


def ensure_motion56(motion: np.ndarray) -> np.ndarray:
    arr = np.asarray(motion, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] < 56:
        raise ValueError(f"expected motion with shape (T, >=56), got {arr.shape}")
    return arr[:, :56].astype(np.float32, copy=True)


def apply_smile_profile(motion: np.ndarray, smile_gain: float = 1.0, smile_profile: str = "default") -> np.ndarray:
    out = ensure_motion56(motion)
    profile_gain = {"default": 1.0, "strong": 1.5, "subtle": 0.7}[smile_profile]
    gain = float(smile_gain) * profile_gain
    if abs(gain - 1.0) < 1e-6:
        return out
    # Keep enhancement localized to expression and light jaw coupling. There is no stable
    # semantic mouth-expression index map in this prototype, so use the smile-active
    # expression direction estimated from the sequence itself instead of scaling head pose.
    expr = out[:, :50]
    neutral = np.median(expr, axis=0, keepdims=True)
    delta = expr - neutral
    strength = np.linalg.norm(delta, axis=1, keepdims=True)
    if float(strength.max()) < 1e-8:
        return out
    active = strength / (float(strength.max()) + 1e-8)
    out[:, :50] = neutral + delta * (1.0 + (gain - 1.0) * active)
    out[:, 53:56] *= 1.0 + 0.15 * (gain - 1.0) * active
    return out.astype(np.float32)


def apply_motion_safety(motion: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    before = ensure_motion56(motion)
    out = before.copy()
    report: dict[str, Any] = {"config": cfg, "before": motion_stats(before), "operations": []}

    head = out[:, 50:53]
    lims = [cfg.get("max_abs_pitch"), cfg.get("max_abs_yaw"), cfg.get("max_abs_roll")]
    for axis, lim in enumerate(lims):
        if lim is None:
            continue
        old = head[:, axis].copy()
        head[:, axis] = np.clip(head[:, axis], -float(lim), float(lim))
        if not np.allclose(old, head[:, axis]):
            report["operations"].append({"op": "head_abs_clamp", "axis": axis, "limit": float(lim), "changed_frames": int(np.sum(old != head[:, axis]))})

    max_head_step = cfg.get("max_head_step")
    if max_head_step is not None and float(max_head_step) > 0:
        changed = 0
        max_step = float(max_head_step)
        for i in range(1, len(head)):
            delta = head[i] - head[i - 1]
            norm = float(np.linalg.norm(delta))
            if norm > max_step:
                head[i] = head[i - 1] + delta * (max_step / (norm + 1e-8))
                changed += 1
        if changed:
            report["operations"].append({"op": "head_step_clamp", "limit": max_step, "changed_frames": changed})

    window = int(cfg.get("head_smooth_window") or 0)
    if window > 1:
        if window % 2 == 0:
            window += 1
        out[:, 50:53] = _moving_average(out[:, 50:53], window)
        report["operations"].append({"op": "head_smoothing", "window": window})

    jaw = out[:, 53:56]
    max_jaw = cfg.get("max_jaw")
    if max_jaw is not None:
        old = jaw.copy()
        jaw[:] = np.clip(jaw, -float(max_jaw), float(max_jaw))
        if not np.allclose(old, jaw):
            report["operations"].append({"op": "jaw_abs_clamp", "limit": float(max_jaw), "changed_values": int(np.sum(old != jaw))})

    max_jaw_step = cfg.get("max_jaw_step")
    if max_jaw_step is not None and float(max_jaw_step) > 0:
        changed = 0
        lim = float(max_jaw_step)
        for i in range(1, len(jaw)):
            delta = jaw[i] - jaw[i - 1]
            norm = float(np.linalg.norm(delta))
            if norm > lim:
                jaw[i] = jaw[i - 1] + delta * (lim / (norm + 1e-8))
                changed += 1
        if changed:
            report["operations"].append({"op": "jaw_step_clamp", "limit": lim, "changed_frames": changed})

    max_expr_norm = cfg.get("max_expr_norm")
    if max_expr_norm is not None and float(max_expr_norm) > 0:
        expr = out[:, :50]
        norms = np.linalg.norm(expr, axis=1, keepdims=True)
        scale = np.minimum(1.0, float(max_expr_norm) / np.maximum(norms, 1e-8))
        changed = int(np.sum(scale[:, 0] < 0.999999))
        if changed:
            out[:, :50] = expr * scale
            report["operations"].append({"op": "expr_norm_clip", "limit": float(max_expr_norm), "changed_frames": changed})

    report["after"] = motion_stats(out)
    report["changed_l2"] = float(np.linalg.norm(out - before))
    return out.astype(np.float32), report


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) <= 1:
        return x.copy()
    pad = window // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = padded[i : i + window].mean(axis=0)
    return out


def safety_flags(stats: dict[str, Any], cfg: dict[str, Any]) -> dict[str, bool]:
    return {
        "abs_yaw_exceeded": max(abs(float(stats["yaw_min"])), abs(float(stats["yaw_max"]))) > float(cfg.get("max_abs_yaw") or np.inf),
        "head_step_exceeded": float(stats["head_step_max"]) > float(cfg.get("max_head_step") or np.inf),
        "expr_norm_exceeded": float(stats["expr_norm_max"]) > float(cfg.get("max_expr_norm") or np.inf),
        "jaw_exceeded": float(stats["jaw_abs_max"]) > float(cfg.get("max_jaw") or np.inf),
        "jaw_step_exceeded": float(stats["jaw_step_max"]) > float(cfg.get("max_jaw_step") or np.inf),
    }
