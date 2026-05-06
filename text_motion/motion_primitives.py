from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def smoothstep(t: np.ndarray) -> np.ndarray:
    """Cubic smoothstep in [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hold_curve(num_frames: int, value: float = 1.0) -> np.ndarray:
    """Constant curve."""
    return np.full((num_frames,), float(value), dtype=np.float32)


def sinusoidal_blink_curve(num_frames: int, center: Optional[int] = None, width: int = 8, amplitude: float = 1.0) -> np.ndarray:
    """Single smooth pulse; useful for blink-like temporal modulation."""
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    center = num_frames // 2 if center is None else int(center)
    start = max(0, center - width // 2)
    end = min(num_frames, start + width)
    curve = np.zeros((num_frames,), dtype=np.float32)
    n = max(1, end - start)
    x = np.linspace(0.0, np.pi, n, dtype=np.float32)
    curve[start:end] = (np.sin(x) ** 2) * float(amplitude)
    return curve


def _ramp_curve(num_frames: int, peak: float = 1.0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    return smoothstep(x) * float(peak)


@dataclass
class PrimitiveResult:
    updates: Dict[str, np.ndarray]
    info: str


def neutral(motion: Dict[str, np.ndarray], num_frames: int) -> PrimitiveResult:
    return PrimitiveResult(updates={}, info="neutral: no fields modified")


def turn_head_left(motion: Dict[str, np.ndarray], num_frames: int, pose_field: str, pose_dim: int = 3, yaw_index: int = 1, amplitude: float = 0.20) -> PrimitiveResult:
    arr = motion[pose_field].copy()
    curve = _ramp_curve(num_frames, peak=amplitude)
    arr[:num_frames, yaw_index] += curve
    return PrimitiveResult({pose_field: arr}, f"turn_head_left on {pose_field}[...,{yaw_index}]")


def turn_head_right(motion: Dict[str, np.ndarray], num_frames: int, pose_field: str, pose_dim: int = 3, yaw_index: int = 1, amplitude: float = 0.20) -> PrimitiveResult:
    arr = motion[pose_field].copy()
    curve = _ramp_curve(num_frames, peak=amplitude)
    arr[:num_frames, yaw_index] -= curve
    return PrimitiveResult({pose_field: arr}, f"turn_head_right on {pose_field}[...,{yaw_index}]")


def nod(motion: Dict[str, np.ndarray], num_frames: int, pose_field: str, pitch_index: int = 0, amplitude: float = 0.15, cycles: float = 1.5) -> PrimitiveResult:
    arr = motion[pose_field].copy()
    t = np.linspace(0.0, 2.0 * np.pi * cycles, num_frames, dtype=np.float32)
    env = smoothstep(np.linspace(0.0, 1.0, num_frames, dtype=np.float32))
    arr[:num_frames, pitch_index] += np.sin(t) * env * float(amplitude)
    return PrimitiveResult({pose_field: arr}, f"nod on {pose_field}[...,{pitch_index}]")


def open_mouth(motion: Dict[str, np.ndarray], num_frames: int, jaw_field: str, jaw_index: int = 0, amplitude: float = 0.25) -> PrimitiveResult:
    arr = motion[jaw_field].copy()
    curve = _ramp_curve(num_frames, peak=amplitude)
    arr[:num_frames, jaw_index] += curve
    return PrimitiveResult({jaw_field: arr}, f"open_mouth on {jaw_field}[...,{jaw_index}]")


def smile_from_exemplar_delta(
    motion: Dict[str, np.ndarray],
    num_frames: int,
    expr_field: str,
    neutral_slice: slice,
    smile_slice: slice,
    alpha: float = 0.5,
) -> PrimitiveResult:
    expr = motion[expr_field].copy()
    neutral_mean = expr[neutral_slice].mean(axis=0)
    smile_mean = expr[smile_slice].mean(axis=0)
    delta = smile_mean - neutral_mean
    curve = _ramp_curve(num_frames, peak=alpha)[:, None]
    expr[:num_frames] = neutral_mean[None, :] + curve * delta[None, :]
    return PrimitiveResult({expr_field: expr}, f"smile_from_exemplar_delta using {expr_field}")
