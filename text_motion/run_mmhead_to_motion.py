#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import yaml
except Exception as e:
    raise SystemExit(f"PyYAML is required: {e}")


def copy_template(src: Path, dst: Path, dry_run: bool) -> None:
    if src.resolve() == dst.resolve():
        raise SystemExit("output_motion cannot equal template_motion")
    if dst.exists():
        raise SystemExit(f"output_motion already exists: {dst}")
    print(f"[Copy template]\n  from: {src}\n  to:   {dst}")
    if not dry_run:
        shutil.copytree(src, dst)


def detect_motion_paths(root: Path) -> Tuple[Path, Path, list[Path]]:
    transforms = root / "transforms.json"
    if not transforms.exists():
        raise SystemExit(f"transforms.json not found under {root}")

    flame_dir = root / "flame_param"
    if not flame_dir.exists():
        raise SystemExit(f"flame_param directory not found under {root}")

    frame_npz = sorted(flame_dir.glob("*.npz"))
    if not frame_npz:
        raise SystemExit(f"no frame npz found under {flame_dir}")

    return transforms, flame_dir, frame_npz


def load_frame_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_frame_npz(npz_path: Path, payload: Dict[str, np.ndarray], dry_run: bool) -> None:
    if dry_run:
        return
    np.savez(npz_path, **payload)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def parse_int_list(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 comma-separated integers, got: {s}")
    return vals


def parse_float_list(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 comma-separated floats, got: {s}")
    return vals


def to_2d(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Convert FastAvatar stacked sequence to (T,D).

    Supported:
      (T,D)
      (T,1,D)
    """
    if arr.ndim == 2:
        return arr, False
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr[:, 0, :], True
    raise ValueError(f"Unsupported sequence shape: {arr.shape}, expected (T,D) or (T,1,D)")


def from_2d(arr2d: np.ndarray, was_3d: bool) -> np.ndarray:
    if was_3d:
        return arr2d[:, None, :]
    return arr2d


def resample_to_length(x: np.ndarray, target_t: int) -> np.ndarray:
    """
    Linear interpolation from (T,D) to (target_t,D).
    """
    if x.ndim != 2:
        raise ValueError(f"resample_to_length expects (T,D), got {x.shape}")

    src_t, dim = x.shape
    if src_t == target_t:
        return x.copy()

    if src_t <= 1:
        return np.repeat(x[:1], target_t, axis=0)

    old = np.linspace(0.0, 1.0, src_t, dtype=np.float32)
    new = np.linspace(0.0, 1.0, target_t, dtype=np.float32)

    out = np.zeros((target_t, dim), dtype=x.dtype)
    for d in range(dim):
        out[:, d] = np.interp(new, old, x[:, d])
    return out


def apply_axis_transform(x: np.ndarray, order: list[int], signs: list[float]) -> np.ndarray:
    """
    x: (T,3)
    order/signs are used to calibrate axis convention between MMHead and FastAvatar.

    Example:
      order=0,1,2 signs=1,1,1: unchanged
      order=0,1,2 signs=1,-1,1: flip second axis
    """
    if x.shape[1] != 3:
        raise ValueError(f"axis transform expects D=3, got {x.shape}")
    y = x[:, order].copy()
    y = y * np.asarray(signs, dtype=y.dtype)[None, :]
    return y


# def delta_retarget(
#     base_seq: np.ndarray,
#     mm_seq: np.ndarray,
#     n_modify: int,
#     scale: float,
#     ref_n: int = 5,
# ) -> np.ndarray:
#     """
#     base_seq: FastAvatar sequence, shape (T,D) or (T,1,D)
#     mm_seq: MMHead sequence, shape (T,D)

#     Only first n_modify frames are modified. Remaining frames keep original base values.

#     target = base_ref + scale * (mm_t - mm_ref)
#     """
#     base2d, was_3d = to_2d(base_seq)
#     total_t, dim = base2d.shape

#     n = min(int(n_modify), total_t)
#     if n <= 0:
#         return base_seq.copy()

#     if mm_seq.ndim != 2:
#         raise ValueError(f"mm_seq must be (T,D), got {mm_seq.shape}")

#     if mm_seq.shape[1] != dim:
#         raise ValueError(
#             f"Dimension mismatch: base dim={dim}, mm dim={mm_seq.shape[1]}. "
#             f"base shape={base_seq.shape}, mm shape={mm_seq.shape}"
#         )

#     mm_rs = resample_to_length(mm_seq.astype(base2d.dtype), n)

#     r = min(int(ref_n), n, mm_rs.shape[0])
#     base_ref = base2d[:r].mean(axis=0, keepdims=True)
#     mm_ref = mm_rs[:r].mean(axis=0, keepdims=True)

#     out2d = base2d.copy()
#     out2d[:n] = base_ref + float(scale) * (mm_rs - mm_ref)

#     return from_2d(out2d, was_3d)

def delta_retarget(
    base_seq: np.ndarray,
    mm_seq: np.ndarray,
    n_modify: int,
    scale: float,
    ref_n: int = 5,
) -> np.ndarray:
    """
    base_seq: FastAvatar sequence, shape (T,D) or (T,1,D)
    mm_seq: MMHead sequence, shape (T,D_mm)

    Only first n_modify frames are modified. Remaining frames keep original base values.

    If D_mm != D_base:
      modify only the first min(D_mm, D_base) channels;
      keep other base channels unchanged.

    target = base_ref + scale * (mm_t - mm_ref)
    """
    base2d, was_3d = to_2d(base_seq)
    total_t, base_dim = base2d.shape

    n = min(int(n_modify), total_t)
    if n <= 0:
        return base_seq.copy()

    if mm_seq.ndim != 2:
        raise ValueError(f"mm_seq must be (T,D), got {mm_seq.shape}")

    mm_dim = mm_seq.shape[1]
    common_dim = min(base_dim, mm_dim)

    if common_dim <= 0:
        raise ValueError(
            f"No common dimension to retarget: base_dim={base_dim}, mm_dim={mm_dim}"
        )

    if base_dim != mm_dim:
        print(
            f"[WARN] Dimension mismatch: base_dim={base_dim}, mm_dim={mm_dim}. "
            f"Will retarget only first {common_dim} channels and keep remaining channels unchanged."
        )

    mm_rs = resample_to_length(mm_seq.astype(base2d.dtype), n)

    r = min(int(ref_n), n, mm_rs.shape[0])

    out2d = base2d.copy()

    base_ref = base2d[:r, :common_dim].mean(axis=0, keepdims=True)
    mm_ref = mm_rs[:r, :common_dim].mean(axis=0, keepdims=True)

    # out2d[:n, :common_dim] = (
    #     base_ref + float(scale) * (mm_rs[:, :common_dim] - mm_ref)
    # )
    out2d[:n, :common_dim] = (
        base2d[:n, :common_dim]
        + float(scale) * (mm_rs[:, :common_dim] - mm_ref)
    )

    return from_2d(out2d, was_3d)

def stack_payloads(frame_payloads: list[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = set(frame_payloads[0].keys())
    seq: Dict[str, np.ndarray] = {}
    for k in keys:
        seq[k] = np.stack([p[k] for p in frame_payloads], axis=0)
    return seq


def write_updates_back(
    frame_payloads: list[Dict[str, np.ndarray]],
    updates: Dict[str, np.ndarray],
) -> None:
    for i, payload in enumerate(frame_payloads):
        for k, arr in updates.items():
            payload[k] = arr[i]


def _to_2d_float32(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    elif a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D after reshape, got {a.shape}")
    return a


def load_mmhead_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"PKL motion must be dict-like, got {type(obj)}")

        # normalized pkl schema fallback
        if all(k in obj for k in ("expression", "head_pose", "jaw_pose")):
            expr = _to_2d_float32(obj["expression"], "expression")
            head = _to_2d_float32(obj["head_pose"], "head_pose")
            jaw = _to_2d_float32(obj["jaw_pose"], "jaw_pose")
            print("[MMHead normalized pkl]")
            print(f"  expression: {expr.shape}")
            print(f"  head_pose:  {head.shape}")
            print(f"  jaw_pose:   {jaw.shape}")
        elif "expcodes" in obj and "posecodes" in obj:
            expr = _to_2d_float32(obj["expcodes"], "expcodes")
            pose = _to_2d_float32(obj["posecodes"], "posecodes")
            if pose.shape[1] >= 6:
                head = pose[:, 0:3].astype(np.float32, copy=False)
                jaw = pose[:, 3:6].astype(np.float32, copy=False)
                print("[MMHead native pkl]")
                print(f"  expcodes -> expression: {expr.shape}")
                print(f"  posecodes[:, 0:3] -> head_pose: {head.shape}")
                print(f"  posecodes[:, 3:6] -> jaw_pose:  {jaw.shape}")
            elif pose.shape[1] == 3:
                head = pose[:, 0:3].astype(np.float32, copy=False)
                jaw = np.zeros_like(head, dtype=np.float32)
                print("[MMHead native pkl]")
                print(f"  expcodes -> expression: {expr.shape}")
                print(f"  posecodes[:, 0:3] -> head_pose: {head.shape}")
                print(f"  [WARN] posecodes has only 3 channels; jaw_pose is zeros: {jaw.shape}")
            else:
                raise ValueError(
                    f"posecodes must have at least 3 channels, got shape={pose.shape}"
                )
        else:
            raise KeyError(
                f"{path} unsupported pkl keys. expected normalized keys "
                f"(expression/head_pose/jaw_pose) or native keys (expcodes/posecodes). "
                f"keys={list(obj.keys())}"
            )
    else:
        data = np.load(path, allow_pickle=True)

        required = ["expression", "head_pose", "jaw_pose"]
        for k in required:
            if k not in data.files:
                raise KeyError(f"{path} missing required key: {k}. keys={data.files}")

        expr = np.asarray(data["expression"], dtype=np.float32)
        head = np.asarray(data["head_pose"], dtype=np.float32)
        jaw = np.asarray(data["jaw_pose"], dtype=np.float32)

    expr = _to_2d_float32(expr, "expression")
    head = _to_2d_float32(head, "head_pose")
    jaw = _to_2d_float32(jaw, "jaw_pose")

    if expr.shape[1] != 50:
        raise ValueError(f"Bad expression shape: {expr.shape}, expected (T,50)")
    if head.shape[1] != 3:
        raise ValueError(f"Bad head_pose shape: {head.shape}, expected (T,3)")
    if jaw.shape[1] != 3:
        raise ValueError(f"Bad jaw_pose shape: {jaw.shape}, expected (T,3)")

    return expr, head, jaw


def print_seq_stats(name: str, arr: np.ndarray) -> None:
    arr2d, _ = to_2d(arr) if arr.ndim in (2, 3) else (arr, False)
    print(
        f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr2d.min():.6f}, max={arr2d.max():.6f}, "
        f"mean={arr2d.mean():.6f}, std={arr2d.std():.6f}"
    )


def apply_mmhead_motion_to_sequence(
    cfg: dict,
    frame_payloads: list[Dict[str, np.ndarray]],
    mmhead_npz: Path,
    mode: str,
    num_frames: int,
    expr_scale: float,
    head_scale: float,
    jaw_scale: float,
    ref_n: int,
    head_axis_order: list[int],
    head_axis_signs: list[float],
    jaw_axis_order: list[int],
    jaw_axis_signs: list[float],
    dry_run: bool,
) -> None:
    fields = cfg.get("field_mapping", {})

    expr_field = fields.get("expression_field")
    head_field = fields.get("head_pose_field")
    jaw_field = fields.get("jaw_pose_field")

    seq = stack_payloads(frame_payloads)
    keys = set(seq.keys())

    print(f"[Detected frame npz keys] {sorted(keys)}")
    print("[Field mapping]")
    print(f"  expression_field: {expr_field}")
    print(f"  head_pose_field:  {head_field}")
    print(f"  jaw_pose_field:   {jaw_field}")

    def require_field(logical_name: str, field: str | None) -> str:
        if not field:
            raise SystemExit(f"Missing field mapping for {logical_name}")
        if field not in keys:
            raise SystemExit(
                f"Mapped field '{field}' for {logical_name} not found. "
                f"Available keys: {sorted(keys)}"
            )
        return field

    mm_expr, mm_head, mm_jaw = load_mmhead_npz(mmhead_npz)

    mm_head = apply_axis_transform(mm_head, head_axis_order, head_axis_signs)
    mm_jaw = apply_axis_transform(mm_jaw, jaw_axis_order, jaw_axis_signs)

    n = min(int(num_frames), len(frame_payloads))
    print(f"[MMHead source] {mmhead_npz}")
    print(f"  mm_expr: {mm_expr.shape}")
    print(f"  mm_head: {mm_head.shape}, order={head_axis_order}, signs={head_axis_signs}")
    print(f"  mm_jaw:  {mm_jaw.shape}, order={jaw_axis_order}, signs={jaw_axis_signs}")
    print(f"  modify frames: {n}/{len(frame_payloads)}")
    print(f"  mode={mode}, expr_scale={expr_scale}, head_scale={head_scale}, jaw_scale={jaw_scale}, ref_n={ref_n}")

    updates: Dict[str, np.ndarray] = {}

    if mode in ("all", "expr_only"):
        ef = require_field("expression_field", expr_field)
        updates[ef] = delta_retarget(
            base_seq=seq[ef],
            mm_seq=mm_expr,
            n_modify=n,
            scale=expr_scale,
            ref_n=ref_n,
        )

    if mode in ("all", "head_only"):
        hf = require_field("head_pose_field", head_field)
        updates[hf] = delta_retarget(
            base_seq=seq[hf],
            mm_seq=mm_head,
            n_modify=n,
            scale=head_scale,
            ref_n=ref_n,
        )

    if mode in ("all", "jaw_only"):
        jf = require_field("jaw_pose_field", jaw_field)
        updates[jf] = delta_retarget(
            base_seq=seq[jf],
            mm_seq=mm_jaw,
            n_modify=n,
            scale=jaw_scale,
            ref_n=ref_n,
        )

    print("[Updates]")
    for k, v in updates.items():
        print_seq_stats(k, v)

    if dry_run:
        print("[dry_run] no files written")
        return

    write_updates_back(frame_payloads, updates)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--template_motion", required=True, type=Path)
    ap.add_argument("--output_motion", required=True, type=Path)
    ap.add_argument("--mmhead_npz", required=True, type=Path)

    ap.add_argument("--mode", default="all", choices=["all", "expr_only", "head_only", "jaw_only"])
    ap.add_argument("--num_frames", type=int, default=90)

    ap.add_argument("--expr_scale", type=float, default=0.3)
    ap.add_argument("--head_scale", type=float, default=0.5)
    ap.add_argument("--jaw_scale", type=float, default=0.5)
    ap.add_argument("--ref_n", type=int, default=5)

    ap.add_argument("--head_axis_order", type=str, default="0,1,2")
    ap.add_argument("--head_axis_signs", type=str, default="1,1,1")
    ap.add_argument("--jaw_axis_order", type=str, default="0,1,2")
    ap.add_argument("--jaw_axis_signs", type=str, default="1,1,1")

    ap.add_argument("--primitive_config", type=Path, default=Path("text_motion/primitives.yaml"))
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    if not args.template_motion.exists():
        raise SystemExit(f"template_motion not found: {args.template_motion}")
    if not args.mmhead_npz.exists():
        raise SystemExit(f"mmhead_npz not found: {args.mmhead_npz}")
    if not args.primitive_config.exists():
        raise SystemExit(f"primitive_config not found: {args.primitive_config}")

    copy_template(args.template_motion, args.output_motion, args.dry_run)

    work_root = args.template_motion if args.dry_run else args.output_motion
    _, flame_dir, frame_npz_paths = detect_motion_paths(work_root)

    print(f"[Motion root] {work_root}")
    print(f"[Flame dir] {flame_dir}")
    print(f"[Frame count] {len(frame_npz_paths)}")

    cfg = load_config(args.primitive_config)
    frame_payloads = [load_frame_npz(p) for p in frame_npz_paths]

    apply_mmhead_motion_to_sequence(
        cfg=cfg,
        frame_payloads=frame_payloads,
        mmhead_npz=args.mmhead_npz,
        mode=args.mode,
        num_frames=args.num_frames,
        expr_scale=args.expr_scale,
        head_scale=args.head_scale,
        jaw_scale=args.jaw_scale,
        ref_n=args.ref_n,
        head_axis_order=parse_int_list(args.head_axis_order),
        head_axis_signs=parse_float_list(args.head_axis_signs),
        jaw_axis_order=parse_int_list(args.jaw_axis_order),
        jaw_axis_signs=parse_float_list(args.jaw_axis_signs),
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        for p, payload in zip(frame_npz_paths, frame_payloads):
            save_frame_npz(p, payload, dry_run=False)

    print("Done.")


if __name__ == "__main__":
    main()