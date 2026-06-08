#!/usr/bin/env python3
"""Inspect whether a P9.2 checkpoint only changed MotionTokenAdapter weights.

Compares a pretrained FastAvatar checkpoint against a trained P9.2 checkpoint and
fails by default if any changed/added/removed tensor outside motion_token_adapter
is detected.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch


ADAPTER_SUBSTRING = "motion_token_adapter"


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload type at {path}: {type(state).__name__}")
    tensors = {str(k): v.detach().cpu() for k, v in state.items() if torch.is_tensor(v)}
    if not tensors:
        raise RuntimeError(f"No tensor entries found in checkpoint: {path}")
    return tensors


def is_adapter_param(name: str, adapter_substring: str = ADAPTER_SUBSTRING) -> bool:
    return adapter_substring in name


def tensor_norms(tensor: torch.Tensor) -> tuple[float, float]:
    value = tensor.detach().cpu()
    if not torch.is_floating_point(value) and not torch.is_complex(value):
        value = value.to(torch.float32)
    else:
        value = value.to(torch.float32)
    l2 = float(torch.linalg.vector_norm(value.reshape(-1), ord=2).item()) if value.numel() else 0.0
    max_abs = float(value.abs().max().item()) if value.numel() else 0.0
    return l2, max_abs


def tensor_diff(pretrained: torch.Tensor, trained: torch.Tensor) -> tuple[float, float, bool]:
    if pretrained.shape != trained.shape:
        return float("nan"), float("nan"), True
    left = pretrained.detach().cpu()
    right = trained.detach().cpu()
    if not torch.is_floating_point(left) and not torch.is_complex(left):
        left = left.to(torch.float32)
    else:
        left = left.to(torch.float32)
    if not torch.is_floating_point(right) and not torch.is_complex(right):
        right = right.to(torch.float32)
    else:
        right = right.to(torch.float32)
    diff = right - left
    l2 = float(torch.linalg.vector_norm(diff.reshape(-1), ord=2).item()) if diff.numel() else 0.0
    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    return l2, max_abs, False


def compare_checkpoints(
    pretrained: dict[str, torch.Tensor],
    trained: dict[str, torch.Tensor],
    atol: float,
    adapter_substring: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = sorted(set(pretrained) | set(trained))
    rows: list[dict[str, Any]] = []
    summary = {
        "pretrained_num_tensors": len(pretrained),
        "trained_num_tensors": len(trained),
        "changed_total": 0,
        "changed_motion_token_adapter": 0,
        "changed_non_motion_token_adapter": 0,
        "added_total": 0,
        "removed_total": 0,
        "shape_mismatch_total": 0,
        "unchanged_total": 0,
    }

    for name in names:
        in_pretrained = name in pretrained
        in_trained = name in trained
        adapter_param = is_adapter_param(name, adapter_substring)
        status = "unchanged"
        l2_norm = 0.0
        max_abs_diff = 0.0
        shape_mismatch = False
        pretrained_shape = list(pretrained[name].shape) if in_pretrained else None
        trained_shape = list(trained[name].shape) if in_trained else None

        if not in_pretrained:
            status = "added"
            l2_norm, max_abs_diff = tensor_norms(trained[name])
            summary["added_total"] += 1
        elif not in_trained:
            status = "removed"
            l2_norm, max_abs_diff = tensor_norms(pretrained[name])
            summary["removed_total"] += 1
        else:
            l2_norm, max_abs_diff, shape_mismatch = tensor_diff(pretrained[name], trained[name])
            if shape_mismatch:
                status = "shape_mismatch"
                summary["shape_mismatch_total"] += 1
            elif max_abs_diff > atol:
                status = "changed"
            else:
                summary["unchanged_total"] += 1

        if status != "unchanged":
            row = {
                "name": name,
                "status": status,
                "is_motion_token_adapter": adapter_param,
                "pretrained_shape": pretrained_shape,
                "trained_shape": trained_shape,
                "l2_norm": l2_norm,
                "max_abs_diff": max_abs_diff,
            }
            rows.append(row)
            summary["changed_total"] += 1
            if adapter_param:
                summary["changed_motion_token_adapter"] += 1
            else:
                summary["changed_non_motion_token_adapter"] += 1

    return rows, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint_diff_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "checkpoint_diff_changed_params.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = output_dir / "checkpoint_diff_changed_params.csv"
    fieldnames = [
        "name",
        "status",
        "is_motion_token_adapter",
        "pretrained_shape",
        "trained_shape",
        "l2_norm",
        "max_abs_diff",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) for key in fieldnames})

    lines = [
        "# P9.2 Checkpoint Diff",
        "",
        "## Summary",
        f"* pretrained checkpoint: `{summary['pretrained_checkpoint']}`",
        f"* trained checkpoint: `{summary['trained_checkpoint']}`",
        f"* changed parameters: `{summary['changed_total']}`",
        f"* changed motion_token_adapter parameters: `{summary['changed_motion_token_adapter']}`",
        f"* changed non-motion_token_adapter parameters: `{summary['changed_non_motion_token_adapter']}`",
        f"* unexpected non-adapter changes: `{summary['unexpected_non_adapter_change_count']}`",
        f"* pass: `{summary['pass']}`",
        "",
        "## Changed Parameters",
        "| status | adapter? | max_abs_diff | l2_norm | name |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['status']} | {row['is_motion_token_adapter']} | "
            f"{row['max_abs_diff']:.8g} | {row['l2_norm']:.8g} | `{row['name']}` |"
        )
    (output_dir / "checkpoint_diff_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pretrained and P9.2 checkpoints and fail on unexpected non-adapter changes."
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=Path,
        default=Path("model_zoo/fastavatar/model.safetensors"),
        help="Baseline pretrained FastAvatar checkpoint.",
    )
    parser.add_argument(
        "--trained_checkpoint",
        type=Path,
        default=Path("exps/checkpoints/fastavatar/fastavatar_motion_zero_token_overfit_micro_render_500step/000500/model.safetensors"),
        help="P9.2 trained checkpoint to inspect.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/mmhead_debug/p9_2_checkpoint_diff"),
        help="Directory for JSON/CSV/Markdown reports.",
    )
    parser.add_argument("--adapter_substring", default=ADAPTER_SUBSTRING)
    parser.add_argument("--atol", type=float, default=0.0, help="Max-abs tolerance for treating common tensors as changed.")
    parser.add_argument(
        "--allow_non_adapter_changes",
        action="store_true",
        help="Do not exit non-zero when non-motion_token_adapter tensors changed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pretrained_path = resolve_path(args.pretrained_checkpoint)
    trained_path = resolve_path(args.trained_checkpoint)
    output_dir = resolve_path(args.output_dir)

    pretrained = load_checkpoint(pretrained_path)
    trained = load_checkpoint(trained_path)
    rows, summary = compare_checkpoints(pretrained, trained, atol=float(args.atol), adapter_substring=args.adapter_substring)

    unexpected = [row for row in rows if not row["is_motion_token_adapter"]]
    summary.update({
        "pretrained_checkpoint": str(pretrained_path),
        "trained_checkpoint": str(trained_path),
        "output_dir": str(output_dir),
        "adapter_substring": args.adapter_substring,
        "atol": float(args.atol),
        "unexpected_non_adapter_change_count": len(unexpected),
        "unexpected_non_adapter_parameter_names": [row["name"] for row in unexpected],
        "allow_non_adapter_changes": bool(args.allow_non_adapter_changes),
        "pass": len(unexpected) == 0 or bool(args.allow_non_adapter_changes),
    })
    write_outputs(rows, summary, output_dir)

    print("[P9.2CKPT-DIFF] Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[P9.2CKPT-DIFF] Wrote report: {output_dir / 'checkpoint_diff_report.md'}")
    if unexpected and not args.allow_non_adapter_changes:
        print("[P9.2CKPT-DIFF][FAIL] Non-motion_token_adapter checkpoint changes detected:", file=sys.stderr)
        for row in unexpected:
            print(
                f"  {row['status']} {row['name']} max_abs_diff={row['max_abs_diff']:.8g} l2={row['l2_norm']:.8g}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
