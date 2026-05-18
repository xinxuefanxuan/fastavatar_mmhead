#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

try:
    from text_motion.channel_intent import parse_channel_intents
    from text_motion.mmhead_retrieval import load_jsonl, retrieve, write_jsonl
    from text_motion.run_mmhead_to_motion import load_mmhead_npz, smooth_sequence_centered, clamp_head_velocity
except ModuleNotFoundError:
    from channel_intent import parse_channel_intents
    from mmhead_retrieval import load_jsonl, retrieve, write_jsonl
    from run_mmhead_to_motion import load_mmhead_npz, smooth_sequence_centered, clamp_head_velocity


def load_frames(root: Path):
    frames = sorted((root / "flame_param").glob("*.npz"))
    if not frames:
        raise SystemExit("no frame npz")
    payloads = [dict(np.load(p, allow_pickle=True)) for p in frames]
    return frames, payloads


def seq_from_payloads(payloads, key):
    return np.stack([np.asarray(p[key]) for p in payloads], axis=0)


def apply_seq(payloads, key, seq):
    for i,p in enumerate(payloads):
        p[key] = seq[i]




def apply_delta_to_field(payloads, key, delta, scale, num_frames, channel_name):
    base = np.stack([np.asarray(p[key], dtype=np.float32) for p in payloads], axis=0)
    orig_shape = base.shape
    if base.ndim < 2:
        raise ValueError(f"{key} must be at least 2D with time axis, got {orig_shape}")
    T = base.shape[0]
    base2d = base.reshape(T, -1)

    d = np.asarray(delta, dtype=np.float32)
    if d.ndim != 2:
        d = d.reshape(d.shape[0], -1)

    n = min(int(num_frames), T, d.shape[0])
    edit_dim = min(base2d.shape[1], d.shape[1])
    if base2d.shape[1] != d.shape[1]:
        print(
            f"[WARN] {channel_name} dimension mismatch: "
            f"target_dim={base2d.shape[1]}, source_dim={d.shape[1]}. "
            f"Editing first {edit_dim} channels only."
        )

    print(f"[Compose] field={key}, target_shape={orig_shape}, delta_shape={d.shape}, edit_dim={edit_dim}")
    out = base2d.copy()
    out[:n, :edit_dim] = out[:n, :edit_dim] + float(scale) * d[:n, :edit_dim]
    out = out.reshape(orig_shape).astype(np.float32)
    print(f"[Compose] field={key}, final_shape={out.shape}")

    for i, p in enumerate(payloads):
        p[key] = out[i]

def resample_or_repeat(x, target_t):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] == target_t:
        return x
    if x.shape[0] < target_t:
        pad = np.repeat(x[-1:], target_t-x.shape[0], axis=0)
        return np.concatenate([x,pad], axis=0)
    return x[:target_t]


def delta_from_source(src, n, ref_n):
    src = resample_or_repeat(src, n)
    r = min(ref_n, n)
    ref = src[:r].mean(axis=0, keepdims=True)
    return src - ref


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--codebook_jsonl', required=True, type=Path)
    ap.add_argument('--template_motion', required=True, type=Path)
    ap.add_argument('--output_motion', required=True, type=Path)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--head_rank_index', type=int, default=0)
    ap.add_argument('--expr_rank_index', type=int, default=0)
    ap.add_argument('--jaw_rank_index', type=int, default=0)
    ap.add_argument('--num_frames', type=int, default=90)
    ap.add_argument('--expr_scale', type=float, default=0.12)
    ap.add_argument('--head_scale', type=float, default=0.15)
    ap.add_argument('--jaw_scale', type=float, default=0.2)
    ap.add_argument('--ref_n', type=int, default=5)
    ap.add_argument('--expr_smooth_window', type=int, default=1)
    ap.add_argument('--head_smooth_window', type=int, default=1)
    ap.add_argument('--jaw_smooth_window', type=int, default=1)
    ap.add_argument('--head_max_step', type=float, default=0.0)
    ap.add_argument('--head_target_field', choices=['neck_pose','rotation'], default='neck_pose')
    ap.add_argument('--save_metadata_json', type=Path, default=None)
    args=ap.parse_args()

    if args.output_motion.exists(): raise SystemExit(f'output exists: {args.output_motion}')
    shutil.copytree(args.template_motion, args.output_motion)

    intents = parse_channel_intents(args.prompt)
    entries = load_jsonl(args.codebook_jsonl)

    head_rows = retrieve(intents['head_prompt'] or args.prompt, entries, args.top_k, 'head') if intents['need_head'] else []
    expr_rows = retrieve(intents['expr_prompt'] or args.prompt, entries, args.top_k, 'expr') if intents['need_expr'] else []
    jaw_rows = retrieve(intents['jaw_prompt'] or args.prompt, entries, args.top_k, 'jaw') if intents['need_jaw'] else []

    out_prefix = str(args.output_motion)
    if head_rows: write_jsonl(Path(out_prefix + '_head_topk.jsonl'), head_rows)
    if expr_rows: write_jsonl(Path(out_prefix + '_expr_topk.jsonl'), expr_rows)
    if jaw_rows: write_jsonl(Path(out_prefix + '_jaw_topk.jsonl'), jaw_rows)

    frames, payloads = load_frames(args.output_motion)
    n = min(args.num_frames, len(frames))

    if intents['need_expr'] and expr_rows:
        mm_expr, _, _ = load_mmhead_npz(Path(expr_rows[args.expr_rank_index]['motion_path']))
        if args.expr_smooth_window > 1: mm_expr = smooth_sequence_centered(mm_expr, args.expr_smooth_window)
        delta = delta_from_source(mm_expr, n, args.ref_n)
        apply_delta_to_field(payloads, 'expr', delta, args.expr_scale, n, 'expr')

    if intents['need_head'] and head_rows:
        _, mm_head, _ = load_mmhead_npz(Path(head_rows[args.head_rank_index]['motion_path']))
        if args.head_smooth_window > 1: mm_head = smooth_sequence_centered(mm_head, args.head_smooth_window)
        if args.head_max_step > 0: mm_head,_,_ = clamp_head_velocity(mm_head, args.head_max_step)
        delta = delta_from_source(mm_head, n, args.ref_n)
        apply_delta_to_field(payloads, args.head_target_field, delta, args.head_scale, n, 'head')

    if intents['need_jaw'] and jaw_rows:
        _, _, mm_jaw = load_mmhead_npz(Path(jaw_rows[args.jaw_rank_index]['motion_path']))
        if args.jaw_smooth_window > 1: mm_jaw = smooth_sequence_centered(mm_jaw, args.jaw_smooth_window)
        delta = delta_from_source(mm_jaw, n, args.ref_n)
        apply_delta_to_field(payloads, 'jaw_pose', delta, args.jaw_scale, n, 'jaw')

    for p,pl in zip(frames,payloads):
        np.savez(p, **pl)

    if args.save_metadata_json:
        meta = {
            'timestamp': datetime.now(timezone.utc).isoformat(), 'prompt': args.prompt, 'parsed_intents': intents,
            'selected_head_sample': head_rows[args.head_rank_index] if head_rows else None,
            'selected_expr_sample': expr_rows[args.expr_rank_index] if expr_rows else None,
            'selected_jaw_sample': jaw_rows[args.jaw_rank_index] if jaw_rows else None,
            'channel_scales': {'head_scale': args.head_scale, 'expr_scale': args.expr_scale, 'jaw_scale': args.jaw_scale},
            'smoothing_clamp': {'expr_smooth_window': args.expr_smooth_window, 'head_smooth_window': args.head_smooth_window, 'jaw_smooth_window': args.jaw_smooth_window, 'head_max_step': args.head_max_step},
            'target_fields': {'head_target_field': args.head_target_field, 'expr_target_field': 'expr', 'jaw_target_field': 'jaw_pose'},
            'output_motion': str(args.output_motion),
            'recommended_infer_command': f"bash scripts/infer/infer.sh configs/inference/infer.yaml model_zoo/fastavatar/ assets/sample_input/mono_video/nersemble_seq_214.mp4 {str(args.output_motion)}/ 16 16 Monocular false"
        }
        args.save_metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_metadata_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
