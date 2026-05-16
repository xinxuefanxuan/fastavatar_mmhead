#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from text_motion.mmhead_retrieval import load_jsonl, retrieve, write_jsonl, print_results
except ModuleNotFoundError:
    from mmhead_retrieval import load_jsonl, retrieve, write_jsonl, print_results


def main() -> None:
    ap = argparse.ArgumentParser(description="Text prompt -> MMHead retrieval -> FastAvatar motion retarget")
    ap.add_argument("--prompt", required=True, type=str)
    ap.add_argument("--codebook_jsonl", required=True, type=Path)
    ap.add_argument("--template_motion", required=True, type=Path)
    ap.add_argument("--output_motion", required=True, type=Path)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--rank_index", type=int, default=0)
    ap.add_argument("--save_retrieval_jsonl", type=Path, default=None)
    ap.add_argument("--save_metadata_json", type=Path, default=None)

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

    if not args.codebook_jsonl.exists():
        raise SystemExit(f"codebook_jsonl not found: {args.codebook_jsonl}")

    entries = load_jsonl(args.codebook_jsonl)
    if not entries:
        raise SystemExit("Empty codebook JSONL.")

    top_rows = retrieve(args.prompt, entries, args.top_k)
    if not top_rows:
        raise SystemExit("No retrieval results.")

    print_results(top_rows, args.prompt)

    if args.rank_index < 0 or args.rank_index >= len(top_rows):
        raise SystemExit(f"rank_index out of range: {args.rank_index}, top_k={len(top_rows)}")

    selected = top_rows[args.rank_index]
    mmhead_path = Path(selected["motion_path"])
    if not mmhead_path.exists() and not args.dry_run:
        raise SystemExit(f"Selected motion_path not found: {mmhead_path}")

    if args.save_retrieval_jsonl:
        write_jsonl(args.save_retrieval_jsonl, top_rows)
        print(f"[Saved] retrieval top-k -> {args.save_retrieval_jsonl}")

    cmd = [
        "python", "text_motion/run_mmhead_to_motion.py",
        "--template_motion", str(args.template_motion),
        "--output_motion", str(args.output_motion),
        "--mmhead_npz", str(mmhead_path),
        "--mode", args.mode,
        "--num_frames", str(args.num_frames),
        "--expr_scale", str(args.expr_scale),
        "--head_scale", str(args.head_scale),
        "--jaw_scale", str(args.jaw_scale),
        "--ref_n", str(args.ref_n),
        "--head_axis_order", args.head_axis_order,
        "--head_axis_signs", args.head_axis_signs,
        "--jaw_axis_order", args.jaw_axis_order,
        "--jaw_axis_signs", args.jaw_axis_signs,
        "--primitive_config", str(args.primitive_config),
    ]
    if args.dry_run:
        cmd.append("--dry_run")

    print("[Run retarget]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if args.save_metadata_json:
        args.save_metadata_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": args.prompt,
            "selected": {
                "sample_id": selected.get("sample_id", ""),
                "motion_path": selected.get("motion_path", ""),
                "searchable_text": selected.get("searchable_text", ""),
                "retrieval_score": selected.get("retrieval_scores", {}).get("total_score", 0.0),
            },
            "top_k_result_path": str(args.save_retrieval_jsonl) if args.save_retrieval_jsonl else "",
            "output_motion_path": str(args.output_motion),
            "command_used": " ".join(cmd),
        }
        args.save_metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Saved] metadata -> {args.save_metadata_json}")


if __name__ == "__main__":
    main()
