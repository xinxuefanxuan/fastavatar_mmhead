#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_PROMPTS = {
    "turn_left": "turn left",
    "turn_right": "turn right",
    "smile": "smile",
    "open_mouth": "open mouth",
    "turn_left_and_smile": "turn left and smile",
    "turn_right_and_smile": "turn right and smile",
    "turn_left_and_open_mouth": "turn left and open mouth",
    "turn_right_and_open_mouth": "turn right and open mouth",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cosine_sim(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    q = query / max(np.linalg.norm(query), 1e-8)
    b = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-8)
    return b @ q


def motion_stats(motion: np.ndarray) -> dict:
    expr = motion[:, :50]
    head = motion[:, 50:53]
    jaw = motion[:, 53:56]
    expr_n = np.linalg.norm(expr, axis=1)
    head_n = np.linalg.norm(head, axis=1)
    jaw_n = np.linalg.norm(jaw, axis=1)
    yaw = head[:, 1]
    return {
        "expr_norm_mean": float(expr_n.mean()),
        "expr_norm_max": float(expr_n.max()),
        "head_norm_mean": float(head_n.mean()),
        "head_norm_max": float(head_n.max()),
        "jaw_norm_mean": float(jaw_n.mean()),
        "jaw_norm_max": float(jaw_n.max()),
        "yaw_min": float(yaw.min()),
        "yaw_max": float(yaw.max()),
    }


def load_render_compatible(npz_path: Path, motion_key: str, copy_motion_key: str) -> tuple[dict, dict]:
    arr = np.load(npz_path, allow_pickle=False)
    if motion_key not in arr and "motion" not in arr and "motion_raw" not in arr:
        raise ValueError(f"motion field missing in {npz_path}")

    if motion_key in arr:
        motion = np.asarray(arr[motion_key], dtype=np.float32)
    elif "motion" in arr:
        motion = np.asarray(arr["motion"], dtype=np.float32)
    else:
        motion = np.asarray(arr["motion_raw"], dtype=np.float32)

    if motion.ndim != 2 or motion.shape[1] < 56:
        raise ValueError(f"invalid motion shape {motion.shape} in {npz_path}")
    motion = motion[:, :56]

    out = {
        "motion": motion.astype(np.float32),
        "motion_raw": np.asarray(arr[copy_motion_key], dtype=np.float32)[:, :56] if copy_motion_key in arr else motion.astype(np.float32),
        "expr_delta": (np.asarray(arr["expr_delta"], dtype=np.float32) if "expr_delta" in arr else motion[:, :50]).astype(np.float32),
        "head_delta": (np.asarray(arr["head_delta"], dtype=np.float32) if "head_delta" in arr else motion[:, 50:53]).astype(np.float32),
        "jaw_delta": (np.asarray(arr["jaw_delta"], dtype=np.float32) if "jaw_delta" in arr else motion[:, 53:56]).astype(np.float32),
    }
    if "motion_norm" in arr:
        out["motion_norm"] = np.asarray(arr["motion_norm"], dtype=np.float32)[:, :56]
    stats = motion_stats(out["motion"])
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", type=Path, default=Path("outputs/mmhead_debug/motion_dataset_v1_ae_debug/train.jsonl"))
    ap.add_argument("--text_embeddings_dir", type=Path, default=Path("outputs/mmhead_debug/text_embeddings_v1/train"))
    ap.add_argument("--encoder_name", type=str, default="/home/yuanyuhao/models/all-MiniLM-L6-v2")
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/mmhead_debug/baselines/nn_retrieval_v1"))
    ap.add_argument("--prompts_json", type=Path, default=None)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--motion_key", type=str, default="motion")
    ap.add_argument("--copy_motion_key", type=str, default="motion")
    ap.add_argument("--save_topk_report", action="store_true")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS if args.prompts_json is None else json.loads(args.prompts_json.read_text(encoding="utf-8"))

    emb_dir = args.text_embeddings_dir
    emb = np.load(emb_dir / "embeddings.npy").astype(np.float32)
    sample_ids = json.loads((emb_dir / "sample_ids.json").read_text(encoding="utf-8"))
    texts = json.loads((emb_dir / "texts.json").read_text(encoding="utf-8"))

    rows = read_jsonl(args.train_manifest)
    sid_to_path = {}
    for r in rows:
        sid = str(r.get("sample_id", ""))
        p = r.get("npz_path") or r.get("motion_path")
        if sid and p:
            sid_to_path[sid] = str(p)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.encoder_name, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = []
    summary_rows = []

    for key, prompt in prompts.items():
        q = np.asarray(model.encode([prompt], convert_to_numpy=True, normalize_embeddings=False), dtype=np.float32)[0]
        sims = cosine_sim(q, emb)
        top_idx = np.argsort(-sims)[: max(1, args.top_k)]

        candidates = []
        selected = None
        output_npz = args.output_dir / f"{key}.npz"

        for rank, idx in enumerate(top_idx, start=1):
            sid = str(sample_ids[idx])
            txt = str(texts[idx]) if idx < len(texts) else ""
            npz_path = sid_to_path.get(sid)
            cand = {
                "rank": rank,
                "sample_id": sid,
                "similarity": float(sims[idx]),
                "text": txt,
                "npz_path": npz_path,
            }
            if npz_path is None:
                print(f"[WARN] sample_id from embeddings missing in manifest: {sid}")
                cand["error"] = "missing_in_manifest"
            elif not Path(npz_path).exists():
                print(f"[WARN] motion npz does not exist: {npz_path}")
                cand["error"] = "npz_missing"
            else:
                try:
                    packed, st = load_render_compatible(Path(npz_path), args.motion_key, args.copy_motion_key)
                    cand["stats"] = st
                    if selected is None:
                        selected = {"sample_id": sid, "similarity": float(sims[idx]), "npz_path": npz_path, "stats": st}
                        np.savez(output_npz, **packed)
                        print(f"[{key}] selected={sid} sim={sims[idx]:.4f}")
                        print(f"[{key}] stats expr={st['expr_norm_mean']:.4f}/{st['expr_norm_max']:.4f} head={st['head_norm_mean']:.4f}/{st['head_norm_max']:.4f} jaw={st['jaw_norm_mean']:.4f}/{st['jaw_norm_max']:.4f} yaw={st['yaw_min']:.4f}/{st['yaw_max']:.4f}")
                except Exception as e:
                    print(f"[WARN] failed to load candidate npz {npz_path}: {e}")
                    cand["error"] = str(e)
            candidates.append(cand)

        rep_item = {
            "prompt_key": key,
            "prompt": prompt,
            "top_k": candidates,
            "selected_top1": selected,
            "output_npz": str(output_npz),
        }
        report.append(rep_item)

        if selected is not None:
            st = selected["stats"]
            summary_rows.append({
                "prompt_key": key,
                "prompt": prompt,
                "sample_id": selected["sample_id"],
                "similarity": selected["similarity"],
                "expr_max": st["expr_norm_max"],
                "head_max": st["head_norm_max"],
                "jaw_max": st["jaw_norm_max"],
                "yaw_min": st["yaw_min"],
                "yaw_max": st["yaw_max"],
                "output_npz": str(output_npz),
            })

    (args.output_dir / "retrieval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = ["# NN Retrieval Baseline Report", ""]
    for item in report:
        md_lines.append(f"## {item['prompt_key']}")
        md_lines.append(f"- prompt: `{item['prompt']}`")
        sel = item.get("selected_top1")
        if sel is None:
            md_lines.append("- selected top1: N/A")
            md_lines.append("")
            continue
        md_lines.append(f"- top1 sample_id: `{sel['sample_id']}`")
        md_lines.append(f"- similarity: `{sel['similarity']:.6f}`")
        md_lines.append(f"- output: `{item['output_npz']}`")
        st = sel["stats"]
        md_lines.append(f"- stats: expr_max={st['expr_norm_max']:.4f}, head_max={st['head_norm_max']:.4f}, jaw_max={st['jaw_norm_max']:.4f}, yaw=[{st['yaw_min']:.4f},{st['yaw_max']:.4f}]")
        if args.save_topk_report:
            md_lines.append("")
            md_lines.append("| rank | sample_id | similarity | npz_path |")
            md_lines.append("|---:|---|---:|---|")
            for c in item["top_k"]:
                md_lines.append(f"| {c['rank']} | {c['sample_id']} | {c['similarity']:.6f} | {c.get('npz_path','')} |")
        md_lines.append("")
    (args.output_dir / "retrieval_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    with (args.output_dir / "output_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["prompt_key", "prompt", "sample_id", "similarity", "expr_max", "head_max", "jaw_max", "yaw_min", "yaw_max", "output_npz"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"[NN] saved {(args.output_dir / 'retrieval_report.json')}")
    print(f"[NN] saved {(args.output_dir / 'retrieval_report.md')}")
    print(f"[NN] saved {(args.output_dir / 'output_summary.csv')}")


if __name__ == "__main__":
    main()
