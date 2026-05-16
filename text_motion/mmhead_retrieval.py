#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

SYNONYMS = {
    "left": ["left", "turn left", "look left", "head left"],
    "right": ["right", "turn right", "look right", "head right"],
    "up": ["up", "look up", "head up", "raise head"],
    "down": ["down", "look down", "head down", "lower head"],
    "smile": ["smile", "smiling", "happy", "grin"],
    "laugh": ["laugh", "laughing", "big smile"],
    "angry": ["angry", "anger", "mad"],
    "sad": ["sad", "sadness", "unhappy"],
    "surprise": ["surprise", "surprised", "shocked"],
    "mouth": ["mouth", "open mouth", "jaw", "speaking", "talk"],
}

PHRASE_BONUS_TABLE = {
    "turn head left": 2.5,
    "turn head right": 2.5,
    "look left": 2.0,
    "look right": 2.0,
    "look up": 2.0,
    "look down": 2.0,
    "open mouth": 2.0,
    "close eyes": 2.0,
    "smile": 1.2,
    "angry": 1.2,
    "surprised": 1.2,
    "sad": 1.2,
}

HEAD_HINTS = {"head", "left", "right", "up", "down", "look", "turn", "pose"}
EXPR_HINTS = {"smile", "laugh", "angry", "sad", "surprise", "expression", "face", "emotion"}
JAW_HINTS = {"mouth", "jaw", "speaking", "talk", "open", "close"}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    norm = normalize_text(text)
    return norm.split() if norm else []


def expand_prompt(prompt: str) -> Tuple[set[str], set[str], str]:
    norm = normalize_text(prompt)
    tokens = set(tokenize(prompt))
    phrases = {norm}
    for canon, syns in SYNONYMS.items():
        if canon in tokens:
            tokens.update(tokenize(" ".join(syns)))
            phrases.update(normalize_text(s) for s in syns)
            continue
        for s in syns:
            if normalize_text(s) in norm:
                tokens.update(tokenize(" ".join(syns)))
                phrases.update(normalize_text(x) for x in syns)
                break
    return tokens, phrases, norm


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] skip invalid json line {i}: {e}")
    return rows


def score_entry(entry: dict, prompt: str) -> dict:
    expanded_tokens, expanded_phrases, norm_prompt = expand_prompt(prompt)
    searchable = entry.get("searchable_text", "")
    norm_searchable = normalize_text(searchable)
    searchable_tokens = set(tokenize(searchable))

    overlap = expanded_tokens.intersection(searchable_tokens)
    token_overlap_score = float(len(overlap)) / max(len(expanded_tokens), 1) * 10.0

    phrase_bonus = 0.0
    for phrase, bonus in PHRASE_BONUS_TABLE.items():
        if phrase in norm_prompt and phrase in norm_searchable:
            phrase_bonus += bonus

    if norm_prompt in norm_searchable and norm_prompt:
        phrase_bonus += 1.0

    stats = entry.get("motion_stats", {})
    head_max = float(stats.get("head_delta_max_norm", 0.0) or 0.0)
    expr_max = float(stats.get("expr_delta_max_norm", 0.0) or 0.0)
    jaw_max = float(stats.get("jaw_delta_max_norm", 0.0) or 0.0)
    intensity = float(stats.get("motion_intensity_score", 0.0) or 0.0)

    prompt_tokens = set(tokenize(prompt))
    motion_stats_score = 0.05 * intensity
    if prompt_tokens & HEAD_HINTS:
        motion_stats_score += 0.2 * head_max
    if prompt_tokens & EXPR_HINTS:
        motion_stats_score += 0.2 * expr_max
    if prompt_tokens & JAW_HINTS:
        motion_stats_score += 0.2 * jaw_max

    total = token_overlap_score + phrase_bonus + motion_stats_score
    return {
        "total_score": float(total),
        "token_score": float(token_overlap_score),
        "phrase_score": float(phrase_bonus),
        "motion_score": float(motion_stats_score),
        "token_overlap_score": float(token_overlap_score),
        "phrase_bonus": float(phrase_bonus),
        "motion_stats_score": float(motion_stats_score),
        "matched_terms": sorted(overlap),
    }


def retrieve(prompt: str, entries: List[dict], top_k: int) -> List[dict]:
    scored = []
    for e in entries:
        s = score_entry(e, prompt)
        row = {
            "sample_id": e.get("sample_id", ""),
            "motion_path": e.get("motion_path", ""),
            "searchable_text": e.get("searchable_text", ""),
            "annotations": e.get("annotations", {}),
            "motion_stats": e.get("motion_stats", {}),
            "retrieval_scores": {
                "total_score": s["total_score"],
                "token_score": s["token_score"],
                "phrase_score": s["phrase_score"],
                "motion_score": s["motion_score"],
                "token_overlap_score": s["token_overlap_score"],
                "phrase_bonus": s["phrase_bonus"],
                "motion_stats_score": s["motion_stats_score"],
            },
            "matched_terms": s["matched_terms"],
        }
        scored.append(row)

    scored.sort(
        key=lambda x: (
            x["retrieval_scores"]["total_score"],
            x["motion_stats"].get("motion_intensity_score", 0.0),
            x.get("sample_id", ""),
        ),
        reverse=True,
    )

    top = scored[: max(0, top_k)]
    for i, row in enumerate(top, start=1):
        row["rank"] = i
    return top


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_results(rows: List[dict], prompt: str) -> None:
    print(f"[Retrieval] prompt={prompt!r}, top_k={len(rows)}")
    for r in rows:
        score = r["retrieval_scores"]["total_score"]
        sid = r.get("sample_id", "")
        motion = r.get("motion_path", "")
        terms = ",".join(r.get("matched_terms", []))
        print(f"  #{r['rank']:02d} score={score:.4f} sample_id={sid} motion={motion}")
        print(f"      matched_terms=[{terms}]")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Keyword retrieval over MMHead codebook JSONL.")
    ap.add_argument("--prompt", required=True, type=str)
    ap.add_argument("--codebook_jsonl", required=True, type=Path)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--output_jsonl", type=Path, default=None)
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    if not args.codebook_jsonl.exists():
        raise SystemExit(f"codebook_jsonl not found: {args.codebook_jsonl}")

    entries = load_jsonl(args.codebook_jsonl)
    if not entries:
        raise SystemExit("Empty codebook JSONL.")

    top_rows = retrieve(prompt=args.prompt, entries=entries, top_k=args.top_k)
    print_results(top_rows, args.prompt)

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, top_rows)
        print(f"[Saved] {args.output_jsonl}")


if __name__ == "__main__":
    main()
