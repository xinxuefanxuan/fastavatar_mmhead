#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmhead_debug/fastavatar_smile_multidim_search}"
NEUTRAL_TEMPLATE="${NEUTRAL_TEMPLATE:-assets/sample_motion/nersemble_seq_214_neutral}"
SEQUENCE_NAME="${SEQUENCE_NAME:-nersemble_seq_214}"
NUM_CANDIDATES="${NUM_CANDIDATES:-20}"
SEED="${SEED:-214}"
RENDER="${RENDER:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESUME="${RESUME:-1}"
CONTINUE_ON_RENDER_FAIL="${CONTINUE_ON_RENDER_FAIL:-1}"
INFERENCE_N_FRAMES="${INFERENCE_N_FRAMES:-16}"
MAX_SINGLE_FRAME_RENDER="${MAX_SINGLE_FRAME_RENDER:-1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-7}"
INFER_CONFIG="${INFER_CONFIG:-configs/inference/infer.yaml}"
MODEL_ROOT="${MODEL_ROOT:-model_zoo/fastavatar/}"
IMAGE_INPUT="${IMAGE_INPUT:-assets/sample_input/mono_video/nersemble_seq_214.mp4}"
MODE="${MODE:-Monocular}"
ENABLE_CAMERA_ROTATION="${ENABLE_CAMERA_ROTATION:-false}"

VIDEO_DIR="${OUTPUT_DIR}/fastavatar_video"
FRAME_DIR="${OUTPUT_DIR}/frames"
MOUTH_FRAME_DIR="${OUTPUT_DIR}/mouth_frames"
MANIFEST="${OUTPUT_DIR}/manifest.json"
RENDER_STATUS="${OUTPUT_DIR}/render_status.tsv"
GRID="${OUTPUT_DIR}/smile_multidim_grid.png"
MOUTH_GRID="${OUTPUT_DIR}/smile_multidim_mouth_crop_grid.png"
REPORT="${OUTPUT_DIR}/smile_multidim_report.md"

mkdir -p "${OUTPUT_DIR}" "${VIDEO_DIR}" "${FRAME_DIR}" "${MOUTH_FRAME_DIR}"

printf '[SmileMultiDim] cwd=%s\n' "$(pwd)"
printf '[SmileMultiDim] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[SmileMultiDim] neutral_template=%s sequence_name=%s\n' "${NEUTRAL_TEMPLATE}" "${SEQUENCE_NAME}"
printf '[SmileMultiDim] num_candidates=%s seed=%s render=%s resume=%s skip_existing=%s\n' \
  "${NUM_CANDIDATES}" "${SEED}" "${RENDER}" "${RESUME}" "${SKIP_EXISTING}"
printf '[SmileMultiDim] inference_n_frames=%s max_single_frame_render=%s cuda=%s\n' \
  "${INFERENCE_N_FRAMES}" "${MAX_SINGLE_FRAME_RENDER}" "${CUDA_VISIBLE_DEVICES_VALUE}"

if [[ ! -d "${NEUTRAL_TEMPLATE}" ]]; then
  printf '[ERROR] neutral template not found: %s\n' "${NEUTRAL_TEMPLATE}" >&2
  exit 1
fi

export OUTPUT_DIR NEUTRAL_TEMPLATE SEQUENCE_NAME NUM_CANDIDATES SEED MANIFEST

python - <<'PY'
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np

ROOT_META_FILES = [
    "canonical_flame_param.npz",
    "transforms.json",
    "transforms_train.json",
    "transforms_val.json",
    "transforms_test.json",
    "transforms_backup.json",
    "transforms_backup_flame.json",
]


def ramp_hold_release(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    ramp = max(2, n // 4)
    hold = max(1, n - 2 * ramp)
    up = np.linspace(0.0, 1.0, ramp, endpoint=False, dtype=np.float32)
    mid = np.ones(hold, dtype=np.float32)
    down = np.linspace(1.0, 0.0, n - ramp - hold, endpoint=True, dtype=np.float32)
    return np.concatenate([up, mid, down])[:n]


def prepare_pack(neutral: Path, pack_root: Path, sequence_name: str) -> Path:
    if pack_root.exists():
        shutil.rmtree(pack_root)
    seq_dir = pack_root / sequence_name
    pack_root.mkdir(parents=True, exist_ok=True)
    for name in ROOT_META_FILES:
        src = neutral / name
        if src.exists():
            shutil.copy2(src, pack_root / name)
    seq_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(neutral / "flame_param", seq_dir / "flame_param")
    processed_src = neutral / "processed_data"
    if processed_src.exists():
        (seq_dir / "processed_data").symlink_to(processed_src.resolve(), target_is_directory=True)
    for link_name, rel_target in [
        ("flame_param", f"{sequence_name}/flame_param"),
        ("processed_data", f"{sequence_name}/processed_data"),
    ]:
        link = pack_root / link_name
        if link.exists() or link.is_symlink():
            link.unlink()
        target = pack_root / rel_target
        if target.exists() or target.is_symlink():
            link.symlink_to(rel_target, target_is_directory=True)
    return seq_dir


def write_pack(neutral: Path, output_dir: Path, sequence_name: str, item: dict, base_exprs: np.ndarray, envelope: np.ndarray) -> None:
    pack_root = output_dir / item["case_name"] / "fastavatar_pack"
    seq_dir = prepare_pack(neutral, pack_root, sequence_name)
    frame_paths = sorted((seq_dir / "flame_param").glob("*.npz"))
    expr_vec = np.zeros(100, dtype=np.float32)
    for term in item["terms"]:
        expr_vec[int(term["dim"])] += float(term["weight"])
    jaw_x = float(item["jaw_x"])
    for idx, frame_path in enumerate(frame_paths):
        data = dict(np.load(frame_path, allow_pickle=True))
        data["expr"] = (base_exprs[idx] + expr_vec * envelope[idx]).reshape(1, 100).astype(np.float32)
        for key in ("rotation", "neck_pose"):
            if key in data:
                data[key] = np.zeros_like(np.asarray(data[key], dtype=np.float32))
        if "jaw_pose" in data:
            jaw = np.zeros_like(np.asarray(data["jaw_pose"], dtype=np.float32))
            jaw.reshape(-1)[0] = jaw_x * float(envelope[idx])
            data["jaw_pose"] = jaw
        np.savez(frame_path, **data)
    item["expr_vector_nonzero"] = {str(i): float(v) for i, v in enumerate(expr_vec) if abs(float(v)) > 1e-8}
    item["expr_l1"] = float(np.abs(expr_vec).sum())
    item["expr_l2"] = float(np.linalg.norm(expr_vec))
    item["pack_root"] = str(pack_root)
    item["motion_seqs_dir"] = str(pack_root / sequence_name)
    item["video"] = f"fastavatar_video/{item['case_name']}.mp4"
    item["infer_log"] = f"{item['case_name']}/fastavatar_render/infer.log"


output_dir = Path(os.environ["OUTPUT_DIR"])
neutral = Path(os.environ["NEUTRAL_TEMPLATE"])
sequence_name = os.environ["SEQUENCE_NAME"]
num_candidates = int(os.environ["NUM_CANDIDATES"])
seed = int(os.environ["SEED"])
manifest_path = Path(os.environ["MANIFEST"])
random.seed(seed)
np.random.seed(seed)

frame_paths = sorted((neutral / "flame_param").glob("*.npz"))
if not frame_paths:
    raise SystemExit(f"no neutral flame_param frames under {neutral}")
base_exprs = []
for frame_path in frame_paths:
    data = np.load(frame_path, allow_pickle=True)
    expr = np.asarray(data["expr"], dtype=np.float32).reshape(-1)
    if expr.shape[0] != 100:
        raise SystemExit(f"expected 100D expr, got {expr.shape} at {frame_path}")
    base_exprs.append(expr)
base_exprs = np.stack(base_exprs, axis=0)
envelope = ramp_hold_release(base_exprs.shape[0])

# Weakly visible ingredients from the one-hot probe. Avoid dim17/dim35 as standalone
# leaders because they did not show useful smile response despite high proxy score.
primary_pool = [0, 48, 57, 60, 65, 81, 95]
secondary_pool = [50, 56, 59, 72, 80, 85, 86, 87, 93]
all_pool = primary_pool + secondary_pool
positive_pref = {57, 60, 65, 80, 81, 85, 86, 87, 93, 95, 48}
negative_ok = {0, 48, 50, 59, 72, 81, 86, 95}

hand_specs = [
    {"dims": [48, 57, 60, 65], "jaw": 0.018, "scale": 0.60, "note": "lower-face soft smile blend"},
    {"dims": [0, 48, 57, 81], "jaw": 0.020, "scale": 0.55, "note": "0:50 plus 50:100 corner/lip blend"},
    {"dims": [57, 60, 65, 81, 95], "jaw": 0.024, "scale": 0.50, "note": "50:100 relaxed mouth blend"},
    {"dims": [48, 60, 65, 85, 95], "jaw": 0.026, "scale": 0.48, "note": "gentle teeth candidate"},
    {"dims": [0, 48, 57, 60, 65, 81], "jaw": 0.030, "scale": 0.42, "note": "wider low-gain blend with small jaw"},
]

items = []
for idx in range(num_candidates):
    if idx < len(hand_specs):
        spec = hand_specs[idx]
        dims = spec["dims"]
        jaw_x = spec["jaw"]
        scale = spec["scale"]
        note = spec["note"]
    else:
        k = random.randint(3, 8)
        dims = random.sample(all_pool, k)
        jaw_x = random.choice([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035])
        scale = random.uniform(0.35, 0.70)
        note = "seeded random blend from weak-response dims"

    terms = []
    for dim in dims:
        sign = 1.0
        if dim in negative_ok and random.random() < 0.25:
            sign = -1.0
        if dim not in positive_pref and dim not in negative_ok:
            sign = 1.0
        weight = sign * scale * random.uniform(0.45, 1.0)
        terms.append({"dim": int(dim), "weight": float(round(weight, 4))})

    # Keep total expression energy bounded; one-hot amp=1 was safe but weak.
    vec = np.zeros(100, dtype=np.float32)
    for term in terms:
        vec[term["dim"]] += term["weight"]
    l2 = float(np.linalg.norm(vec))
    if l2 > 1.6:
        shrink = 1.6 / l2
        for term in terms:
            term["weight"] = float(round(term["weight"] * shrink, 4))
        vec *= shrink

    case_name = f"smile_md_{idx + 1:03d}"
    item = {
        "case_name": case_name,
        "terms": terms,
        "jaw_x": float(jaw_x),
        "note": note,
        "manual_label": "TODO inspect grid/video",
        "metric": {},
    }
    write_pack(neutral, output_dir, sequence_name, item, base_exprs, envelope)
    items.append(item)

manifest = {
    "output_dir": str(output_dir),
    "neutral_template": str(neutral),
    "sequence_name": sequence_name,
    "num_candidates": num_candidates,
    "seed": seed,
    "source_probe_dims": {
        "primary_pool": primary_pool,
        "secondary_pool": secondary_pool,
        "excluded_bad_leaders": [17, 35],
    },
    "items": items,
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"[SmileMultiDim] manifest={manifest_path}")
print(f"[SmileMultiDim] candidates={len(items)}")
PY

if [[ "${RENDER}" == "1" ]]; then
  if [[ "${RESUME}" == "1" && -f "${RENDER_STATUS}" ]]; then
    printf '[SmileMultiDim] RESUME=1, appending to existing render status: %s\n' "${RENDER_STATUS}"
  else
    printf 'case\tstatus\tvideo\tinfer_log\tpack_root\tmotion_seqs_dir\n' > "${RENDER_STATUS}"
  fi
  mapfile -t CASES < <(python - "${MANIFEST}" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["items"]:
    print(f"{item['case_name']}|{item['pack_root']}|{item['motion_seqs_dir']}")
PY
)

  for row in "${CASES[@]}"; do
    IFS='|' read -r case_name pack_root motion_seqs_dir <<< "${row}"
    case_dir="${OUTPUT_DIR}/${case_name}/fastavatar_render"
    infer_video_dump="${case_dir}/infer_videos"
    infer_image_dump="${case_dir}/infer_images"
    infer_tmp_dump="${case_dir}/infer_tmp"
    infer_log="${case_dir}/infer.log"
    video_out="${VIDEO_DIR}/${case_name}.mp4"
    mkdir -p "${case_dir}" "${infer_video_dump}" "${infer_image_dump}" "${infer_tmp_dump}" "${VIDEO_DIR}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${video_out}" ]]; then
      printf '[SmileMultiDim] skipping existing video %s\n' "${case_name}"
      printf '%s\tskipped_existing\t%s\t%s\t%s\t%s\n' \
        "${case_name}" "${video_out}" "${infer_log}" "${pack_root}" "${motion_seqs_dir}" >> "${RENDER_STATUS}"
      continue
    fi
    printf '[SmileMultiDim] rendering %s\n' "${case_name}"
    : > "${infer_log}"
    if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
      VIDEO_DUMP="${infer_video_dump}" \
      IMAGE_DUMP="${infer_image_dump}" \
      SAVE_TMP_DUMP="${infer_tmp_dump}" \
      bash scripts/infer/infer.sh \
        "${INFER_CONFIG}" \
        "${MODEL_ROOT}" \
        "${IMAGE_INPUT}" \
        "${motion_seqs_dir}" \
        "${INFERENCE_N_FRAMES}" \
        "${MAX_SINGLE_FRAME_RENDER}" \
        "${MODE}" \
        "${ENABLE_CAMERA_ROTATION}" >> "${infer_log}" 2>&1; then
      latest_mp4="$(find "${infer_video_dump}" -type f -name "*.mp4" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
      if [[ -z "${latest_mp4}" || ! -f "${latest_mp4}" ]]; then
        printf '[WARN] render finished but no mp4 found for %s\n' "${case_name}" >&2
        printf '%s\tfailed_no_mp4\t%s\t%s\t%s\t%s\n' \
          "${case_name}" "${video_out}" "${infer_log}" "${pack_root}" "${motion_seqs_dir}" >> "${RENDER_STATUS}"
        continue
      fi
      cp -f "${latest_mp4}" "${video_out}"
      printf '%s\tok\t%s\t%s\t%s\t%s\n' \
        "${case_name}" "${video_out}" "${infer_log}" "${pack_root}" "${motion_seqs_dir}" >> "${RENDER_STATUS}"
    else
      printf '[WARN] FastAvatar render failed for %s; last 120 log lines:\n' "${case_name}" >&2
      tail -n 120 "${infer_log}" >&2 || true
      printf '%s\tfailed\t%s\t%s\t%s\t%s\n' \
        "${case_name}" "${video_out}" "${infer_log}" "${pack_root}" "${motion_seqs_dir}" >> "${RENDER_STATUS}"
      if [[ "${CONTINUE_ON_RENDER_FAIL}" != "1" ]]; then
        exit 1
      fi
    fi
  done
else
  if [[ "${RESUME}" != "1" || ! -f "${RENDER_STATUS}" ]]; then
    printf 'case\tstatus\tvideo\tinfer_log\tpack_root\tmotion_seqs_dir\n' > "${RENDER_STATUS}"
  fi
fi

export VIDEO_DIR FRAME_DIR MOUTH_FRAME_DIR GRID MOUTH_GRID MANIFEST RENDER_STATUS REPORT
python - <<'PY'
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

video_dir = Path(os.environ["VIDEO_DIR"])
frame_dir = Path(os.environ["FRAME_DIR"])
mouth_dir = Path(os.environ["MOUTH_FRAME_DIR"])
grid_path = Path(os.environ["GRID"])
mouth_grid_path = Path(os.environ["MOUTH_GRID"])
manifest_path = Path(os.environ["MANIFEST"])
render_status_path = Path(os.environ["RENDER_STATUS"])
report_path = Path(os.environ["REPORT"])
manifest = json.load(open(manifest_path, encoding="utf-8"))
items = manifest["items"]
frame_dir.mkdir(parents=True, exist_ok=True)
mouth_dir.mkdir(parents=True, exist_ok=True)

status = {}
if render_status_path.exists():
    for line in render_status_path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 6:
            status[parts[0]] = {
                "status": parts[1],
                "video": parts[2],
                "infer_log": parts[3],
                "pack_root": parts[4],
                "motion_seqs_dir": parts[5],
            }


def frame_count(path: Path) -> int:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1",
            str(path),
        ], text=True).strip()
        return int(out) if out and out != "N/A" else 1
    except Exception:
        return 1


def crop_mouth(im: Image.Image) -> Image.Image:
    w, h = im.size
    return im.crop((int(w * 0.30), int(h * 0.42), int(w * 0.70), int(h * 0.72)))


def metrics_for(case_name: str) -> dict:
    video = video_dir / f"{case_name}.mp4"
    if not video.exists():
        return {}
    n = frame_count(video)
    picks = [0, max(0, n // 2), max(0, n - 1)]
    crops = []
    for i, frame_no in enumerate(picks):
        out = frame_dir / f"{case_name}_{i}.png"
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-vf", f"select=eq(n\\,{frame_no})",
            "-vframes", "1", str(out),
        ], check=False)
        if out.exists():
            with Image.open(out) as im:
                im = im.convert("RGB")
                crop = crop_mouth(im).resize((220, 165), Image.Resampling.LANCZOS)
                crop.save(mouth_dir / f"{case_name}_{i}.png")
                crops.append(np.asarray(crop, dtype=np.float32) / 255.0)
    if len(crops) < 2:
        return {}
    mid = crops[1]
    first = crops[0]
    dark = (mid.mean(axis=2) < 0.22).mean()
    red = np.maximum(mid[:, :, 0] - (mid[:, :, 1] + mid[:, :, 2]) * 0.5, 0).mean()
    motion = np.abs(mid - first).mean()
    return {
        "mouth_dark_ratio_mid": float(dark),
        "mouth_red_score_mid": float(red),
        "mouth_delta_first_mid": float(motion),
    }


def make_grid(src_dir: Path, out: Path, tile_w: int, tile_h: int) -> None:
    cells = []
    for item in items:
        case_name = item["case_name"]
        for i in range(3):
            p = src_dir / f"{case_name}_{i}.png"
            if not p.exists():
                continue
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((tile_w, tile_h - 28), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (tile_w, tile_h), "white")
                canvas.paste(im, ((tile_w - im.width) // 2, 24))
                draw = ImageDraw.Draw(canvas)
                draw.text((4, 4), f"{case_name} f{i}", fill=(0, 0, 0), font=ImageFont.load_default())
                cells.append(canvas)
    if not cells:
        return
    cols = min(9, len(cells))
    rows = (len(cells) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for idx, cell in enumerate(cells):
        grid.paste(cell, ((idx % cols) * tile_w, (idx // cols) * tile_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)


for item in items:
    item.update(status.get(item["case_name"], {"status": "not_run"}))
    item["metric"] = metrics_for(item["case_name"])

make_grid(frame_dir, grid_path, 220, 220)
make_grid(mouth_dir, mouth_grid_path, 220, 190)
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

counts = Counter(item.get("status", "not_run") for item in items)
rendered = [x for x in items if x.get("status") in {"ok", "skipped_existing"}]

manual_verdicts = {
    "smile_md_001": "weak natural-ish / subtle teeth, best conservative seed",
    "smile_md_002": "too weak / slight tension",
    "smile_md_003": "weak gentle teeth, usable seed but not natural enough",
    "smile_md_004": "too weak / neutral",
    "smile_md_005": "most visible teeth, still tense and not relaxed smile",
    "smile_md_006": "grimace-prone / tense mouth",
    "smile_md_007": "too busy / tense lower face",
    "smile_md_008": "too weak",
    "smile_md_009": "too weak",
    "smile_md_010": "too weak",
    "smile_md_011": "visible mouth response, slight grimace",
    "smile_md_012": "too weak",
    "smile_md_013": "visible response, neutral/concerned not smile",
    "smile_md_014": "tense / not smile",
    "smile_md_015": "visible teeth, not enough corner lift",
    "smile_md_016": "too weak",
    "smile_md_017": "tense / grimace-like",
    "smile_md_018": "too weak",
    "smile_md_019": "open mouth / not smile",
    "smile_md_020": "weak visible response, not natural smile",
}
for item in items:
    if item["case_name"] in manual_verdicts:
        item["manual_label"] = manual_verdicts[item["case_name"]]

lines = [
    "# FastAvatar Multi-Dim Smile Search",
    "",
    f"Output directory: `{manifest['output_dir']}`",
    f"Grid: `{grid_path.name}`" if grid_path.exists() else "Grid: not generated yet",
    f"Mouth crop grid: `{mouth_grid_path.name}`" if mouth_grid_path.exists() else "Mouth crop grid: not generated yet",
    f"Manifest: `{manifest_path}`",
    f"Render status: `{render_status_path}`",
    "",
    "## Setup",
    "",
    f"- candidates: `{manifest['num_candidates']}`",
    f"- seed: `{manifest['seed']}`",
    f"- source primary dims: `{manifest['source_probe_dims']['primary_pool']}`",
    f"- source secondary dims: `{manifest['source_probe_dims']['secondary_pool']}`",
    f"- excluded as leaders from one-hot pass: `{manifest['source_probe_dims']['excluded_bad_leaders']}`",
    f"- render status counts: `{dict(counts)}`",
    "- each candidate edits `expr(100D)` directly, uses ramp-hold-release, fixes head/neck/rotation to zero, and sets a small jaw x in `[0.005, 0.035]`.",
    "",
    "## Candidate Table",
    "",
    "| case | status | jaw_x | expr_l2 | dims | metric dark | metric delta | video | label |",
    "|---|---|---:|---:|---|---:|---:|---|---|",
]
for item in items:
    dims = ", ".join(f"{t['dim']}:{t['weight']:+.2f}" for t in item["terms"])
    metric = item.get("metric") or {}
    dark = metric.get("mouth_dark_ratio_mid", 0.0)
    delta = metric.get("mouth_delta_first_mid", 0.0)
    video = item.get("video") or f"fastavatar_video/{item['case_name']}.mp4"
    lines.append(
        f"| `{item['case_name']}` | `{item.get('status', 'not_run')}` | {item['jaw_x']:.3f} | "
        f"{item.get('expr_l2', 0.0):.3f} | `{dims}` | {dark:.4f} | {delta:.4f} | "
        f"[video]({video}) | {item.get('manual_label', 'TODO inspect grid/video')} |"
    )

lines += [
    "",
    "## Questions",
    "",
]
if rendered:
    lines += [
        "- multi-dim vs single-dim: multi-dim is visibly stronger than the one-hot probe. It produces clearer lip opening and occasional light teeth, so coordinated expression vectors are more effective than isolated dimensions.",
        "- natural smile candidate: no candidate in this first 20 reads as a clean natural default smile. The best cases are still subtle, tense, or missing clear mouth-corner lift.",
        "- recommended top 3 expr100 vector + jaw for a second pass: `smile_md_001`, `smile_md_003`, and `smile_md_005`. Treat these as optimization seeds, not final primitives.",
        "- whether to store best vector as FastAvatar-specific smile primitive: not yet. `smile_md_001` is the safest conservative seed, but it is still too weak/neutral for the requested default smile.",
        "- fallback: if a second pass around these seeds still fails, move to a higher-quality expression source such as MEAD or another curated smile dataset before changing the P7/P8 default primitive.",
    ]
else:
    lines += [
        "- multi-dim vs single-dim: pending render.",
        "- natural smile candidate: pending render.",
        "- recommended top 3 expr100 vector + jaw: pending render.",
        "- whether to store best vector as FastAvatar-specific smile primitive: pending render.",
    ]
lines += [
    "",
    "## Manual Visual Review",
    "",
    "- inspected `smile_multidim_grid.png` and `smile_multidim_mouth_crop_grid.png`.",
    "- overall: multi-dimensional blends do activate the mouth more than single dimensions, but the face remains largely neutral/concerned and lacks robust upward mouth-corner pull.",
    "- gentle teeth: visible in `smile_md_001`, `smile_md_003`, `smile_md_005`, `smile_md_011`, and `smile_md_015`, but these read as weak/tense mouth opening rather than a relaxed smile.",
    "- grimace / lip tightening: `smile_md_006`, `smile_md_007`, `smile_md_014`, and `smile_md_017` are too tense or busy for a default smile.",
    "- neutral / too weak: many candidates, especially `smile_md_004`, `smile_md_008`, `smile_md_009`, `smile_md_010`, `smile_md_012`, `smile_md_016`, and `smile_md_018`, remain close to neutral.",
    "- best seed: `smile_md_001` because it is the least forced and has mild mouth response with jaw_x=0.018.",
    "- stronger but risky seed: `smile_md_005` because it has the most visible teeth/opening, but it is still tense and not natural enough.",
    "- recommendation: do not store a default FastAvatar-specific smile primitive from this first pass. Run a focused second pass around `smile_md_001`/`003`/`005` with stronger corner-lift ingredients, or introduce MEAD/curated smile vectors if that still fails.",
]

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[SmileMultiDim] report={report_path}")
PY

printf '[SmileMultiDim] manifest=%s\n' "${MANIFEST}"
printf '[SmileMultiDim] report=%s\n' "${REPORT}"
if [[ -f "${GRID}" ]]; then
  printf '[SmileMultiDim] grid=%s\n' "${GRID}"
fi
if [[ -f "${MOUTH_GRID}" ]]; then
  printf '[SmileMultiDim] mouth_grid=%s\n' "${MOUTH_GRID}"
fi
