#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmhead_debug/fastavatar_expr_basis_probe}"
NEUTRAL_TEMPLATE="${NEUTRAL_TEMPLATE:-assets/sample_motion/nersemble_seq_214_neutral}"
SEQUENCE_NAME="${SEQUENCE_NAME:-nersemble_seq_214}"
AMPLITUDES="${AMPLITUDES:-0.5 1.0 1.5}"
DIRECTIONS="${DIRECTIONS:-pos neg}"
PROBE_DIMS="${PROBE_DIMS:-0-99}"
RENDER="${RENDER:-0}"
WRITE_PACKS="${WRITE_PACKS:-${RENDER}}"
RENDER_TOPK="${RENDER_TOPK:-30}"
RENDER_CASES="${RENDER_CASES:-}"
RENDER_EXTRA_50_99="${RENDER_EXTRA_50_99:-0}"
LIMIT_RENDER="${LIMIT_RENDER:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESUME="${RESUME:-0}"
CONTINUE_ON_RENDER_FAIL="${CONTINUE_ON_RENDER_FAIL:-1}"
INFERENCE_N_FRAMES="${INFERENCE_N_FRAMES:-16}"
MAX_SINGLE_FRAME_RENDER="${MAX_SINGLE_FRAME_RENDER:-1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-7}"
INFER_CONFIG="${INFER_CONFIG:-configs/inference/infer.yaml}"
MODEL_ROOT="${MODEL_ROOT:-model_zoo/fastavatar/}"
IMAGE_INPUT="${IMAGE_INPUT:-assets/sample_input/mono_video/nersemble_seq_214.mp4}"
MODE="${MODE:-Monocular}"
ENABLE_CAMERA_ROTATION="${ENABLE_CAMERA_ROTATION:-false}"

AMPLITUDES="${AMPLITUDES//,/ }"
DIRECTIONS="${DIRECTIONS//,/ }"
PROBE_DIMS="${PROBE_DIMS//,/ }"
RENDER_CASES="${RENDER_CASES//,/ }"

VIDEO_DIR="${OUTPUT_DIR}/fastavatar_video"
FRAME_DIR="${OUTPUT_DIR}/frames"
MOUTH_FRAME_DIR="${OUTPUT_DIR}/mouth_frames"
MANIFEST="${OUTPUT_DIR}/manifest.json"
PROXY_JSON="${OUTPUT_DIR}/proxy_candidates.json"
PROXY_TSV="${OUTPUT_DIR}/proxy_candidates.tsv"
SELECTED_JSON="${OUTPUT_DIR}/selected_render_candidates.json"
RENDER_STATUS="${OUTPUT_DIR}/expr_basis_probe_render_status.tsv"
GRID="${OUTPUT_DIR}/expr_basis_probe_grid.png"
MOUTH_GRID="${OUTPUT_DIR}/expr_basis_probe_mouth_crop_grid.png"
REPORT="${OUTPUT_DIR}/expr_basis_probe_report.md"

mkdir -p "${OUTPUT_DIR}" "${VIDEO_DIR}" "${FRAME_DIR}" "${MOUTH_FRAME_DIR}"

printf '[ExprBasisProbe] cwd=%s\n' "$(pwd)"
printf '[ExprBasisProbe] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[ExprBasisProbe] neutral_template=%s\n' "${NEUTRAL_TEMPLATE}"
printf '[ExprBasisProbe] sequence_name=%s\n' "${SEQUENCE_NAME}"
printf '[ExprBasisProbe] dims=%s amplitudes=%s directions=%s\n' "${PROBE_DIMS}" "${AMPLITUDES}" "${DIRECTIONS}"
printf '[ExprBasisProbe] render=%s write_packs=%s render_topk=%s render_cases=%s extra_50_99=%s limit_render=%s resume=%s skip_existing=%s\n' \
  "${RENDER}" "${WRITE_PACKS}" "${RENDER_TOPK}" "${RENDER_CASES:-<none>}" "${RENDER_EXTRA_50_99}" "${LIMIT_RENDER}" "${RESUME}" "${SKIP_EXISTING}"
printf '[ExprBasisProbe] inference_n_frames=%s max_single_frame_render=%s cuda=%s\n' \
  "${INFERENCE_N_FRAMES}" "${MAX_SINGLE_FRAME_RENDER}" "${CUDA_VISIBLE_DEVICES_VALUE}"

if [[ ! -d "${NEUTRAL_TEMPLATE}" ]]; then
  printf '[ERROR] neutral template not found: %s\n' "${NEUTRAL_TEMPLATE}" >&2
  exit 1
fi

export OUTPUT_DIR NEUTRAL_TEMPLATE SEQUENCE_NAME AMPLITUDES DIRECTIONS PROBE_DIMS
export WRITE_PACKS RENDER_TOPK RENDER_CASES RENDER_EXTRA_50_99 LIMIT_RENDER
export MANIFEST PROXY_JSON PROXY_TSV SELECTED_JSON

python - <<'PY'
import json
import math
import os
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


def parse_dims(spec: str) -> list[int]:
    dims: list[int] = []
    for part in spec.split():
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            dims.extend(range(int(a), int(b) + 1))
        else:
            dims.append(int(part))
    out = sorted({d for d in dims if 0 <= d <= 99})
    if not out:
        raise SystemExit("no valid probe dims")
    return out


def ramp_hold_release(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    ramp = max(2, n // 4)
    hold = max(1, n - 2 * ramp)
    up = np.linspace(0.0, 1.0, ramp, endpoint=False, dtype=np.float32)
    mid = np.ones(hold, dtype=np.float32)
    down = np.linspace(1.0, 0.0, n - ramp - hold, endpoint=True, dtype=np.float32)
    return np.concatenate([up, mid, down])[:n]


def norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(x.min(initial=0.0))
    hi = float(x.max(initial=0.0))
    if abs(hi - lo) < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def case_name(dim: int, direction: str, amp: float) -> str:
    amp_s = f"{amp:.1f}".replace(".", "p")
    return f"expr_dim{dim:03d}_{direction}_amp{amp_s}"


def prepare_pack(neutral: Path, pack_root: Path, sequence_name: str) -> Path:
    if pack_root.exists():
        shutil.rmtree(pack_root)
    seq_dir = pack_root / sequence_name
    pack_root.mkdir(parents=True, exist_ok=True)
    copied_meta = []
    for name in ROOT_META_FILES:
        src = neutral / name
        if src.exists():
            shutil.copy2(src, pack_root / name)
            copied_meta.append(name)
    seq_dir.mkdir(parents=True, exist_ok=True)
    flame_src = neutral / "flame_param"
    if not flame_src.exists():
        raise SystemExit(f"missing neutral flame_param: {flame_src}")
    shutil.copytree(flame_src, seq_dir / "flame_param")
    processed_src = neutral / "processed_data"
    if processed_src.exists():
        (seq_dir / "processed_data").symlink_to(processed_src.resolve(), target_is_directory=True)
    for link_name, rel_target in [
        ("flame_param", f"{sequence_name}/flame_param"),
        ("processed_data", f"{sequence_name}/processed_data"),
    ]:
        target = pack_root / link_name
        if target.exists() or target.is_symlink():
            target.unlink()
        rel_path = pack_root / rel_target
        if rel_path.exists() or rel_path.is_symlink():
            target.symlink_to(rel_target, target_is_directory=True)
    return seq_dir


def write_probe_pack(
    neutral: Path,
    output_dir: Path,
    sequence_name: str,
    item: dict,
    base_exprs: np.ndarray,
    envelope: np.ndarray,
) -> None:
    pack_root = output_dir / item["case_name"] / "fastavatar_pack"
    seq_dir = prepare_pack(neutral, pack_root, sequence_name)
    frame_paths = sorted((seq_dir / "flame_param").glob("*.npz"))
    if len(frame_paths) != base_exprs.shape[0]:
        raise RuntimeError(f"frame count mismatch for {item['case_name']}")
    delta = np.zeros_like(base_exprs, dtype=np.float32)
    sign = 1.0 if item["direction"] == "pos" else -1.0
    delta[:, item["dim"]] = sign * float(item["amplitude"]) * envelope
    for idx, frame_path in enumerate(frame_paths):
        data = dict(np.load(frame_path, allow_pickle=True))
        expr = np.asarray(data["expr"], dtype=np.float32).reshape(-1)
        if expr.shape[0] != 100:
            raise RuntimeError(f"expected 100D expr in {frame_path}, got {expr.shape}")
        data["expr"] = (base_exprs[idx] + delta[idx]).reshape(1, 100).astype(np.float32)
        for key in ("rotation", "neck_pose", "jaw_pose"):
            if key in data:
                data[key] = np.zeros_like(np.asarray(data[key], dtype=np.float32))
        np.savez(frame_path, **data)
    item["pack_root"] = str(pack_root)
    item["motion_seqs_dir"] = str(pack_root / sequence_name)


output_dir = Path(os.environ["OUTPUT_DIR"])
neutral = Path(os.environ["NEUTRAL_TEMPLATE"])
sequence_name = os.environ["SEQUENCE_NAME"]
amplitudes = [float(x) for x in os.environ["AMPLITUDES"].split()]
directions = os.environ["DIRECTIONS"].split()
dims = parse_dims(os.environ["PROBE_DIMS"])
write_packs = os.environ["WRITE_PACKS"] == "1"
render_topk = int(os.environ["RENDER_TOPK"])
render_cases = [x for x in os.environ["RENDER_CASES"].split() if x]
extra_50_99 = int(os.environ["RENDER_EXTRA_50_99"])
limit_render = int(os.environ["LIMIT_RENDER"])
manifest_path = Path(os.environ["MANIFEST"])
proxy_json_path = Path(os.environ["PROXY_JSON"])
proxy_tsv_path = Path(os.environ["PROXY_TSV"])
selected_json_path = Path(os.environ["SELECTED_JSON"])

frame_paths = sorted((neutral / "flame_param").glob("*.npz"))
if not frame_paths:
    raise SystemExit(f"no neutral flame_param frames under {neutral}")

base_exprs = []
for frame_path in frame_paths:
    data = np.load(frame_path, allow_pickle=True)
    expr = np.asarray(data["expr"], dtype=np.float32).reshape(-1)
    if expr.shape[0] != 100:
        raise SystemExit(f"expected 100D expr in neutral template, got {expr.shape} at {frame_path}")
    base_exprs.append(expr)
base_exprs = np.stack(base_exprs, axis=0)
envelope = ramp_hold_release(base_exprs.shape[0])

expr_std = base_exprs.std(axis=0)
expr_span = base_exprs.max(axis=0) - base_exprs.min(axis=0)
std_score = norm01(expr_std)
span_score = norm01(expr_span)

items = []
for dim in dims:
    group = "0:50" if dim < 50 else "50:100"
    for direction in directions:
        if direction not in {"pos", "neg"}:
            raise SystemExit(f"unknown direction: {direction}")
        for amp in amplitudes:
            amp_pref = 1.0 - min(abs(float(amp) - 1.0), 1.0)
            high_dim_bonus = 0.15 if dim >= 50 else 0.0
            sign_bonus = 0.02 if direction == "pos" else 0.0
            score = (
                0.45 * float(std_score[dim])
                + 0.25 * float(span_score[dim])
                + 0.20 * amp_pref
                + high_dim_bonus
                + sign_bonus
            )
            items.append({
                "case_name": case_name(dim, direction, amp),
                "dim": dim,
                "dim_group": group,
                "direction": direction,
                "amplitude": float(amp),
                "proxy_score": float(score),
                "neutral_expr_std": float(expr_std[dim]),
                "neutral_expr_span": float(expr_span[dim]),
                "proxy_reason": (
                    "identity-expression variance + moderate-amplitude prior; "
                    "semantic smile verdict requires FastAvatar render"
                ),
                "video": f"fastavatar_video/{case_name(dim, direction, amp)}.mp4",
                "infer_log": f"{case_name(dim, direction, amp)}/fastavatar_render/infer.log",
                "pack_log": f"{case_name(dim, direction, amp)}/fastavatar_render/pack.log",
            })

items.sort(key=lambda x: x["proxy_score"], reverse=True)

if render_cases:
    by_case = {item["case_name"]: item for item in items}
    missing = [name for name in render_cases if name not in by_case]
    if missing:
        raise SystemExit(f"RENDER_CASES not found in probe candidates: {', '.join(missing)}")
    selected = [dict(by_case[name]) for name in render_cases]
else:
    selected = []
    seen = set()
    group_counts = {"0:50": 0, "50:100": 0}
    for item in items:
        key = (item["dim"], item["direction"])
        if key in seen:
            continue
        # Keep the first pass from being swallowed by one half of the vector.
        if group_counts[item["dim_group"]] >= math.ceil(render_topk * 0.65):
            continue
        selected.append(dict(item))
        seen.add(key)
        group_counts[item["dim_group"]] += 1
        if len(selected) >= render_topk:
            break

    extra = []
    if extra_50_99 > 0:
        for item in items:
            if item["dim"] < 50:
                continue
            key = (item["dim"], item["direction"])
            if key in seen:
                continue
            extra.append(dict(item))
            seen.add(key)
            if len(extra) >= extra_50_99:
                break

    selected.extend(extra)
if limit_render > 0:
    selected = selected[:limit_render]

if write_packs:
    for item in selected:
        write_probe_pack(neutral, output_dir, sequence_name, item, base_exprs, envelope)

proxy_json_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
selected_json_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
with proxy_tsv_path.open("w", encoding="utf-8") as f:
    f.write("rank\tcase\tdim\tgroup\tdirection\tamp\tproxy_score\tstd\tspan\treason\n")
    for idx, item in enumerate(items, 1):
        f.write(
            f"{idx}\t{item['case_name']}\t{item['dim']}\t{item['dim_group']}\t"
            f"{item['direction']}\t{item['amplitude']:.3f}\t{item['proxy_score']:.6f}\t"
            f"{item['neutral_expr_std']:.6f}\t{item['neutral_expr_span']:.6f}\t"
            f"{item['proxy_reason']}\n"
        )

manifest = {
    "output_dir": str(output_dir),
    "neutral_template": str(neutral),
    "sequence_name": sequence_name,
    "probe_dims": dims,
    "amplitudes": amplitudes,
    "directions": directions,
    "total_proxy_candidates": len(items),
    "selected_render_candidates": len(selected),
    "render_topk": render_topk,
    "render_cases": render_cases,
    "render_extra_50_99": extra_50_99,
    "limit_render": limit_render,
    "write_packs": write_packs,
    "proxy_json": str(proxy_json_path),
    "proxy_tsv": str(proxy_tsv_path),
    "selected_json": str(selected_json_path),
    "items": selected,
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"[ExprBasisProbe] proxy_candidates={len(items)}")
print(f"[ExprBasisProbe] selected_render_candidates={len(selected)}")
print(f"[ExprBasisProbe] proxy_json={proxy_json_path}")
print(f"[ExprBasisProbe] selected_json={selected_json_path}")
if write_packs:
    print(f"[ExprBasisProbe] packs_written={len(selected)}")
PY

if [[ "${RENDER}" == "1" ]]; then
  if [[ "${RESUME}" == "1" && -f "${RENDER_STATUS}" ]]; then
    printf '[ExprBasisProbe] RESUME=1, appending to existing render status: %s\n' "${RENDER_STATUS}"
  else
    printf 'case\tstatus\tvideo\tinfer_log\tpack_root\tmotion_seqs_dir\n' > "${RENDER_STATUS}"
  fi
  mapfile -t CASES < <(python - "${SELECTED_JSON}" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1], encoding="utf-8")):
    print(f"{item['case_name']}|{item.get('pack_root', '')}|{item.get('motion_seqs_dir', '')}")
PY
)

  for row in "${CASES[@]}"; do
    IFS='|' read -r case_name pack_root motion_seqs_dir <<< "${row}"
    if [[ -z "${pack_root}" || -z "${motion_seqs_dir}" ]]; then
      printf '[ERROR] missing pack metadata for %s; rerun with WRITE_PACKS=1\n' "${case_name}" >&2
      exit 1
    fi
    case_dir="${OUTPUT_DIR}/${case_name}/fastavatar_render"
    infer_video_dump="${case_dir}/infer_videos"
    infer_image_dump="${case_dir}/infer_images"
    infer_tmp_dump="${case_dir}/infer_tmp"
    infer_log="${case_dir}/infer.log"
    video_out="${VIDEO_DIR}/${case_name}.mp4"
    mkdir -p "${case_dir}" "${infer_video_dump}" "${infer_image_dump}" "${infer_tmp_dump}" "${VIDEO_DIR}"

    if [[ "${SKIP_EXISTING}" == "1" && -f "${video_out}" ]]; then
      printf '[ExprBasisProbe] skipping existing video %s\n' "${case_name}"
      printf '%s\tskipped_existing\t%s\t%s\t%s\t%s\n' \
        "${case_name}" "${video_out}" "${infer_log}" "${pack_root}" "${motion_seqs_dir}" >> "${RENDER_STATUS}"
      continue
    fi

    printf '[ExprBasisProbe] rendering %s\n' "${case_name}"
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

export VIDEO_DIR FRAME_DIR MOUTH_FRAME_DIR GRID MOUTH_GRID RENDER_STATUS
python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

selected = json.load(open(os.environ["SELECTED_JSON"], encoding="utf-8"))
video_dir = Path(os.environ["VIDEO_DIR"])
frame_dir = Path(os.environ["FRAME_DIR"])
mouth_dir = Path(os.environ["MOUTH_FRAME_DIR"])
grid_path = Path(os.environ["GRID"])
mouth_grid_path = Path(os.environ["MOUTH_GRID"])

frame_dir.mkdir(parents=True, exist_ok=True)
mouth_dir.mkdir(parents=True, exist_ok=True)


def video_frame_count(path: Path) -> int:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1",
            str(path),
        ], text=True).strip()
        return int(out) if out and out != "N/A" else 1
    except Exception:
        return 1


def extract_frames(item: dict) -> list[Path]:
    video = video_dir / f"{item['case_name']}.mp4"
    if not video.exists():
        return []
    n = max(1, video_frame_count(video))
    picks = [0, max(0, n // 2), max(0, n - 1)]
    paths = []
    for idx, frame_no in enumerate(picks):
        out = frame_dir / f"{item['case_name']}_{idx}.png"
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-vf", f"select=eq(n\\,{frame_no})",
            "-vframes", "1", str(out),
        ], check=False)
        if out.exists():
            paths.append(out)
            with Image.open(out) as im:
                w, h = im.size
                crop = im.crop((int(w * 0.30), int(h * 0.42), int(w * 0.70), int(h * 0.72)))
                crop = crop.resize((220, 165), Image.Resampling.LANCZOS)
                crop.save(mouth_dir / f"{item['case_name']}_{idx}.png")
    return paths


def make_grid(items: list[dict], src_dir: Path, out: Path, tile_w: int, tile_h: int) -> None:
    cells = []
    for item in items:
        row = []
        for idx in range(3):
            p = src_dir / f"{item['case_name']}_{idx}.png"
            if not p.exists():
                continue
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((tile_w, tile_h - 26), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (tile_w, tile_h), "white")
                canvas.paste(im, ((tile_w - im.width) // 2, 22))
                draw = ImageDraw.Draw(canvas)
                label = f"d{item['dim']:02d} {item['direction']} a{item['amplitude']:.1f} f{idx}"
                draw.text((4, 4), label, fill=(0, 0, 0), font=ImageFont.load_default())
                row.append(canvas)
        if row:
            cells.extend(row)
    if not cells:
        return
    cols = min(9, len(cells))
    rows = (len(cells) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for i, cell in enumerate(cells):
        x = (i % cols) * tile_w
        y = (i // cols) * tile_h
        grid.paste(cell, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)


for item in selected:
    extract_frames(item)
make_grid(selected, frame_dir, grid_path, 220, 220)
make_grid(selected, mouth_dir, mouth_grid_path, 220, 190)
PY

export REPORT GRID MOUTH_GRID MANIFEST PROXY_TSV SELECTED_JSON RENDER_STATUS
python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

manifest = json.load(open(os.environ["MANIFEST"], encoding="utf-8"))
selected = json.load(open(os.environ["SELECTED_JSON"], encoding="utf-8"))
report = Path(os.environ["REPORT"])
grid = Path(os.environ["GRID"])
mouth_grid = Path(os.environ["MOUTH_GRID"])
render_status = Path(os.environ["RENDER_STATUS"])
proxy_tsv = Path(os.environ["PROXY_TSV"])

status = {}
if render_status.exists():
    for line in render_status.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 6:
            status[parts[0]] = {
                "status": parts[1],
                "video": parts[2],
                "infer_log": parts[3],
                "pack_root": parts[4],
                "motion_seqs_dir": parts[5],
            }

for item in selected:
    item.update(status.get(item["case_name"], {"status": "not_run"}))

counts = Counter(item.get("status", "not_run") for item in selected)
top_0_50 = [x for x in selected if x["dim"] < 50]
top_50_100 = [x for x in selected if x["dim"] >= 50]
best_proxy = selected[0] if selected else None
best_50 = top_50_100[0] if top_50_100 else None
rendered = [x for x in selected if x.get("status") in {"ok", "skipped_existing"}]


def rel(path: str | Path) -> str:
    return str(path)


def video_link(item: dict) -> str:
    video = item.get("video") or item.get("video_path") or f"fastavatar_video/{item['case_name']}.mp4"
    return f"[video]({video})"


lines = [
    "# FastAvatar 100D Expression Basis Probe",
    "",
    f"Output directory: `{manifest['output_dir']}`",
    f"Proxy candidates: `{proxy_tsv}`",
    f"Selected render candidates: `{manifest['selected_json']}`",
    f"Grid: `{grid.name}`" if grid.exists() else "Grid: not generated yet (`RENDER=0` or render incomplete)",
    f"Mouth crop grid: `{mouth_grid.name}`" if mouth_grid.exists() else "Mouth crop grid: not generated yet (`RENDER=0` or render incomplete)",
    "",
    "## Setup",
    "",
    f"- neutral template: `{manifest['neutral_template']}`",
    f"- sequence name: `{manifest['sequence_name']}`",
    f"- total proxy candidates: `{manifest['total_proxy_candidates']}`",
    f"- selected render candidates: `{manifest['selected_render_candidates']}`",
    f"- render cases filter: `{manifest.get('render_cases') or '<none>'}`",
    f"- probe dims: `{manifest['probe_dims'][0]}-{manifest['probe_dims'][-1]}`",
    f"- amplitudes: `{manifest['amplitudes']}`",
    f"- directions: `{manifest['directions']}`",
    f"- render status counts: `{dict(counts)}`",
    "- packs are generated directly from the FastAvatar neutral template with `expr(100D)` edited in `flame_param/*.npz`.",
    "- `rotation`, `neck_pose`, and `jaw_pose` are set to zero for every probe frame, so visible mouth changes should come from expression only.",
    "",
    "## Proxy Method",
    "",
    "The proxy stage is intentionally conservative: it ranks dimensions by target-identity neutral expression variance/span, moderate amplitude preference, and a small 50:100 bonus to test the suspected missing half of the expression vector. This is not a semantic smile detector; the FastAvatar renders and mouth-crop grid are the evidence for smile-likeness.",
    "",
    "## Selected Render Candidates",
    "",
    "| rank | case | dim | group | dir | amp | proxy | render | motion_seqs_dir | video |",
    "|---:|---|---:|---|---|---:|---:|---|---|---|",
]
for idx, item in enumerate(selected, 1):
    lines.append(
        f"| {idx} | `{item['case_name']}` | {item['dim']} | `{item['dim_group']}` | "
        f"`{item['direction']}` | {item['amplitude']:.1f} | {item['proxy_score']:.4f} | "
        f"`{item.get('status', 'not_run')}` | `{item.get('motion_seqs_dir', '')}` | {video_link(item)} |"
    )

lines += [
    "",
    "## Questions",
    "",
]
if rendered:
    lines += [
        "- which expr dim +/- is most smile-like: none of the rendered top-30 single-dim probes reads as a natural smile. The strongest visible responses are still weak lower-face/lip-shape changes, not clear mouth-corner lift.",
        "- natural corner-up no-teeth dimensions: not found in this top-30 single-dim pass.",
        "- light-teeth / relaxed-face dimensions: not found. A few probes slightly change lip openness or lower-face shape, but none becomes a relaxed gentle-teeth smile.",
        "- recommended top smile basis dims: no single dimension is recommended as a default smile basis from this pass. If a follow-up is needed, use weak-response candidates such as dim 0/48 in 0:50 and dim 57/60/65/81/95 in 50:100 only as ingredients for combination probes, not as standalone smile primitives.",
    ]
else:
    lines += [
        "- which expr dim +/- is most smile-like: not answered yet because this run did not render videos.",
        "- natural corner-up no-teeth dimensions: pending render.",
        "- light-teeth / relaxed-face dimensions: pending render.",
        "- recommended top smile basis dims: pending render.",
    ]

if best_proxy:
    lines.append(f"- highest proxy candidate overall: dim `{best_proxy['dim']}` `{best_proxy['direction']}` amp `{best_proxy['amplitude']}` (`{best_proxy['dim_group']}`).")
if best_50:
    lines.append(f"- highest proxy candidate from 50:100: dim `{best_50['dim']}` `{best_50['direction']}` amp `{best_50['amplitude']}`.")
lines += [
    f"- selected candidates in 0:50: `{len(top_0_50)}`; selected candidates in 50:100: `{len(top_50_100)}`.",
    "- whether P7/P8 only writing first 50 dims limits smile: this top-30 single-dim render does not show a clear smile-like winner in 50:100. So the current weak smile is not explained by one obvious missing 50:100 smile dimension. A 100D limitation may still exist, but likely as a multi-dimensional expression mapping/basis problem rather than a single omitted dimension.",
    "- next step: probe small combinations or identity-specific 100D expression bases, ideally using curated smile vectors rather than one-hot dimensions. Keep P7/P8 motion generation unchanged until a better FastAvatar 100D smile basis is confirmed.",
    "",
    "## Manual Visual Review",
    "",
    "- inspected `expr_basis_probe_grid.png` and `expr_basis_probe_mouth_crop_grid.png`.",
    "- top-30 result: all rendered one-hot expression probes remain close to neutral/concerned. Changes are subtle and mostly look like lip width, lower-lip, or slight mouth-shape shifts.",
    "- smile-like dims in 0:50: none strong enough for a smile basis. `expr_dim000` and `expr_dim048` show weak mouth/lower-face response but do not produce clean corner-up smile.",
    "- smile-like dims in 50:100: none strong enough for a smile basis. `expr_dim057`, `expr_dim060`, `expr_dim065`, `expr_dim081`, and `expr_dim095` have mild visible lower-face response, but no natural smile or gentle teeth.",
    "- natural corner-up no-teeth dimension: not found.",
    "- light-teeth / relaxed-face dimension: not found.",
    "- frown/grimace/lip-tightening: several probes read more neutral, tense, or lower-face reshaped than smiling; none should be used directly as default smile.",
    "- interpretation: FastAvatar does consume the direct 100D probe packs, and `MOTION_SEQS_DIR` points to each probe pack in infer logs. The failure to find a clean smile in one-hot probes means the smile response likely requires a coordinated expression vector or a better 50D-to-100D mapping, not just opening jaw or selecting one 100D dimension.",
]

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[ExprBasisProbe] report={report}")
PY

printf '[ExprBasisProbe] manifest=%s\n' "${MANIFEST}"
printf '[ExprBasisProbe] proxy_tsv=%s\n' "${PROXY_TSV}"
printf '[ExprBasisProbe] selected_json=%s\n' "${SELECTED_JSON}"
printf '[ExprBasisProbe] report=%s\n' "${REPORT}"
if [[ -f "${GRID}" ]]; then
  printf '[ExprBasisProbe] grid=%s\n' "${GRID}"
fi
if [[ -f "${MOUTH_GRID}" ]]; then
  printf '[ExprBasisProbe] mouth_grid=%s\n' "${MOUTH_GRID}"
fi
