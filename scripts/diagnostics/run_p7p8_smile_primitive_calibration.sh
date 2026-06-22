#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmhead_debug/p7p8_smile_calibration}"
MANIFEST="${MANIFEST:-outputs/mmhead_debug/motion_dataset_v1_ae_debug/manifest.jsonl}"
CODEBOOK="${CODEBOOK:-outputs/mmhead_debug/codebook_full.jsonl}"
CURRENT_SMILE_NPZ="${CURRENT_SMILE_NPZ:-outputs/mmhead_debug/p7p8_quality_diagnosis/smile/motion_post_safety.npz}"
RENDER_WRAPPER="${RENDER_WRAPPER:-scripts/diagnostics/render_fastavatar_case.sh}"
NEUTRAL_TEMPLATE="${NEUTRAL_TEMPLATE:-assets/sample_motion/nersemble_seq_214_neutral}"
SEQUENCE_NAME="${SEQUENCE_NAME:-nersemble_seq_214}"
TOPK="${TOPK:-12}"
CANDIDATE_POOL="${CANDIDATE_POOL:-80}"
INFERENCE_N_FRAMES="${INFERENCE_N_FRAMES:-32}"
MAX_SINGLE_FRAME_RENDER="${MAX_SINGLE_FRAME_RENDER:-4}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-7}"
RENDER="${RENDER:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RERANK_MODE="${RERANK_MODE:-balanced}"
CONTINUE_ON_RENDER_FAIL="${CONTINUE_ON_RENDER_FAIL:-1}"

NPZ_DIR="${OUTPUT_DIR}/candidate_npz"
FRAME_DIR="${OUTPUT_DIR}/frames"
VIDEO_DIR="${OUTPUT_DIR}/fastavatar_video"
REPORT="${OUTPUT_DIR}/smile_calibration_report.md"
GRID="${OUTPUT_DIR}/smile_candidate_grid.png"
CANDIDATES_JSON="${OUTPUT_DIR}/smile_candidates.json"
RENDER_STATUS="${OUTPUT_DIR}/smile_render_status_${RERANK_MODE}.tsv"

if [[ "${RERANK_MODE}" == "clean" ]]; then
  NPZ_DIR="${OUTPUT_DIR}/clean_candidate_npz"
  FRAME_DIR="${OUTPUT_DIR}/clean_frames"
  REPORT="${OUTPUT_DIR}/smile_clean_calibration_report.md"
  GRID="${OUTPUT_DIR}/smile_clean_candidate_grid.png"
  CANDIDATES_JSON="${OUTPUT_DIR}/smile_clean_candidates.json"
fi

mkdir -p "${OUTPUT_DIR}" "${NPZ_DIR}" "${FRAME_DIR}" "${VIDEO_DIR}"

printf '[SmileCalibration] cwd=%s\n' "$(pwd)"
printf '[SmileCalibration] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[SmileCalibration] manifest=%s\n' "${MANIFEST}"
printf '[SmileCalibration] codebook=%s\n' "${CODEBOOK}"
printf '[SmileCalibration] topk=%s candidate_pool=%s render=%s rerank_mode=%s\n' "${TOPK}" "${CANDIDATE_POOL}" "${RENDER}" "${RERANK_MODE}"
printf '[SmileCalibration] cuda_visible_devices=%s inference_n_frames=%s max_single_frame_render=%s\n' "${CUDA_VISIBLE_DEVICES_VALUE}" "${INFERENCE_N_FRAMES}" "${MAX_SINGLE_FRAME_RENDER}"

if [[ "${RERANK_MODE}" != "balanced" && "${RERANK_MODE}" != "clean" ]]; then
  printf '[ERROR] RERANK_MODE must be balanced or clean; got: %s\n' "${RERANK_MODE}" >&2
  exit 2
fi

if [[ ! -f "${MANIFEST}" ]]; then
  printf '[ERROR] manifest not found: %s\n' "${MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${CODEBOOK}" ]]; then
  printf '[ERROR] codebook not found: %s\n' "${CODEBOOK}" >&2
  exit 1
fi
if [[ ! -f "${CURRENT_SMILE_NPZ}" ]]; then
  printf '[ERROR] current smile npz not found: %s\n' "${CURRENT_SMILE_NPZ}" >&2
  exit 1
fi
if [[ "${RENDER}" == "1" && ! -f "${RENDER_WRAPPER}" ]]; then
  printf '[ERROR] render wrapper not found: %s\n' "${RENDER_WRAPPER}" >&2
  exit 1
fi

python - "${MANIFEST}" "${CODEBOOK}" "${CURRENT_SMILE_NPZ}" "${NPZ_DIR}" "${CANDIDATES_JSON}" "${TOPK}" "${CANDIDATE_POOL}" "${RERANK_MODE}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

manifest_path = Path(sys.argv[1])
codebook_path = Path(sys.argv[2])
current_smile_npz = Path(sys.argv[3])
npz_dir = Path(sys.argv[4])
candidates_json = Path(sys.argv[5])
topk = int(sys.argv[6])
candidate_pool = int(sys.argv[7])
rerank_mode = sys.argv[8]

POSITIVE = [
    "smile", "smiling", "smiles", "happy", "happiness",
    "laugh", "laughing", "grin", "cheerful",
]
CLEAN_SMILE = [
    "cheek raise", "cheeks raised", "raised cheeks", "lip corner",
    "corners pulled", "corners of the lips", "dimple", "dimples",
    "gentle smile", "subtle smile", "broad smile", "joyful",
    "joyful smile", "full joyful smile", "full, joyful smile",
    "cheeks lift", "cheek lifts", "corners of the lips pull",
    "corners of the mouth pull", "mouth corners pull",
]
JAW_OPEN = [
    "mouth open", "open mouth", "jaw open", "lips part", "lips parted",
    "parting", "tongue", "wide open",
]
LAUGH = [
    "laugh", "laughing", "laughs", "laughter",
]
TALK = [
    "talk", "talking", "speaks", "speaking", "speech", "says", "utter",
]
LIP_TIGHT = [
    "lip tightening", "lips tightly", "tight lips", "pressed lips",
    "lip press", "lips press", "compress", "pucker", "pushed forward",
    "hide lips", "lips together", "lips tighten", "lip tighten",
    "tightening of the lips", "tightening of lips",
]
NEGATIVE_EMOTION = [
    "neutral throughout", "remains neutral", "contempt", "frown",
    "angry", "sad", "fear", "disgust",
]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def text_score(text: str) -> tuple[float, dict]:
    low = text.lower()
    pos_hits = sum(low.count(k) for k in POSITIVE)
    clean_hits = sum(low.count(k) for k in CLEAN_SMILE)
    jaw_hits = sum(low.count(k) for k in JAW_OPEN)
    laugh_hits = sum(low.count(k) for k in LAUGH)
    talk_hits = sum(low.count(k) for k in TALK)
    tight_hits = sum(low.count(k) for k in LIP_TIGHT)
    neg_hits = sum(low.count(k) for k in NEGATIVE_EMOTION)
    direct = 0.0
    if re.search(r"\b(starts? to smile|begins? to smile|intensifies the smile|wide smile|broad smile|full smile|full, joyful smile|joyful smile|grin|laugh)", low):
        direct += 2.0
    if "happy throughout" in low or "looks happy" in low:
        direct += 1.0
    score = 1.2 * pos_hits + 1.8 * clean_hits + direct - 1.8 * tight_hits - 0.8 * jaw_hits - 1.0 * neg_hits
    return score, {
        "positive_hits": pos_hits,
        "clean_smile_hits": clean_hits,
        "jaw_open_text_hits": jaw_hits,
        "laugh_text_hits": laugh_hits,
        "talk_text_hits": talk_hits,
        "lip_tight_text_hits": tight_hits,
        "negative_text_hits": neg_hits,
        "direct_phrase_bonus": direct,
    }


def load_motion(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    key = "motion" if "motion" in data else "motion_raw" if "motion_raw" in data else data.files[0]
    motion = np.asarray(data[key], dtype=np.float32)
    if motion.ndim == 1:
        motion = motion[None, :]
    if motion.shape[1] < 56:
        raise ValueError(f"expected [T,>=56], got {motion.shape}")
    return motion[:, :56]


def fit_len(motion: np.ndarray, n: int = 32) -> np.ndarray:
    if motion.shape[0] == n:
        return motion.astype(np.float32)
    if motion.shape[0] > n:
        idx = np.linspace(0, motion.shape[0] - 1, n).round().astype(np.int64)
        return motion[idx].astype(np.float32)
    pad = np.repeat(motion[-1:], n - motion.shape[0], axis=0)
    return np.concatenate([motion, pad], axis=0).astype(np.float32)


def smooth_centered(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([xp[i : i + window].mean(axis=0) for i in range(x.shape[0])]).astype(np.float32)


def motion_metrics(motion: np.ndarray) -> dict:
    expr = motion[:, :50]
    head = motion[:, 50:53]
    jaw = motion[:, 53:56]
    expr_rel = expr - expr[:1]
    jaw_norm = np.linalg.norm(jaw, axis=1)
    expr_norm = np.linalg.norm(expr, axis=1)
    expr_change_norm = np.linalg.norm(expr_rel, axis=1)
    lower_face = expr_rel[:, 20:50] if expr_rel.shape[1] >= 50 else expr_rel

    # Same lightweight proxy family used by render_flame_mesh_sequence.py,
    # plus relative metrics that are more useful for ranking candidate basis motions.
    smile_axis = expr[:, :10].mean(axis=1) - expr[:, 10:20].mean(axis=1)
    smile = np.tanh(smile_axis)
    mouth_left = np.stack([-0.32 - 0.04 * smile, -0.10 + 0.05 * smile, 0.18 + 0.015 * np.abs(smile)], axis=1)
    mouth_right = np.stack([0.32 + 0.04 * smile, -0.10 + 0.05 * smile, 0.18 + 0.015 * np.abs(smile)], axis=1)
    corner_disp = 0.5 * (
        np.linalg.norm(mouth_left - mouth_left[:1], axis=1)
        + np.linalg.norm(mouth_right - mouth_right[:1], axis=1)
    )
    lip_distance = 0.08 + 0.10 * jaw_norm
    lip_distance_change = np.abs(lip_distance - lip_distance[0])

    return {
        "mouth_corner_displacement_max": float(corner_disp.max(initial=0.0)),
        "lower_face_deformation": float(np.std(lower_face)) if lower_face.size else 0.0,
        "lip_distance_change_max": float(lip_distance_change.max(initial=0.0)),
        "jaw_norm_max": float(jaw_norm.max(initial=0.0)),
        "jaw_norm_mean": float(jaw_norm.mean()) if len(jaw_norm) else 0.0,
        "expr_norm_max": float(expr_norm.max(initial=0.0)),
        "expr_norm_mean": float(expr_norm.mean()) if len(expr_norm) else 0.0,
        "expr_change_norm_max": float(expr_change_norm.max(initial=0.0)),
        "head_norm_max": float(np.linalg.norm(head, axis=1).max(initial=0.0)),
    }


def norm01(value: float, values: list[float]) -> float:
    hi = max(values) if values else 0.0
    lo = min(values) if values else 0.0
    if not math.isfinite(value) or hi <= lo:
        return 0.0
    return float((value - lo) / (hi - lo))


def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s[:80] or "candidate"


def classify_candidate(record: dict) -> tuple[str, list[str]]:
    parts = record["text_score_parts"]
    reasons = []
    if record["jaw_norm_max"] > 0.08:
        reasons.append("jaw>0.08")
    if record["lip_distance_change_max"] > 0.012:
        reasons.append("large_lip_distance_change")
    if parts["jaw_open_text_hits"] > 0:
        reasons.append("open_mouth_text")
    if parts["laugh_text_hits"] > 0:
        reasons.append("laugh_text")
    if parts["talk_text_hits"] > 0:
        reasons.append("talk_text")
    if parts["lip_tight_text_hits"] > 0:
        reasons.append("lip_tight_text")

    has_clean_text = (
        parts["clean_smile_hits"] > 0
        or re.search(r"\b(smile|smiling|happy|happiness|grin|cheerful|joyful)\b", record["text"].lower())
    )
    clean = (
        has_clean_text
        and record["jaw_norm_max"] <= 0.08
        and record["lip_distance_change_max"] <= 0.012
        and parts["jaw_open_text_hits"] == 0
        and parts["laugh_text_hits"] == 0
        and parts["talk_text_hits"] == 0
        and parts["lip_tight_text_hits"] == 0
    )
    if clean:
        return "clean_smile", ["low_jaw", "clean_smile_text"]
    if reasons:
        return "open_smile_or_laugh", reasons
    return "needs_visual_review", ["borderline_metrics"]


codebook_text = {}
for obj in read_jsonl(codebook_path):
    sid = obj.get("sample_id")
    if sid:
        codebook_text[sid] = obj.get("searchable_text", "")

records = []
for obj in read_jsonl(manifest_path):
    sid = obj.get("sample_id", "")
    npz_path = Path(obj.get("npz_path", ""))
    text = obj.get("searchable_text", "") or codebook_text.get(sid, "")
    if codebook_text.get(sid) and codebook_text[sid] not in text:
        text = text + "\n" + codebook_text[sid]
    low = f"{sid}\n{text}".lower()
    if not any(k in low for k in POSITIVE):
        continue
    if not npz_path.exists():
        continue
    try:
        motion = fit_len(load_motion(npz_path), 32)
        metrics = motion_metrics(motion)
    except Exception as exc:
        print(f"[WARN] skip {sid}: {exc}")
        continue
    t_score, t_parts = text_score(low)
    records.append({
        "sample_id": sid,
        "text": text,
        "npz_path": str(npz_path),
        "text_score_raw": t_score,
        "text_score_parts": t_parts,
        **metrics,
    })

if not records:
    raise SystemExit("no smile candidates found")

fields = [
    "mouth_corner_displacement_max", "lower_face_deformation",
    "lip_distance_change_max", "jaw_norm_max", "expr_norm_max",
    "expr_change_norm_max", "text_score_raw",
]
field_values = {f: [float(r[f]) for r in records] for f in fields}
for r in records:
    mouth = norm01(r["mouth_corner_displacement_max"], field_values["mouth_corner_displacement_max"])
    lower = norm01(r["lower_face_deformation"], field_values["lower_face_deformation"])
    lip = norm01(r["lip_distance_change_max"], field_values["lip_distance_change_max"])
    jaw = norm01(r["jaw_norm_max"], field_values["jaw_norm_max"])
    expr = norm01(r["expr_norm_max"], field_values["expr_norm_max"])
    expr_change = norm01(r["expr_change_norm_max"], field_values["expr_change_norm_max"])
    text = norm01(r["text_score_raw"], field_values["text_score_raw"])
    # Smile should move corners/lower face and be text-supported, while avoiding
    # very large jaw opening and lip-tightening annotations.
    r["score"] = (
        0.24 * mouth
        + 0.20 * lower
        + 0.12 * lip
        + 0.10 * expr
        + 0.09 * expr_change
        + 0.30 * text
        - 0.20 * jaw
    )
    if r["text_score_parts"]["lip_tight_text_hits"] > 0:
        r["score"] -= 0.22
    if r["text_score_parts"]["jaw_open_text_hits"] > 2:
        r["score"] -= 0.08
    category, category_reasons = classify_candidate(r)
    r["category"] = category
    r["category_reasons"] = category_reasons
    r["clean_score"] = (
        0.32 * mouth
        + 0.22 * lower
        + 0.10 * expr
        + 0.08 * expr_change
        + 0.36 * text
        - 0.55 * jaw
        - 0.18 * lip
    )
    r["clean_score"] -= 0.35 * r["text_score_parts"]["talk_text_hits"]
    r["clean_score"] -= 0.30 * r["text_score_parts"]["laugh_text_hits"]
    r["clean_score"] -= 0.25 * r["text_score_parts"]["jaw_open_text_hits"]
    r["clean_score"] -= 0.25 * r["text_score_parts"]["lip_tight_text_hits"]
    if category == "clean_smile":
        r["clean_score"] += 0.50
    r["score_parts"] = {
        "mouth_corner_norm": mouth,
        "lower_face_norm": lower,
        "lip_change_norm": lip,
        "jaw_norm_penalty": jaw,
        "expr_norm": expr,
        "expr_change_norm": expr_change,
        "text_norm": text,
        "clean_score": r["clean_score"],
    }

if rerank_mode == "clean":
    records.sort(key=lambda x: x["clean_score"], reverse=True)
    pool = records[:candidate_pool]
    clean_pool = [r for r in pool if r["category"] == "clean_smile"]
    fallback_pool = [r for r in pool if r["category"] != "clean_smile"]
    selected = (clean_pool + fallback_pool)[:topk]
else:
    records.sort(key=lambda x: x["score"], reverse=True)
    pool = records[:candidate_pool]
    selected = pool[:topk]

current_motion = fit_len(load_motion(current_smile_npz), 32)
current_metrics = motion_metrics(current_motion)

npz_dir.mkdir(parents=True, exist_ok=True)
out_cases = []
for rank, r in enumerate(selected, 1):
    src = fit_len(load_motion(Path(r["npz_path"])), 32)
    motion = np.zeros_like(src, dtype=np.float32)
    # Isolate smile primitive candidates from dataset head motion. Preserve jaw
    # so the render can reveal open-mouth leakage and the report can reject it.
    motion[:, :50] = smooth_centered(src[:, :50], 5)
    motion[:, 53:56] = smooth_centered(src[:, 53:56], 5)
    prefix = "clean_smile_candidate" if rerank_mode == "clean" else "smile_candidate"
    case_name = f"{prefix}_{rank:02d}_{slug(r['sample_id'])}"
    out_path = npz_dir / f"{case_name}.npz"
    np.savez(
        out_path,
        motion=motion,
        motion_raw=motion,
        motion_norm=motion,
        expr_delta=motion[:, :50],
        head_delta=motion[:, 50:53],
        jaw_delta=motion[:, 53:56],
    )
    r = dict(r)
    r.update({
        "rank": rank,
        "rerank_mode": rerank_mode,
        "case_name": case_name,
        "candidate_npz": str(out_path),
        "video": f"fastavatar_video/{case_name}.mp4",
        "visual_clean_smile": "TODO inspect smile_candidate_grid.png",
    })
    out_cases.append(r)

payload = {
    "source_manifest": str(manifest_path),
    "source_codebook": str(codebook_path),
    "current_smile_npz": str(current_smile_npz),
    "current_smile_metrics": current_metrics,
    "candidate_pool_count": len(pool),
    "positive_candidate_count": len(records),
    "rerank_mode": rerank_mode,
    "selected": out_cases,
}
candidates_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
for r in out_cases:
    print(f"{r['case_name']}|{r['candidate_npz']}")
PY

if [[ "${RENDER}" == "1" ]]; then
  printf 'case\tstatus\tvideo\tlog\n' > "${RENDER_STATUS}"
  mapfile -t CASES < <(python - "${CANDIDATES_JSON}" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for case in data["selected"]:
    print(f"{case['case_name']}|{case['candidate_npz']}")
PY
)

  for item in "${CASES[@]}"; do
    IFS='|' read -r case_name candidate_npz <<< "${item}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${VIDEO_DIR}/${case_name}.mp4" ]]; then
      printf '[SmileCalibration] skipping existing video %s\n' "${case_name}"
      printf '%s\tskipped_existing\t%s\t%s\n' \
        "${case_name}" "${VIDEO_DIR}/${case_name}.mp4" "${OUTPUT_DIR}/${case_name}/render_wrapper.log" >> "${RENDER_STATUS}"
      continue
    fi
    printf '[SmileCalibration] rendering %s\n' "${case_name}"
    mkdir -p "${OUTPUT_DIR}/${case_name}"
    if MAX_SINGLE_FRAME_RENDER="${MAX_SINGLE_FRAME_RENDER}" bash "${RENDER_WRAPPER}" \
      --motion_npz "${candidate_npz}" \
      --output_dir "${OUTPUT_DIR}" \
      --case_name "${case_name}" \
      --sequence_name "${SEQUENCE_NAME}" \
      --neutral_template "${NEUTRAL_TEMPLATE}" \
      --pack_root "${OUTPUT_DIR}/${case_name}/fastavatar_pack" \
      --head_target neck_pose \
      --inference_n_frames "${INFERENCE_N_FRAMES}" \
      --cuda_visible_devices "${CUDA_VISIBLE_DEVICES_VALUE}" > "${OUTPUT_DIR}/${case_name}/render_wrapper.log" 2>&1; then
      printf '%s\tok\t%s\t%s\n' \
        "${case_name}" "${VIDEO_DIR}/${case_name}.mp4" "${OUTPUT_DIR}/${case_name}/render_wrapper.log" >> "${RENDER_STATUS}"
    else
      printf '[WARN] render failed for %s; see %s\n' "${case_name}" "${OUTPUT_DIR}/${case_name}/render_wrapper.log" >&2
      tail -n 80 "${OUTPUT_DIR}/${case_name}/render_wrapper.log" >&2 || true
      printf '%s\tfailed\t%s\t%s\n' \
        "${case_name}" "${VIDEO_DIR}/${case_name}.mp4" "${OUTPUT_DIR}/${case_name}/render_wrapper.log" >> "${RENDER_STATUS}"
      if [[ "${CONTINUE_ON_RENDER_FAIL}" != "1" ]]; then
        exit 1
      fi
    fi
  done

  for item in "${CASES[@]}"; do
    IFS='|' read -r case_name _candidate_npz <<< "${item}"
    video="${VIDEO_DIR}/${case_name}.mp4"
    if [[ ! -f "${video}" ]]; then
      printf '[WARN] expected video missing, skip frames: %s\n' "${video}" >&2
      continue
    fi
    frame_count="$(ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "${video}" || true)"
    if [[ -z "${frame_count}" || "${frame_count}" == "N/A" ]]; then
      frame_count="${INFERENCE_N_FRAMES}"
    fi
    mid_frame=$(( frame_count / 2 ))
    last_frame=$(( frame_count > 0 ? frame_count - 1 : 0 ))
    ffmpeg -y -hide_banner -loglevel error -i "${video}" \
      -vf "select='eq(n,0)+eq(n,${mid_frame})+eq(n,${last_frame})'" -vsync 0 \
      "${FRAME_DIR}/${case_name}_%02d.png"
  done

  montage_inputs=()
  for item in "${CASES[@]}"; do
    IFS='|' read -r case_name _candidate_npz <<< "${item}"
    for i in 01 02 03; do
      frame="${FRAME_DIR}/${case_name}_${i}.png"
      [[ -f "${frame}" ]] && montage_inputs+=("${frame}")
    done
  done
  if (( ${#montage_inputs[@]} > 0 )); then
    montage "${montage_inputs[@]}" -tile 3x -geometry 224x224+8+8 "${GRID}"
  fi
else
  printf '[SmileCalibration] RENDER=0, skipping FastAvatar render and grid generation\n'
  printf 'case\tstatus\tvideo\tlog\n' > "${RENDER_STATUS}"
fi

python - "${CANDIDATES_JSON}" "${REPORT}" "${GRID}" "${RENDER_STATUS}" <<'PY'
import json
import sys
from pathlib import Path

data = json.load(open(sys.argv[1], encoding="utf-8"))
report = Path(sys.argv[2])
grid = Path(sys.argv[3])
render_status_path = Path(sys.argv[4])

def cell(text: str, n: int = 180) -> str:
    text = " ".join(str(text).replace("|", "/").split())
    return text[: n - 3] + "..." if len(text) > n else text

selected = data["selected"]
best = selected[0]
cur = data["current_smile_metrics"]
render_status = {}
if render_status_path.exists():
    for line in render_status_path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 4:
            render_status[parts[0]] = {"status": parts[1], "video": parts[2], "log": parts[3]}

for item in selected:
    status = render_status.get(item["case_name"], {})
    item["render_status"] = status.get("status", "not_run")
    item["render_log"] = status.get("log", f"{report.parent}/{item['case_name']}/render_wrapper.log")
    if Path(report.parent, item["video"]).exists():
        item["render_status"] = "ok" if item["render_status"] == "not_run" else item["render_status"]

clean_candidates = [x for x in selected if x.get("category") == "clean_smile"]
open_candidates = [x for x in selected if x.get("category") == "open_smile_or_laugh"]
review_candidates = [x for x in selected if x.get("category") == "needs_visual_review"]
best_clean = clean_candidates[0] if clean_candidates else None
best_rendered_clean = next((x for x in clean_candidates if x["render_status"] in {"ok", "skipped_existing"}), None)

lines = [
    "# P7/P8 Smile Primitive Calibration",
    "",
    f"Output directory: `{report.parent}`",
    f"Candidate grid: `{grid.name}`" if grid.exists() else "Candidate grid: not generated yet (`RENDER=0` or render incomplete)",
    "",
    "## Inputs",
    "",
    f"- manifest: `{data['source_manifest']}`",
    f"- codebook: `{data['source_codebook']}`",
    f"- current smile npz: `{data['current_smile_npz']}`",
    f"- positive keyword candidates scored: `{data['positive_candidate_count']}`",
    f"- candidate pool retained before top-k: `{data['candidate_pool_count']}`",
    f"- rerank mode: `{data.get('rerank_mode', 'balanced')}`",
    f"- render status: `{render_status_path}`",
    "",
    "## Scoring",
    "",
    "Candidates are filtered by smile/happy/laugh/grin/cheerful keywords, then scored with mouth-corner displacement, lower-face deformation, lip-distance change, jaw-opening penalty, expression magnitude/change, and explicit smile/happy/laugh text support. The score intentionally does not sort by expr norm alone.",
    "",
    "Clean-smile classification requires smile/happy/grin/cheerful text support, jaw max <= 0.08, small lip-distance change, and no talk/laugh/open-mouth/lip-tightening text. Open-smile/laugh candidates are kept as controls, not default primitive recommendations.",
    "",
    "## Current Smile Primitive Diagnosis",
    "",
    f"- current expr_norm_max: `{cur['expr_norm_max']:.6f}`",
    f"- current expr_change_norm_max: `{cur['expr_change_norm_max']:.6f}`",
    f"- current mouth_corner_displacement_max proxy: `{cur['mouth_corner_displacement_max']:.6f}`",
    f"- current lip_distance_change_max proxy: `{cur['lip_distance_change_max']:.6f}`",
    f"- current jaw_norm_max: `{cur['jaw_norm_max']:.6f}`",
    "- observed FastAvatar issue: prior probe showed `strong_smile_x3` changes the mouth but reads closer to lip tightening than a clean smile.",
    "- likely cause: the existing smile basis has high expression energy but the active expression direction does not align strongly enough with clean cheek/lip-corner pull; simply increasing `smile_gain` amplifies the wrong basis direction.",
    "",
    "## Top Candidates",
    "",
    "| rank | class | render | sample_id | score | clean score | source motion | candidate npz | mouth corner | lower face | lip change | jaw max | expr max | video | log | text |",
    "|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|---|",
]

for r in selected:
    lines.append(
        f"| {r['rank']} | `{r.get('category', 'unknown')}` | `{r['render_status']}` | "
        f"`{r['sample_id']}` | {r['score']:.4f} | {r.get('clean_score', 0.0):.4f} | "
        f"`{r['npz_path']}` | `{r['candidate_npz']}` | "
        f"{r['mouth_corner_displacement_max']:.4f} | {r['lower_face_deformation']:.4f} | "
        f"{r['lip_distance_change_max']:.4f} | {r['jaw_norm_max']:.4f} | "
        f"{r['expr_norm_max']:.4f} | [video]({r['video']}) | "
        f"`{r['render_log']}` | {cell(r['text'])} |"
    )

lines += [
    "",
    "## Clean Smile Candidates",
    "",
    "| rank | sample_id | jaw max | lip change | mouth corner | render | video | reason |",
    "|---:|---|---:|---:|---:|---|---|---|",
]
if clean_candidates:
    for r in clean_candidates:
        lines.append(
            f"| {r['rank']} | `{r['sample_id']}` | {r['jaw_norm_max']:.4f} | "
            f"{r['lip_distance_change_max']:.4f} | {r['mouth_corner_displacement_max']:.4f} | "
            f"`{r['render_status']}` | [video]({r['video']}) | {cell(', '.join(r.get('category_reasons', [])), 80)} |"
        )
else:
    lines.append("| - | none in this top-k | - | - | - | - | - | balanced ranking is dominated by open-mouth/laugh/talk candidates |")

lines += [
    "",
    "## Open Smile Or Laugh Controls",
    "",
    "| rank | sample_id | jaw max | lip change | mouth corner | render | video | why not default smile |",
    "|---:|---|---:|---:|---:|---|---|---|",
]
if open_candidates:
    for r in open_candidates:
        lines.append(
            f"| {r['rank']} | `{r['sample_id']}` | {r['jaw_norm_max']:.4f} | "
            f"{r['lip_distance_change_max']:.4f} | {r['mouth_corner_displacement_max']:.4f} | "
            f"`{r['render_status']}` | [video]({r['video']}) | {cell(', '.join(r.get('category_reasons', [])), 100)} |"
        )
else:
    lines.append("| - | none | - | - | - | - | - | - |")

if review_candidates:
    lines += [
        "",
        "## Borderline Candidates",
        "",
        "| rank | sample_id | jaw max | lip change | render | video |",
        "|---:|---|---:|---:|---|---|",
    ]
    for r in review_candidates:
        lines.append(
            f"| {r['rank']} | `{r['sample_id']}` | {r['jaw_norm_max']:.4f} | "
            f"{r['lip_distance_change_max']:.4f} | `{r['render_status']}` | [video]({r['video']}) |"
        )

lines += [
    "",
    "## Answers",
    "",
    f"- best candidate by composite score: `{best['sample_id']}` (`{best.get('category', 'unknown')}`), npz `{best['candidate_npz']}`.",
    f"- best clean-smile candidate by rules: `{best_clean['sample_id']}` with npz `{best_clean['candidate_npz']}`." if best_clean else "- best clean-smile candidate by rules: none in this top-k; run `RERANK_MODE=clean`.",
    f"- best rendered clean-smile candidate: `{best_rendered_clean['sample_id']}`." if best_rendered_clean else "- best rendered clean-smile candidate: none yet.",
    "- candidates marked `open_smile_or_laugh` are useful controls but should not be used as the default smile primitive.",
    "- current smile primitive likely looks like lip tightening because its expression direction has high energy but weak clean lip-corner/cheek pull; increasing gain amplifies that direction instead of changing the basis.",
    "- recommendation: consider replacing the P7/P8 smile basis only after visual inspection confirms a clean-smile candidate in the grid/video.",
    "- after replacement, sweep `smile_gain` over `1.0`, `1.5`, and `2.0`.",
    "",
    "## Recommendations",
    "",
    "- Do not update `primitive_presets` yet based only on gain. The current failure mode is basis direction, not just amplitude.",
    "- Recommended next action: visually inspect top rendered candidates and select a clean-smile basis candidate before replacing the existing smile primitive.",
    "- Suggested `smile_gain` default: keep `1.0` until a better basis is selected; after replacement, sweep `1.0`, `1.5`, and `2.0` on the chosen basis.",
    "- If none of the top rendered candidates reads as a clean smile, broaden candidate mining to include original MMHead pkl sequences and/or build a curated smile expression basis from multiple candidates.",
]

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[SmileCalibration] report={report}")
PY

printf '[SmileCalibration] candidates=%s\n' "${CANDIDATES_JSON}"
printf '[SmileCalibration] report=%s\n' "${REPORT}"
if [[ -f "${GRID}" ]]; then
  printf '[SmileCalibration] grid=%s\n' "${GRID}"
fi
