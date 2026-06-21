#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/diagnostics/render_fastavatar_case.sh \
    --motion_npz <path> \
    --output_dir <dir> \
    --case_name <name> \
    [--sequence_name <name>] \
    [--neutral_template <dir>] \
    [--pack_root <dir>] \
    [--inference_n_frames <n>] \
    [--cuda_visible_devices <ids>]

Environment variables with matching uppercase names are also supported.
EOF
}

MOTION_NPZ="${MOTION_NPZ:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmhead_debug/p7p8_quality_diagnosis}"
CASE_NAME="${CASE_NAME:-}"
SEQUENCE_NAME="${SEQUENCE_NAME:-nersemble_seq_214}"
NEUTRAL_TEMPLATE="${NEUTRAL_TEMPLATE:-assets/sample_motion/nersemble_seq_214_neutral}"
PACK_ROOT="${PACK_ROOT:-}"
INFERENCE_N_FRAMES="${INFERENCE_N_FRAMES:-32}"
MAX_SINGLE_FRAME_RENDER="${MAX_SINGLE_FRAME_RENDER:-8}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-7}"
INFER_CONFIG="${INFER_CONFIG:-configs/inference/infer.yaml}"
MODEL_ROOT="${MODEL_ROOT:-model_zoo/fastavatar/}"
IMAGE_INPUT="${IMAGE_INPUT:-assets/sample_input/mono_video/nersemble_seq_214.mp4}"
MODE="${MODE:-Monocular}"
ENABLE_CAMERA_ROTATION="${ENABLE_CAMERA_ROTATION:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion_npz)
      MOTION_NPZ="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --case_name)
      CASE_NAME="$2"
      shift 2
      ;;
    --sequence_name)
      SEQUENCE_NAME="$2"
      shift 2
      ;;
    --neutral_template)
      NEUTRAL_TEMPLATE="$2"
      shift 2
      ;;
    --pack_root)
      PACK_ROOT="$2"
      shift 2
      ;;
    --inference_n_frames)
      INFERENCE_N_FRAMES="$2"
      shift 2
      ;;
    --cuda_visible_devices)
      CUDA_VISIBLE_DEVICES_VALUE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf '[ERROR] Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MOTION_NPZ}" || -z "${CASE_NAME}" ]]; then
  printf '[ERROR] --motion_npz and --case_name are required\n' >&2
  usage >&2
  exit 2
fi
if [[ ! -f "${MOTION_NPZ}" ]]; then
  printf '[ERROR] motion_npz not found: %s\n' "${MOTION_NPZ}" >&2
  exit 1
fi
if [[ ! -d "${NEUTRAL_TEMPLATE}" ]]; then
  printf '[ERROR] neutral_template not found: %s\n' "${NEUTRAL_TEMPLATE}" >&2
  exit 1
fi

CASE_RENDER_DIR="${OUTPUT_DIR}/${CASE_NAME}/fastavatar_render"
PACK_ROOT="${PACK_ROOT:-${CASE_RENDER_DIR}/pack}"
MOTION_DIR="${PACK_ROOT}/${SEQUENCE_NAME}"
VIDEO_DIR="${OUTPUT_DIR}/fastavatar_video"
VIDEO_OUT="${VIDEO_DIR}/${CASE_NAME}.mp4"
PACK_LOG="${CASE_RENDER_DIR}/pack.log"
INFER_LOG="${CASE_RENDER_DIR}/infer.log"
INFER_VIDEO_DUMP="${CASE_RENDER_DIR}/infer_videos"
INFER_IMAGE_DUMP="${CASE_RENDER_DIR}/infer_images"
INFER_TMP_DUMP="${CASE_RENDER_DIR}/infer_tmp"

mkdir -p "${CASE_RENDER_DIR}" "${VIDEO_DIR}" "${INFER_VIDEO_DUMP}" "${INFER_IMAGE_DUMP}" "${INFER_TMP_DUMP}"

printf '[FastAvatarCase] cwd=%s\n' "$(pwd)"
printf '[FastAvatarCase] motion_npz=%s\n' "${MOTION_NPZ}"
printf '[FastAvatarCase] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[FastAvatarCase] case_name=%s\n' "${CASE_NAME}"
printf '[FastAvatarCase] sequence_name=%s\n' "${SEQUENCE_NAME}"
printf '[FastAvatarCase] neutral_template=%s\n' "${NEUTRAL_TEMPLATE}"
printf '[FastAvatarCase] pack_root=%s\n' "${PACK_ROOT}"
printf '[FastAvatarCase] inference_n_frames=%s\n' "${INFERENCE_N_FRAMES}"
printf '[FastAvatarCase] cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES_VALUE}"

python text_motion/render_motion_npz.py \
  --motion_npz "${MOTION_NPZ}" \
  --neutral_template "${NEUTRAL_TEMPLATE}" \
  --fastavatar_pack \
  --pack_root "${PACK_ROOT}" \
  --sequence_name "${SEQUENCE_NAME}" \
  --motion_key motion \
  --head_target neck_pose \
  --overwrite > "${PACK_LOG}" 2>&1

: > "${INFER_LOG}"
run_infer() {
  local frames="$1"
  printf '\n[Infer] INFERENCE_N_FRAMES=%s\n' "${frames}" >> "${INFER_LOG}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  VIDEO_DUMP="${INFER_VIDEO_DUMP}" \
  IMAGE_DUMP="${INFER_IMAGE_DUMP}" \
  SAVE_TMP_DUMP="${INFER_TMP_DUMP}" \
  bash scripts/infer/infer.sh \
    "${INFER_CONFIG}" \
    "${MODEL_ROOT}" \
    "${IMAGE_INPUT}" \
    "${MOTION_DIR}" \
    "${frames}" \
    "${MAX_SINGLE_FRAME_RENDER}" \
    "${MODE}" \
    "${ENABLE_CAMERA_ROTATION}" >> "${INFER_LOG}" 2>&1
}

if ! run_infer "${INFERENCE_N_FRAMES}"; then
  if (( INFERENCE_N_FRAMES > 16 )) && grep -Eiq 'out of memory|CUDA.*memory|CUBLAS.*alloc|CUDNN.*alloc' "${INFER_LOG}"; then
    printf '[WARN] infer failed with memory-like error; retrying with INFERENCE_N_FRAMES=16\n' >&2
    printf '\n[Retry] INFERENCE_N_FRAMES=16 after memory-like failure\n' >> "${INFER_LOG}"
    if ! run_infer 16; then
      printf '[ERROR] FastAvatar infer failed for case=%s after retry; last 120 lines of %s:\n' "${CASE_NAME}" "${INFER_LOG}" >&2
      tail -n 120 "${INFER_LOG}" >&2 || true
      exit 1
    fi
  else
    printf '[ERROR] FastAvatar infer failed for case=%s; last 120 lines of %s:\n' "${CASE_NAME}" "${INFER_LOG}" >&2
    tail -n 120 "${INFER_LOG}" >&2 || true
    exit 1
  fi
fi

LATEST_MP4="$(find "${INFER_VIDEO_DUMP}" -type f -name "*.mp4" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "${LATEST_MP4}" || ! -f "${LATEST_MP4}" ]]; then
  printf '[ERROR] Inference finished but no output mp4 found under %s\n' "${INFER_VIDEO_DUMP}" >&2
  tail -n 120 "${INFER_LOG}" >&2 || true
  exit 1
fi

cp -f "${LATEST_MP4}" "${VIDEO_OUT}"

printf '[FastAvatarCase] pack_log=%s\n' "${PACK_LOG}"
printf '[FastAvatarCase] infer_log=%s\n' "${INFER_LOG}"
printf '[FastAvatarCase] source_video=%s\n' "${LATEST_MP4}"
printf '[FastAvatarCase] video=%s\n' "${VIDEO_OUT}"
