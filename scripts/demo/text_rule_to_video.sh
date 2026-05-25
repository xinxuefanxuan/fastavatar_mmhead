#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/demo/text_rule_to_video.sh \"<prompt>\" <run_name>"
  exit 1
fi

PROMPT="$1"
RUN_NAME="$2"

PROTOTYPE_PATH="outputs/mmhead_debug/primitive_prototypes_v1/prototypes.pt"
VAE_CHECKPOINT="outputs/mmhead_debug/vae_debug_beta1e4/best.pt"
NORM_STATS="outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json"
PRESET_CONFIG="motion_model/primitive_presets.json"
NEUTRAL_TEMPLATE="assets/sample_motion/nersemble_seq_214_neutral"
IMAGE_INPUT="assets/sample_input/mono_video/nersemble_seq_214.mp4"
INFER_CONFIG="configs/inference/infer.yaml"
MODEL_DIR="model_zoo/fastavatar/"
OUTPUT_ROOT="outputs/mmhead_debug/text_demo_runs"

RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
NPZ_DIR="${RUN_ROOT}/generated_npz"
PACK_ROOT="${RUN_ROOT}/pack/${RUN_NAME}_pack"
LOG_DIR="${RUN_ROOT}/logs"
NPZ_PATH="${NPZ_DIR}/${RUN_NAME}.npz"
MOTION_DIR="${PACK_ROOT}/${RUN_NAME}"
VIDEO_OUT="${RUN_ROOT}/video.mp4"

mkdir -p "${NPZ_DIR}" "${LOG_DIR}"

echo "[Demo] prompt=${PROMPT}"

echo "[Demo] Step1 generate_from_text_rule.py"
python motion_model/generate_from_text_rule.py \
  --prompt "${PROMPT}" \
  --prototype_path "${PROTOTYPE_PATH}" \
  --vae_checkpoint "${VAE_CHECKPOINT}" \
  --norm_stats "${NORM_STATS}" \
  --preset_config "${PRESET_CONFIG}" \
  --output_npz "${NPZ_PATH}" \
  --target_len 64 \
  --output_len 32 \
  --temporal_mode hold \
  --ramp_frames 10 \
  --hold_frames 18 \
  --release_frames 4 \
  --release_ratio 0.75 \
  --device cuda \
  2>&1 | tee "${LOG_DIR}/01_generate.log"

echo "[Demo] Step2 render_motion_npz.py --fastavatar_pack"
python text_motion/render_motion_npz.py \
  --motion_npz "${NPZ_PATH}" \
  --neutral_template "${NEUTRAL_TEMPLATE}" \
  --fastavatar_pack \
  --pack_root "${PACK_ROOT}" \
  --sequence_name "${RUN_NAME}" \
  --motion_key motion \
  --head_target neck_pose \
  --overwrite \
  2>&1 | tee "${LOG_DIR}/02_pack.log"

echo "[Demo] Step3 scripts/infer/infer.sh"
bash scripts/infer/infer.sh \
  "${INFER_CONFIG}" \
  "${MODEL_DIR}" \
  "${IMAGE_INPUT}" \
  "${MOTION_DIR}" \
  32 \
  8 \
  Monocular \
  false \
  2>&1 | tee "${LOG_DIR}/03_infer.log"

LATEST_MP4=$(find infer_results/videos -type f -name "*.mp4" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -z "${LATEST_MP4}" || ! -f "${LATEST_MP4}" ]]; then
  echo "[ERROR] Inference finished but no output mp4 found under infer_results/videos"
  exit 1
fi

cp -f "${LATEST_MP4}" "${VIDEO_OUT}"

echo "[Demo] npz=${NPZ_PATH}"
echo "[Demo] motion_dir=${MOTION_DIR}"
echo "[Demo] video=${VIDEO_OUT}"
