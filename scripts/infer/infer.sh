#!/usr/bin/env bash

# Core parameters
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/inference/infer.yaml}"
MODEL_NAME="${MODEL_NAME:-model_zoo/fastavatar/}"
IMAGE_INPUT="${IMAGE_INPUT:-assets/sample_input/mono_video/nersemble_seq_214.mp4}"
SEQUENCE_NAME="${SEQUENCE_NAME:-nersemble_seq_214}"
MOTION_SEQS_ROOT="${MOTION_SEQS_ROOT:-assets/sample_motion}"
MOTION_SEQS_DIR="${MOTION_SEQS_DIR:-${MOTION_SEQS_ROOT}/${SEQUENCE_NAME}/}"
INFERENCE_N_FRAMES="${INFERENCE_N_FRAMES:-16}"
MAX_SINGLE_FRAME_RENDER="${MAX_SINGLE_FRAME_RENDER:-16}"
MODE="${MODE:-Monocular}"  # Options: "Monocular", "MultiView"
ENABLE_CAMERA_ROTATION="${ENABLE_CAMERA_ROTATION:-false}"  # Set to true to enable camera rotation
RENDER_FPS="${RENDER_FPS:-20}"
MOTION_VIDEO_READ_FPS="${MOTION_VIDEO_READ_FPS:-7.5}"
EXPORT_VIDEO="${EXPORT_VIDEO:-true}"
EXPORT_MESH="${EXPORT_MESH:-true}"
FASTAVATAR_CUDA_VISIBLE_DEVICES="${FASTAVATAR_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
SAVE_TMP_DUMP="${SAVE_TMP_DUMP:-}"
IMAGE_DUMP="${IMAGE_DUMP:-}"
VIDEO_DUMP="${VIDEO_DUMP:-}"

# Allow command line overrides
TRAIN_CONFIG=${1:-$TRAIN_CONFIG}
MODEL_NAME=${2:-$MODEL_NAME}
IMAGE_INPUT=${3:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${4:-$MOTION_SEQS_DIR}
INFERENCE_N_FRAMES=${5:-$INFERENCE_N_FRAMES}
MAX_SINGLE_FRAME_RENDER=${6:-$MAX_SINGLE_FRAME_RENDER}
MODE=${7:-$MODE}
ENABLE_CAMERA_ROTATION=${8:-$ENABLE_CAMERA_ROTATION}

echo "TRAIN_CONFIG: $TRAIN_CONFIG"
echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "MODEL_NAME: $MODEL_NAME"
echo "SEQUENCE_NAME: $SEQUENCE_NAME"
echo "MOTION_SEQS_ROOT: $MOTION_SEQS_ROOT"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"
echo "INFERENCE_N_FRAMES: $INFERENCE_N_FRAMES"
echo "MAX_SINGLE_FRAME_RENDER: $MAX_SINGLE_FRAME_RENDER"
echo "MODE: $MODE"
echo "ENABLE_CAMERA_ROTATION: $ENABLE_CAMERA_ROTATION"
echo "CUDA_VISIBLE_DEVICES: $FASTAVATAR_CUDA_VISIBLE_DEVICES"
echo "VIDEO_DUMP: ${VIDEO_DUMP:-<default>}"

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

extra_args=()
if [[ -n "$SAVE_TMP_DUMP" ]]; then
    extra_args+=("save_tmp_dump=$SAVE_TMP_DUMP")
fi
if [[ -n "$IMAGE_DUMP" ]]; then
    extra_args+=("image_dump=$IMAGE_DUMP")
fi
if [[ -n "$VIDEO_DUMP" ]]; then
    extra_args+=("video_dump=$VIDEO_DUMP")
fi

# Run inference
CUDA_VISIBLE_DEVICES="$FASTAVATAR_CUDA_VISIBLE_DEVICES" python -m FastAvatar.launch infer.fastavatar \
    --config "$TRAIN_CONFIG" \
    "model_name=$MODEL_NAME" \
    "image_input=$IMAGE_INPUT" \
    "export_video=$EXPORT_VIDEO" \
    "export_mesh=$EXPORT_MESH" \
    "motion_seqs_dir=$MOTION_SEQS_DIR" \
    "render_fps=$RENDER_FPS" \
    "motion_video_read_fps=$MOTION_VIDEO_READ_FPS" \
    "inference_N_frames=$INFERENCE_N_FRAMES" \
    "max_single_frame_render=$MAX_SINGLE_FRAME_RENDER" \
    "mode=$MODE" \
    "enable_camera_rotation=$ENABLE_CAMERA_ROTATION" \
    "${extra_args[@]}"
