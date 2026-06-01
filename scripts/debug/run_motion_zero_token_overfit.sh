#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/train/fastavatar_motion_zero_token_overfit.yaml}"

export FASTAVATAR_TOKEN_DEBUG="${FASTAVATAR_TOKEN_DEBUG:-1}"

echo "[P9.2] Running motion-zero + GT motion-token overfit"
echo "[P9.2] Config: ${CONFIG_PATH}"
echo "[P9.2] FASTAVATAR_TOKEN_DEBUG=${FASTAVATAR_TOKEN_DEBUG}"

python FastAvatar/launch.py train.fastavatar --config "${CONFIG_PATH}" "$@"
