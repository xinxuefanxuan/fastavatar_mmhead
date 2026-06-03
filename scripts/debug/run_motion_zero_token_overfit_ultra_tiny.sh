#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

CONFIG_PATH="${CONFIG_PATH:-configs/train/fastavatar_motion_zero_token_overfit_ultra_tiny.yaml}" \
  bash "${REPO_ROOT}/scripts/debug/run_motion_zero_token_overfit.sh" "$@"
