#!/usr/bin/env bash
set -euo pipefail

ABC_CONFIG_DIR="${P9_ABC_CONFIG_DIR:-outputs/mmhead_debug/p9_2_abc_configs}"
ABC_LOG_DIR="${P9_ABC_LOG_DIR:-outputs/mmhead_debug/logs/p9_2_abc}"
ABC_BASE_CONFIG="${P9_ABC_BASE_CONFIG:-configs/train/fastavatar_motion_zero_token_overfit_micro_render.yaml}"

mkdir -p "${ABC_CONFIG_DIR}" "${ABC_LOG_DIR}"

python scripts/debug/create_p9_2_abc_configs.py \
  --base_config "${ABC_BASE_CONFIG}" \
  --output_dir "${ABC_CONFIG_DIR}"

for name in A_normal_no_token B_zero_no_token C_zero_gt_token; do
  config="${ABC_CONFIG_DIR}/${name}.yaml"
  log="${ABC_LOG_DIR}/${name}.log"
  echo "[P9.2ABC] running ${name} config=${config} log=${log}"
  CONFIG_PATH="${config}" bash scripts/debug/run_motion_zero_token_overfit.sh "$@" 2>&1 | tee "${log}"
done
