#!/usr/bin/env bash
set -euo pipefail

MOTION_NPZ="${1:-}"
OUTPUT_DIR="${2:-outputs/mmhead_debug/p7p8_mask_sensitivity}"
FASTAVATAR_RENDER_CMD_TEMPLATE="${FASTAVATAR_RENDER_CMD_TEMPLATE:-}"

mkdir -p "${OUTPUT_DIR}"
REPORT="${OUTPUT_DIR}/mask_sensitivity_report.md"

cat > "${REPORT}" <<EOF
# Mask Sensitivity Render Ablation

Motion: \\`${MOTION_NPZ:-not provided}\\`

This script is intentionally an optional ablation entrypoint. It does **not** modify FastAvatar default inference behavior.

## Current mask audit

Known mask-related files/functions in this repository:

- \\`data_preprocessing/video_track.py\\`: loads binary masks, composites RGB with background, and saves processed \\`mask.npy\\`.
- \\`FastAvatar/losses/pixelwise.py\\`: applies masks to training losses.
- \\`FastAvatar/models/rendering/utils/renderer.py\\`: applies in-box ray validity masking during rendering.
- \\`configs/stylematte_config.json\\`: matting/mask model configuration.

If paper-like motion or right-turn artifacts disappear with softer/dilated masks, the minimal future change point is a render-time/inference mask preprocessing hook, not a default removal of masks.

## Ablation variants

| variant | status | output |
|---|---|---|
EOF

variants=(baseline soft_dilated weaker_boundary)
for variant in "${variants[@]}"; do
  variant_dir="${OUTPUT_DIR}/${variant}"
  mkdir -p "${variant_dir}"
  if [[ -n "${FASTAVATAR_RENDER_CMD_TEMPLATE}" && -n "${MOTION_NPZ}" ]]; then
    cmd="${FASTAVATAR_RENDER_CMD_TEMPLATE//\{motion\}/${MOTION_NPZ}}"
    cmd="${cmd//\{out_dir\}/${variant_dir}}"
    cmd="${cmd//\{mask_variant\}/${variant}}"
    if bash -lc "${cmd}" > "${variant_dir}/render.log" 2>&1; then
      echo "| ${variant} | rendered | \\`${variant_dir}\\` |" >> "${REPORT}"
    else
      echo "| ${variant} | render_failed (see render.log) | \\`${variant_dir}\\` |" >> "${REPORT}"
    fi
  else
    echo "| ${variant} | audit_only: set FASTAVATAR_RENDER_CMD_TEMPLATE with {motion}, {out_dir}, {mask_variant} to render | \\`${variant_dir}\\` |" >> "${REPORT}"
  fi
done

cat >> "${REPORT}" <<'EOF'

## Interpretation guide

- Baseline artifact only, soft/dilated OK: strong boundary/mask likely clips visible head volume.
- All variants artifact with normal FLAME mesh: renderer/source-view visibility range likely dominates.
- FLAME mesh already weak or paper-like: motion amplitude/prototype issue comes before rendering.
EOF

echo "${REPORT}"
