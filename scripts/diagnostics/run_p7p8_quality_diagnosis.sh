#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmhead_debug/p7p8_quality_diagnosis}"
GENERATOR="${GENERATOR:-motion_model/generate_from_text_temporal.py}"
SAFETY_CONFIG="${SAFETY_CONFIG:-configs/p7p8_motion_safety.yaml}"
FASTAVATAR_RENDER_CMD_TEMPLATE="${FASTAVATAR_RENDER_CMD_TEMPLATE:-}"
mkdir -p "${OUTPUT_DIR}"

prompts=("turn right" "turn left" "smile" "open mouth" "turn right and smile" "turn left and smile")

summary="${OUTPUT_DIR}/summary_report.md"
cat > "${summary}" <<EOF
# P7/P8 Motion Quality Diagnosis Summary

Output directory: \\`${OUTPUT_DIR}\\`

| case | generation | motion stats | FLAME/proxy preview | FastAvatar |
|---|---|---|---|---|
EOF

for prompt in "${prompts[@]}"; do
  slug="${prompt// /_}"
  case_dir="${OUTPUT_DIR}/${slug}"
  mkdir -p "${case_dir}"
  pre_npz="${case_dir}/motion_pre_safety.npz"
  post_npz="${case_dir}/motion_post_safety.npz"
  plan_json="${case_dir}/plan.json"

  if python "${GENERATOR}" \
      --prompt "${prompt}" \
      --output_npz "${post_npz}" \
      --output_plan_json "${plan_json}" \
      --motion_safety_config "${SAFETY_CONFIG}" \
      --apply_motion_safety \
      --save_pre_safety_motion "${pre_npz}" \
      --save_safety_report "${case_dir}/safety_report.json" \
      ${GENERATOR_ARGS:-} > "${case_dir}/generate.log" 2>&1; then
    gen_status="ok"
  else
    gen_status="failed"
  fi

  if [[ -f "${post_npz}" ]]; then
    python scripts/diagnostics/inspect_motion_npz.py --input "${pre_npz}" "${post_npz}" --output_dir "${case_dir}/motion_stats" > "${case_dir}/inspect.log" 2>&1 || true
    python scripts/diagnostics/render_flame_mesh_sequence.py --input "${post_npz}" --output_dir "${case_dir}/flame_preview" --wireframe --save_views > "${case_dir}/flame.log" 2>&1 || true
  fi

  if [[ -n "${FASTAVATAR_RENDER_CMD_TEMPLATE}" && -f "${post_npz}" ]]; then
    fa_dir="${case_dir}/fastavatar"
    mkdir -p "${fa_dir}"
    cmd="${FASTAVATAR_RENDER_CMD_TEMPLATE//\{motion\}/${post_npz}}"
    cmd="${cmd//\{out_dir\}/${fa_dir}}"
    if bash -lc "${cmd}" > "${case_dir}/fastavatar.log" 2>&1; then
      fa_status="rendered"
    else
      fa_status="render_failed"
    fi
  else
    echo "Set FASTAVATAR_RENDER_CMD_TEMPLATE with {motion} and {out_dir} to run actual FastAvatar rendering." > "${case_dir}/fastavatar_render_command.txt"
    fa_status="command_only"
  fi

  python scripts/diagnostics/compare_flame_fastavatar_motion_quality.py \
    --motion_npz "${post_npz}" \
    --flame_dir "${case_dir}/flame_preview" \
    --fastavatar_dir "${case_dir}/fastavatar" \
    --output_report "${case_dir}/case_report.md" \
    --render_issue "${prompt}" > "${case_dir}/compare.log" 2>&1 || true

  echo "| ${prompt} | ${gen_status} | [motion_report](${slug}/motion_stats/motion_report.md) | [flame_report](${slug}/flame_preview/flame_mesh_report.md) | ${fa_status} |" >> "${summary}"
done

cat >> "${summary}" <<'EOF'

## Required answers checklist

1. **Right-turn artifact vs yaw amplitude/velocity**: inspect each case's `motion_stats/motion_report.md` and `safety_report.json`; if post-safety yaw and max-step are safe but renders still artifact, classify as renderer/mask/source-view.
2. **Weak smile at FLAME layer**: inspect `flame_preview/flame_mesh_metrics.json`; low mouth-corner displacement indicates generator/prototype calibration issue.
3. **Paper-like motion**: compare boundary motion in FLAME/proxy metrics with mask sensitivity outputs. If FLAME boundary/head motion is low, increase safe motion amplitude; if FLAME is normal but render is flat, test mask/render range.
4. **Recommended parameters**: start from `configs/p7p8_motion_safety.yaml`, sweep safe right/left yaw scales via `sweep_p7p8_motion_quality.py`, use `calibrate_smile_primitive.py` for smile gain, and tune `head_smooth_window`/`max_head_step` from safety reports.
5. **Next steps**: run actual FastAvatar renders with `FASTAVATAR_RENDER_CMD_TEMPLATE`, then run `run_mask_sensitivity_render.sh` on cases where FLAME/proxy is normal but render artifacts remain.
EOF

echo "${summary}"
