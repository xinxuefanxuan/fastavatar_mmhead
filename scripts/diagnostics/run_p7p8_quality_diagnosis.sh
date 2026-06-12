#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmhead_debug/p7p8_quality_diagnosis}"
GENERATOR="${GENERATOR:-motion_model/generate_from_text_temporal.py}"
SAFETY_CONFIG="${SAFETY_CONFIG:-configs/p7p8_motion_safety.yaml}"
FASTAVATAR_RENDER_CMD_TEMPLATE="${FASTAVATAR_RENDER_CMD_TEMPLATE:-}"
LOG_DIR="outputs/mmhead_debug/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

printf '[P7/P8 diagnosis] cwd=%s\n' "$(pwd)"
printf '[P7/P8 diagnosis] git_branch=%s\n' "$(git branch --show-current 2>/dev/null || echo unknown)"
printf '[P7/P8 diagnosis] git_commit=%s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
printf '[P7/P8 diagnosis] python=%s\n' "$(command -v python || echo missing)"
printf '[P7/P8 diagnosis] GENERATOR=%s\n' "${GENERATOR}"
printf '[P7/P8 diagnosis] SAFETY_CONFIG=%s\n' "${SAFETY_CONFIG}"
printf '[P7/P8 diagnosis] OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
printf '[P7/P8 diagnosis] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"

if [[ ! -f "${GENERATOR}" ]]; then
  printf '[ERROR] GENERATOR file does not exist: %s\n' "${GENERATOR}" >&2
  exit 1
fi
if [[ ! -f "${SAFETY_CONFIG}" ]]; then
  printf '[ERROR] SAFETY_CONFIG file does not exist: %s\n' "${SAFETY_CONFIG}" >&2
  exit 1
fi

help_log="${LOG_DIR}/p7p8_quality_generator_help.log"
if ! python "${GENERATOR}" --help > "${help_log}" 2>&1; then
  printf '[ERROR] Failed to run generator --help. See %s\n' "${help_log}" >&2
  tail -n 80 "${help_log}" >&2 || true
  exit 1
fi
required_args=(
  "--prompt"
  "--output_npz"
  "--motion_safety_config"
  "--apply_motion_safety"
  "--save_pre_safety_motion"
  "--save_safety_report"
)
missing_args=()
for required_arg in "${required_args[@]}"; do
  if ! grep -q -- "${required_arg}" "${help_log}"; then
    missing_args+=("${required_arg}")
  fi
done
if (( ${#missing_args[@]} > 0 )); then
  printf '[ERROR] Generator help is missing required args: %s\n' "${missing_args[*]}" >&2
  cat "${help_log}" >&2
  exit 1
fi

prompts=("turn right" "turn left" "smile" "open mouth" "turn right and smile" "turn left and smile")
summary="${OUTPUT_DIR}/summary_report.md"
{
  printf '# P7/P8 Motion Quality Diagnosis Summary\n\n'
  printf 'Output directory: `%s`\n\n' "${OUTPUT_DIR}"
  printf '| case | generation | generate_log | motion stats | FLAME/proxy preview | FastAvatar |\n'
  printf '|---|---|---|---|---|---|\n'
} > "${summary}"

failed_generations=0
successful_generations=0

for prompt in "${prompts[@]}"; do
  slug="${prompt// /_}"
  case_dir="${OUTPUT_DIR}/${slug}"
  mkdir -p "${case_dir}"
  pre_npz="${case_dir}/motion_pre_safety.npz"
  post_npz="${case_dir}/motion_post_safety.npz"
  plan_json="${case_dir}/plan.json"
  generate_log="${case_dir}/generate.log"
  generate_log_link="[generate.log](${slug}/generate.log)"
  motion_link="skipped"
  flame_link="skipped"
  fa_status="skipped"

  printf '[P7/P8 diagnosis] generating case=%s\n' "${prompt}"
  if python "${GENERATOR}" \
      --prompt "${prompt}" \
      --output_npz "${post_npz}" \
      --output_plan_json "${plan_json}" \
      --motion_safety_config "${SAFETY_CONFIG}" \
      --apply_motion_safety \
      --save_pre_safety_motion "${pre_npz}" \
      --save_safety_report "${case_dir}/safety_report.json" \
      ${GENERATOR_ARGS:-} > "${generate_log}" 2>&1; then
    gen_status="ok"
    successful_generations=$((successful_generations + 1))
  else
    gen_status="failed"
    failed_generations=$((failed_generations + 1))
    printf '[ERROR] generation failed for case=%s; last 40 lines of %s:\n' "${prompt}" "${generate_log}" >&2
    tail -n 40 "${generate_log}" >&2 || true
  fi

  if [[ -f "${post_npz}" ]]; then
    python scripts/diagnostics/inspect_motion_npz.py --input "${pre_npz}" "${post_npz}" --output_dir "${case_dir}/motion_stats" > "${case_dir}/inspect.log" 2>&1 || true
    python scripts/diagnostics/render_flame_mesh_sequence.py --input "${post_npz}" --output_dir "${case_dir}/flame_preview" --wireframe --save_views > "${case_dir}/flame.log" 2>&1 || true
    motion_link="[motion_report](${slug}/motion_stats/motion_report.md)"
    flame_link="[flame_report](${slug}/flame_preview/flame_mesh_report.md)"

    if [[ -n "${FASTAVATAR_RENDER_CMD_TEMPLATE}" ]]; then
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
  else
    printf '[WARN] post-safety motion missing for case=%s; skipping inspect/flame/render/compare. Expected: %s\n' "${prompt}" "${post_npz}" >&2
  fi

  printf '| %s | %s | %s | %s | %s | %s |\n' "${prompt}" "${gen_status}" "${generate_log_link}" "${motion_link}" "${flame_link}" "${fa_status}" >> "${summary}"
done

cat >> "${summary}" <<'EOF'

## Required answers checklist

1. **Right-turn artifact vs yaw amplitude/velocity**: inspect each case's `motion_stats/motion_report.md` and `safety_report.json`; if post-safety yaw and max-step are safe but renders still artifact, classify as renderer/mask/source-view.
2. **Weak smile at FLAME layer**: inspect `flame_preview/flame_mesh_metrics.json`; low mouth-corner displacement indicates generator/prototype calibration issue.
3. **Paper-like motion**: compare boundary motion in FLAME/proxy metrics with mask sensitivity outputs. If FLAME boundary/head motion is low, increase safe motion amplitude; if FLAME is normal but render is flat, test mask/render range.
4. **Recommended parameters**: start from `configs/p7p8_motion_safety.yaml`, sweep safe right/left yaw scales via `sweep_p7p8_motion_quality.py`, use `calibrate_smile_primitive.py` for smile gain, and tune `head_smooth_window`/`max_head_step` from safety reports.
5. **Next steps**: run actual FastAvatar renders with `FASTAVATAR_RENDER_CMD_TEMPLATE`, then run `run_mask_sensitivity_render.sh` on cases where FLAME/proxy is normal but render artifacts remain.
EOF

printf '[P7/P8 diagnosis] summary=%s\n' "${summary}"
if (( successful_generations == 0 && failed_generations > 0 )); then
  printf '[ERROR] all %d case generations failed; see per-case generate.log files and %s\n' "${failed_generations}" "${summary}" >&2
  exit 1
fi
