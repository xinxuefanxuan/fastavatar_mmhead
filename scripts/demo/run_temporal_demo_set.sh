#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="outputs/mmhead_debug/text_temporal_planner_v1"
NPZ_DIR="$OUT_ROOT/generated_npz"
PLAN_DIR="$OUT_ROOT/plans"
mkdir -p "$NPZ_DIR" "$PLAN_DIR"

run_one() {
  local key="$1"
  local prompt="$2"
  echo "[TemporalDemo] $key :: $prompt"
  python motion_model/generate_from_text_temporal.py \
    --prompt "$prompt" \
    --output_npz "$NPZ_DIR/${key}.npz" \
    --output_plan_json "$PLAN_DIR/${key}.json"
}

run_one turn_left_then_smile "turn left then smile"
run_one turn_right_then_open_mouth "turn right then open mouth"
run_one slowly_look_down "slowly look down"
run_one quickly_open_mouth "quickly open mouth"
run_one look_up_then_smile "look up then smile"
run_one tilt_left_then_smile "tilt left then smile"

echo "[TemporalDemo] generated npz: $NPZ_DIR"
echo "[TemporalDemo] plans: $PLAN_DIR"
