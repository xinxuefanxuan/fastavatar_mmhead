#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmhead_debug/p7p8_gentle_teeth_smile_ablation}"
RENDER_WRAPPER="${RENDER_WRAPPER:-scripts/diagnostics/render_fastavatar_case.sh}"
SEQUENCE_NAME="${SEQUENCE_NAME:-nersemble_seq_214}"
NEUTRAL_TEMPLATE="${NEUTRAL_TEMPLATE:-assets/sample_motion/nersemble_seq_214_neutral}"
GAINS="${GAINS:-0.5 0.7 0.9}"
JAW_MODES="${JAW_MODES:-original_0.15 original_0.25 original_0.35}"
JAW_CAPS="${JAW_CAPS:-0.025 0.04 0.055}"
CANDIDATE_RANKS="${CANDIDATE_RANKS:-}"
CASE_SPECS="${CASE_SPECS:-}"
JAW_SOURCE_MODE="${JAW_SOURCE_MODE:-stable_peak}"
INFERENCE_N_FRAMES="${INFERENCE_N_FRAMES:-16}"
MAX_SINGLE_FRAME_RENDER="${MAX_SINGLE_FRAME_RENDER:-2}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-7}"
RENDER="${RENDER:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_RENDER_FAIL="${CONTINUE_ON_RENDER_FAIL:-1}"
LIMIT_COMBOS="${LIMIT_COMBOS:-0}"

GAINS="${GAINS//,/ }"
JAW_MODES="${JAW_MODES//,/ }"
JAW_CAPS="${JAW_CAPS//,/ }"
CANDIDATE_RANKS="${CANDIDATE_RANKS//,/ }"
CASE_SPECS="${CASE_SPECS//,/ }"

NPZ_DIR="${OUTPUT_DIR}/ablation_npz"
FRAME_DIR="${OUTPUT_DIR}/frames"
VIDEO_DIR="${OUTPUT_DIR}/fastavatar_video"
REPORT="${REPORT:-${OUTPUT_DIR}/gentle_teeth_smile_report.md}"
GRID="${GRID:-${OUTPUT_DIR}/gentle_teeth_smile_grid.png}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest.json}"
RENDER_STATUS="${RENDER_STATUS:-${OUTPUT_DIR}/gentle_teeth_smile_render_status.tsv}"

mkdir -p "${OUTPUT_DIR}" "${NPZ_DIR}" "${FRAME_DIR}" "${VIDEO_DIR}"
rm -f "${GRID}"

printf '[GentleTeethSmile] cwd=%s\n' "$(pwd)"
printf '[GentleTeethSmile] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[GentleTeethSmile] gains=%s\n' "${GAINS}"
printf '[GentleTeethSmile] jaw_modes=%s\n' "${JAW_MODES}"
printf '[GentleTeethSmile] jaw_caps=%s\n' "${JAW_CAPS}"
printf '[GentleTeethSmile] jaw_source_mode=%s\n' "${JAW_SOURCE_MODE}"
printf '[GentleTeethSmile] candidate_ranks=%s\n' "${CANDIDATE_RANKS:-all}"
printf '[GentleTeethSmile] case_specs=%s\n' "${CASE_SPECS:-all}"
printf '[GentleTeethSmile] render=%s inference_n_frames=%s max_single_frame_render=%s cuda=%s\n' \
  "${RENDER}" "${INFERENCE_N_FRAMES}" "${MAX_SINGLE_FRAME_RENDER}" "${CUDA_VISIBLE_DEVICES_VALUE}"

if [[ "${RENDER}" == "1" && ! -f "${RENDER_WRAPPER}" ]]; then
  printf '[ERROR] render wrapper not found: %s\n' "${RENDER_WRAPPER}" >&2
  exit 1
fi

python - "${NPZ_DIR}" "${MANIFEST}" "${GAINS}" "${JAW_MODES}" "${JAW_CAPS}" "${LIMIT_COMBOS}" "${CANDIDATE_RANKS}" "${JAW_SOURCE_MODE}" "${CASE_SPECS}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

npz_dir = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
gains = [float(x) for x in sys.argv[3].split()]
jaw_modes = sys.argv[4].split()
jaw_caps = [float(x) for x in sys.argv[5].split()]
limit = int(sys.argv[6])
candidate_ranks = {x for x in sys.argv[7].split()}
jaw_source_mode = sys.argv[8]
case_specs_arg = sys.argv[9].split()


def norm_float(x: float) -> str:
    return f"{float(x):.3f}"


case_specs = set()
for spec in case_specs_arg:
    parts = spec.split(":")
    if len(parts) != 4:
        raise SystemExit(
            "CASE_SPECS entries must be rank:gain:jaw_mode:jaw_cap, "
            f"got {spec!r}"
        )
    case_specs.add((parts[0], norm_float(parts[1]), parts[2], norm_float(parts[3])))

npz_dir.mkdir(parents=True, exist_ok=True)

candidates = [
    {
        "rank": "01",
        "label": "rank01_closed_to_gentle",
        "sample_id": "CELEBVTEXT_3KlGlvQ7Jzg_3_0",
        "source_npz": "outputs/mmhead_debug/p7p8_smile_calibration/clean_candidate_npz/clean_smile_candidate_01_CELEBVTEXT_3KlGlvQ7Jzg_3_0.npz",
        "role": "clean_base",
        "note": "previous best subtle closed-mouth direction; needs gentle jaw to become visible smile",
    },
    {
        "rank": "04",
        "label": "rank04_cheerful_gentle",
        "sample_id": "CELEBVTEXT_fy2SVmBBK5U_1_0",
        "source_npz": "outputs/mmhead_debug/p7p8_smile_calibration/clean_candidate_npz/clean_smile_candidate_04_CELEBVTEXT_fy2SVmBBK5U_1_0.npz",
        "role": "clean_base",
        "note": "second-best closed-mouth candidate; may become natural with small jaw opening",
    },
    {
        "rank": "10",
        "label": "rank10_wide_control",
        "sample_id": "CELEBVTEXT_NDmnK5fgpH8_18_0",
        "source_npz": "outputs/mmhead_debug/p7p8_smile_calibration/clean_candidate_npz/clean_smile_candidate_10_CELEBVTEXT_NDmnK5fgpH8_18_0.npz",
        "role": "wide_smile_control",
        "note": "previous shortlist became toothy/grin-like; test lower gain and capped jaw",
    },
    {
        "rank": "03",
        "label": "rank03_negative_forced_grin",
        "sample_id": "CELEBVHQ__0tf2n3rlJU_0",
        "source_npz": "outputs/mmhead_debug/p7p8_smile_calibration/clean_candidate_npz/clean_smile_candidate_03_CELEBVHQ__0tf2n3rlJU_0.npz",
        "role": "negative_control",
        "note": "user rejected as unnatural exposed-teeth / forced grin; test low gain only to confirm exclusion",
    },
    {
        "rank": "open01",
        "label": "open01_joyful_smile",
        "sample_id": "CELEBVTEXT_ShFpGNbUM4A_9_0",
        "source_npz": "outputs/mmhead_debug/p7p8_smile_calibration/candidate_npz/smile_candidate_01_CELEBVTEXT_ShFpGNbUM4A_9_0.npz",
        "role": "gentle_teeth_candidate",
        "note": "balanced candidate: smile intensifies into joyful smile; not talk text, use middle stable segment and cap jaw",
    },
    {
        "rank": "open10",
        "label": "open10_smile_turn_control",
        "sample_id": "CELEBVTEXT_9BpaCRU9_Ko_10_0",
        "source_npz": "outputs/mmhead_debug/p7p8_smile_calibration/candidate_npz/smile_candidate_10_CELEBVTEXT_9BpaCRU9_Ko_10_0.npz",
        "role": "gentle_teeth_candidate",
        "note": "balanced candidate: smile then turns/looks around; no laugh/talk label, head is neutralized",
    },
]

if candidate_ranks:
    selected = [c for c in candidates if c["rank"] in candidate_ranks]
    missing = sorted(candidate_ranks - {c["rank"] for c in selected})
    if missing:
        raise SystemExit(f"unknown CANDIDATE_RANKS: {','.join(missing)}")
    candidates = selected

for cand in candidates:
    if not Path(cand["source_npz"]).is_file():
        raise SystemExit(f"source npz missing: {cand['source_npz']}")


def load_motion(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    key = "motion" if "motion" in data else "motion_raw" if "motion_raw" in data else data.files[0]
    motion = np.asarray(data[key], dtype=np.float32)
    if motion.ndim == 1:
        motion = motion[None, :]
    if motion.shape[1] < 56:
        raise ValueError(f"expected [T,>=56], got {motion.shape} in {path}")
    return motion[:, :56]


def jaw_scale(mode: str) -> float:
    if not mode.startswith("original_"):
        raise ValueError(f"jaw mode must be original_<scale>, got {mode}")
    return float(mode.split("_", 1)[1])


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def ramp_hold_release(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((length, 1), dtype=np.float32)
    n_ramp = max(2, length // 4)
    n_hold = max(1, length - 2 * n_ramp)
    up = np.linspace(0.0, 1.0, n_ramp, endpoint=False, dtype=np.float32)
    hold = np.ones(n_hold, dtype=np.float32)
    down = np.linspace(1.0, 0.0, length - n_ramp - n_hold, endpoint=True, dtype=np.float32)
    env = np.concatenate([up, hold, down])
    return env[:length, None]


def stable_slice(motion: np.ndarray) -> np.ndarray:
    start = motion.shape[0] // 4
    end = max(start + 1, (motion.shape[0] * 3) // 4)
    return motion[start:end]


items = []
seen = {}
counter = 0
for cand in candidates:
    src_path = Path(cand["source_npz"])
    src = load_motion(src_path)
    stable = stable_slice(src)
    stable_expr = stable[:, :50].mean(axis=0, keepdims=True)
    if jaw_source_mode == "stable_mean":
        stable_jaw = stable[:, 53:56].mean(axis=0, keepdims=True)
    elif jaw_source_mode == "stable_peak":
        jaw_x = np.abs(stable[:, 53])
        stable_jaw = stable[int(np.argmax(jaw_x)): int(np.argmax(jaw_x)) + 1, 53:56]
    elif jaw_source_mode == "full_peak":
        jaw_x = np.abs(src[:, 53])
        stable_jaw = src[int(np.argmax(jaw_x)): int(np.argmax(jaw_x)) + 1, 53:56]
    else:
        raise ValueError(f"unknown JAW_SOURCE_MODE: {jaw_source_mode}")
    length = src.shape[0]
    env = ramp_hold_release(length)
    for gain in gains:
        for jaw_mode in jaw_modes:
            scale = jaw_scale(jaw_mode)
            for jaw_cap in jaw_caps:
                combo_key = (cand["rank"], norm_float(gain), jaw_mode, norm_float(jaw_cap))
                if case_specs and combo_key not in case_specs:
                    continue
                counter += 1
                if limit and counter > limit:
                    break
                expr = np.repeat(stable_expr, length, axis=0) * gain * env
                jaw = np.repeat(stable_jaw, length, axis=0) * scale * env
                jaw[:, 0] = np.clip(jaw[:, 0], -jaw_cap, jaw_cap)
                motion = np.zeros((length, 56), dtype=np.float32)
                motion[:, :50] = expr.astype(np.float32)
                motion[:, 50:53] = 0.0
                motion[:, 53:56] = jaw.astype(np.float32)

                digest = hashlib.sha1(np.round(motion, 6).tobytes()).hexdigest()[:10]
                duplicate_of = seen.get(digest)
                case_name = slug(
                    f"{counter:03d}_{cand['label']}_gain{gain:.1f}_{jaw_mode}_cap{jaw_cap:.3f}"
                )
                out_npz = npz_dir / f"{case_name}.npz"
                np.savez(
                    out_npz,
                    motion=motion,
                    motion_raw=motion,
                    motion_norm=motion,
                    expr_delta=motion[:, :50],
                    head_delta=motion[:, 50:53],
                    jaw_delta=motion[:, 53:56],
                )
                if duplicate_of is None:
                    seen[digest] = case_name
                expr_norm = np.linalg.norm(motion[:, :50], axis=1)
                jaw_norm = np.linalg.norm(motion[:, 53:56], axis=1)
                items.append({
                    **cand,
                    "case_name": case_name,
                    "motion_npz": str(out_npz),
                    "expr_gain": gain,
                    "jaw_mode": jaw_mode,
                    "jaw_scale": scale,
                    "jaw_cap": jaw_cap,
                    "jaw_x_max": float(np.max(np.abs(motion[:, 53]), initial=0.0)),
                    "jaw_norm_max": float(jaw_norm.max(initial=0.0)),
                    "expr_norm_max": float(expr_norm.max(initial=0.0)),
                    "duplicate_of": duplicate_of,
                    "video": f"fastavatar_video/{case_name}.mp4",
                    "render_log": f"{case_name}/render_wrapper.log",
                    "manual_verdict": "TODO inspect grid/video",
                })
            if limit and counter >= limit:
                break
        if limit and counter >= limit:
            break
    if limit and counter >= limit:
        break

if not items:
    raise SystemExit("no gentle-teeth smile ablation items generated")

payload = {
    "output_dir": str(manifest_path.parent),
    "candidate_ranks": sorted(candidate_ranks) if candidate_ranks else "all",
    "case_specs": sorted(":".join(x) for x in case_specs) if case_specs else "all",
    "jaw_source_mode": jaw_source_mode,
    "gains": gains,
    "jaw_modes": jaw_modes,
    "jaw_caps": jaw_caps,
    "limit_combos": limit,
    "total_combos": len(items),
    "items": items,
}
manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
for item in items:
    print(f"{item['case_name']}|{item['motion_npz']}")
PY

if [[ "${RENDER}" == "1" ]]; then
  printf 'case\tstatus\tvideo\tlog\n' > "${RENDER_STATUS}"
  mapfile -t CASES < <(python - "${MANIFEST}" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for item in data["items"]:
    print(f"{item['case_name']}|{item['motion_npz']}")
PY
)

  for item in "${CASES[@]}"; do
    IFS='|' read -r case_name motion_npz <<< "${item}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${VIDEO_DIR}/${case_name}.mp4" ]]; then
      printf '[GentleTeethSmile] skipping existing video %s\n' "${case_name}"
      printf '%s\tskipped_existing\t%s\t%s\n' \
        "${case_name}" "${VIDEO_DIR}/${case_name}.mp4" "${OUTPUT_DIR}/${case_name}/render_wrapper.log" >> "${RENDER_STATUS}"
      continue
    fi
    printf '[GentleTeethSmile] rendering %s\n' "${case_name}"
    mkdir -p "${OUTPUT_DIR}/${case_name}"
    if MAX_SINGLE_FRAME_RENDER="${MAX_SINGLE_FRAME_RENDER}" bash "${RENDER_WRAPPER}" \
      --motion_npz "${motion_npz}" \
      --output_dir "${OUTPUT_DIR}" \
      --case_name "${case_name}" \
      --sequence_name "${SEQUENCE_NAME}" \
      --neutral_template "${NEUTRAL_TEMPLATE}" \
      --pack_root "${OUTPUT_DIR}/${case_name}/fastavatar_pack" \
      --head_target neck_pose \
      --inference_n_frames "${INFERENCE_N_FRAMES}" \
      --cuda_visible_devices "${CUDA_VISIBLE_DEVICES_VALUE}" > "${OUTPUT_DIR}/${case_name}/render_wrapper.log" 2>&1; then
      printf '%s\tok\t%s\t%s\n' \
        "${case_name}" "${VIDEO_DIR}/${case_name}.mp4" "${OUTPUT_DIR}/${case_name}/render_wrapper.log" >> "${RENDER_STATUS}"
    else
      printf '[WARN] render failed for %s; see %s\n' "${case_name}" "${OUTPUT_DIR}/${case_name}/render_wrapper.log" >&2
      tail -n 80 "${OUTPUT_DIR}/${case_name}/render_wrapper.log" >&2 || true
      printf '%s\tfailed\t%s\t%s\n' \
        "${case_name}" "${VIDEO_DIR}/${case_name}.mp4" "${OUTPUT_DIR}/${case_name}/render_wrapper.log" >> "${RENDER_STATUS}"
      if [[ "${CONTINUE_ON_RENDER_FAIL}" != "1" ]]; then
        exit 1
      fi
    fi
  done

  mapfile -t ALL_CASES < <(python - "${MANIFEST}" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for item in data["items"]:
    print(item["case_name"])
PY
)
  for case_name in "${ALL_CASES[@]}"; do
    video="${VIDEO_DIR}/${case_name}.mp4"
    if [[ ! -f "${video}" ]]; then
      printf '[WARN] expected video missing, skip frames: %s\n' "${video}" >&2
      continue
    fi
    frame_count="$(ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "${video}" || true)"
    if [[ -z "${frame_count}" || "${frame_count}" == "N/A" ]]; then
      frame_count="${INFERENCE_N_FRAMES}"
    fi
    mid_frame=$(( frame_count / 2 ))
    last_frame=$(( frame_count > 0 ? frame_count - 1 : 0 ))
    ffmpeg -y -hide_banner -loglevel error -i "${video}" \
      -vf "select='eq(n,0)+eq(n,${mid_frame})+eq(n,${last_frame})'" -vsync 0 \
      "${FRAME_DIR}/${case_name}_%02d.png"
  done

  montage_inputs=()
  for case_name in "${ALL_CASES[@]}"; do
    for i in 01 02 03; do
      frame="${FRAME_DIR}/${case_name}_${i}.png"
      [[ -f "${frame}" ]] && montage_inputs+=("${frame}")
    done
  done
  if (( ${#montage_inputs[@]} > 0 )); then
    montage "${montage_inputs[@]}" -tile 12x -geometry 192x192+6+6 "${GRID}"
  fi
else
  printf '[GentleTeethSmile] RENDER=0, skipping FastAvatar render and grid generation\n'
  printf 'case\tstatus\tvideo\tlog\n' > "${RENDER_STATUS}"
fi

python - "${MANIFEST}" "${REPORT}" "${GRID}" "${RENDER_STATUS}" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.load(open(manifest_path, encoding="utf-8"))
report = Path(sys.argv[2])
grid = Path(sys.argv[3])
render_status_path = Path(sys.argv[4])

status = {}
if render_status_path.exists():
    for line in render_status_path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 4:
            status[parts[0]] = {"status": parts[1], "video": parts[2], "log": parts[3]}

items = manifest["items"]
for item in items:
    st = status.get(item["case_name"], {})
    item["render_status"] = st.get("status", "not_run")
    item["render_log"] = st.get("log", item["render_log"])


def cell(text, n=110):
    text = " ".join(str(text).replace("|", "/").split())
    return text[: n - 3] + "..." if len(text) > n else text


status_counts = Counter(item["render_status"] for item in items)
by_candidate = defaultdict(list)
for item in items:
    by_candidate[item["label"]].append(item)

rank03 = [
    x for x in items
    if x["rank"] == "03" and x["render_status"] in {"ok", "skipped_existing"}
]
priority = [
    x for x in items
    if x["role"] != "negative_control"
    and x["render_status"] in {"ok", "skipped_existing", "not_run"}
    and x["jaw_cap"] <= 0.04
    and x["expr_gain"] in {0.5, 0.7}
]
priority.sort(key=lambda x: (
    x["render_status"] not in {"ok", "skipped_existing"},
    abs(x["expr_gain"] - 0.7),
    abs(x["jaw_cap"] - 0.04),
    x["rank"],
))
best_rule = priority[0] if priority else items[0]

lines = [
    "# P7/P8 Gentle Teeth Smile Ablation",
    "",
    f"Output directory: `{report.parent}`",
    f"Grid: `{grid.name}`" if grid.exists() else "Grid: not generated yet (`RENDER=0` or render incomplete)",
    f"Manifest: `{manifest_path}`",
    f"Render status: `{render_status_path}`",
    "",
    "## Setup",
    "",
    f"- candidate ranks: `{manifest.get('candidate_ranks')}`",
    f"- case specs: `{manifest.get('case_specs')}`",
    f"- expr_gain values: `{', '.join(str(x) for x in manifest['gains'])}`",
    f"- jaw_mode values: `{', '.join(manifest['jaw_modes'])}`",
    f"- jaw x caps: `{', '.join(str(x) for x in manifest['jaw_caps'])}`",
    f"- jaw source mode: `{manifest.get('jaw_source_mode')}`",
    f"- total combinations: `{manifest['total_combos']}`",
    f"- render status counts: `{dict(status_counts)}`",
    "- head pose is forced neutral in generated ablation motions.",
    "- expressions are built from the middle stable segment; jaw uses the configured middle-segment source mode and a ramp-hold-release envelope.",
    "- full original jaw is never used; jaw x is capped after scaling.",
    "",
    "## Candidate Summary",
    "",
    "| label | role | sample_id | combinations | source npz | note |",
    "|---|---|---|---:|---|---|",
]
for label, rows in by_candidate.items():
    first = rows[0]
    lines.append(
        f"| `{label}` | `{first['role']}` | `{first['sample_id']}` | {len(rows)} | "
        f"`{first['source_npz']}` | {cell(first['note'])} |"
    )

lines += [
    "",
    "## Ablation Table",
    "",
    "| case | role | gain | jaw_mode | jaw_cap | jaw_x_max | expr_max | render | video | note |",
    "|---|---|---:|---|---:|---:|---:|---|---|---|",
]
for item in items:
    duplicate_note = f"duplicate of `{item['duplicate_of']}`" if item.get("duplicate_of") else "TODO inspect grid/video"
    lines.append(
        f"| `{item['case_name']}` | `{item['role']}` | {item['expr_gain']:.1f} | "
        f"`{item['jaw_mode']}` | {item['jaw_cap']:.3f} | {item['jaw_x_max']:.4f} | "
        f"{item['expr_norm_max']:.4f} | `{item['render_status']}` | [video]({item['video']}) | {duplicate_note} |"
    )

lines += [
    "",
    "## Questions To Answer From The Grid",
    "",
    f"- Rule-based starting point before manual review: `{best_rule['case_name']}`.",
    "- Mark each promising case for natural teeth, smile-likeness, grimace/biting, mouth-corner lift, and relaxed face.",
    "- If rank03 is still forced at low gain and small jaw, exclude it completely from default smile.",
    "- Decide whether default smile should be gentle_teeth_smile or subtle_closed_smile after visual review.",
    "",
    "## Manual Visual Review",
    "",
    f"- TODO inspect `{grid.name}` and videos.",
    "- Is it natural light teeth: TODO.",
    "- Does it read as smile: TODO.",
    "- Does it look like biting/grimace/forced grin: TODO.",
    "- Are mouth corners lifted: TODO.",
    "- Is the face relaxed: TODO.",
    "- Recommended top 3 gentle teeth smile: TODO.",
    "- rank03 low-gain/small-jaw verdict: TODO.",
    "- Final default smile recommendation: TODO.",
    "- Keep subtle_closed_smile as smile_subtle: TODO.",
    "- Keep open_teeth/grin as separate primitive: TODO.",
]
if rank03:
    lines.append("")
    lines.append("Rendered rank03 controls:")
    for item in rank03:
        lines.append(f"- `{item['case_name']}` -> [video]({item['video']})")

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[GentleTeethSmile] report={report}")
PY

printf '[GentleTeethSmile] manifest=%s\n' "${MANIFEST}"
printf '[GentleTeethSmile] report=%s\n' "${REPORT}"
if [[ -f "${GRID}" ]]; then
  printf '[GentleTeethSmile] grid=%s\n' "${GRID}"
fi
