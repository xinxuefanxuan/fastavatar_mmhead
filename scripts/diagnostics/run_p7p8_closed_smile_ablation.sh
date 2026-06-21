#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmhead_debug/p7p8_closed_smile_ablation}"
SOURCE_DIR="${SOURCE_DIR:-outputs/mmhead_debug/p7p8_smile_calibration/clean_candidate_npz}"
RENDER_WRAPPER="${RENDER_WRAPPER:-scripts/diagnostics/render_fastavatar_case.sh}"
SEQUENCE_NAME="${SEQUENCE_NAME:-nersemble_seq_214}"
NEUTRAL_TEMPLATE="${NEUTRAL_TEMPLATE:-assets/sample_motion/nersemble_seq_214_neutral}"
EXPR_GAINS="${EXPR_GAINS:-0.6 0.8 1.0 1.2}"
JAW_MODES="${JAW_MODES:-zero original_0.25 original_0.5}"
INFERENCE_N_FRAMES="${INFERENCE_N_FRAMES:-16}"
MAX_SINGLE_FRAME_RENDER="${MAX_SINGLE_FRAME_RENDER:-2}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-7}"
RENDER="${RENDER:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_RENDER_FAIL="${CONTINUE_ON_RENDER_FAIL:-1}"
LIMIT_COMBOS="${LIMIT_COMBOS:-0}"

NPZ_DIR="${OUTPUT_DIR}/ablation_npz"
FRAME_DIR="${OUTPUT_DIR}/frames"
VIDEO_DIR="${OUTPUT_DIR}/fastavatar_video"
REPORT="${OUTPUT_DIR}/closed_smile_ablation_report.md"
GRID="${OUTPUT_DIR}/closed_smile_ablation_grid.png"
MANIFEST="${OUTPUT_DIR}/closed_smile_ablation_manifest.json"
RENDER_STATUS="${OUTPUT_DIR}/closed_smile_render_status.tsv"

mkdir -p "${OUTPUT_DIR}" "${NPZ_DIR}" "${FRAME_DIR}" "${VIDEO_DIR}"

printf '[ClosedSmileAblation] cwd=%s\n' "$(pwd)"
printf '[ClosedSmileAblation] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[ClosedSmileAblation] source_dir=%s\n' "${SOURCE_DIR}"
printf '[ClosedSmileAblation] expr_gains=%s\n' "${EXPR_GAINS}"
printf '[ClosedSmileAblation] jaw_modes=%s\n' "${JAW_MODES}"
printf '[ClosedSmileAblation] render=%s inference_n_frames=%s max_single_frame_render=%s cuda=%s\n' \
  "${RENDER}" "${INFERENCE_N_FRAMES}" "${MAX_SINGLE_FRAME_RENDER}" "${CUDA_VISIBLE_DEVICES_VALUE}"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  printf '[ERROR] SOURCE_DIR not found: %s\n' "${SOURCE_DIR}" >&2
  exit 1
fi
if [[ "${RENDER}" == "1" && ! -f "${RENDER_WRAPPER}" ]]; then
  printf '[ERROR] render wrapper not found: %s\n' "${RENDER_WRAPPER}" >&2
  exit 1
fi

required=(
  "clean_smile_candidate_01_CELEBVTEXT_3KlGlvQ7Jzg_3_0.npz"
  "clean_smile_candidate_04_CELEBVTEXT_fy2SVmBBK5U_1_0.npz"
  "clean_smile_candidate_05_CELEBVHQ_83qspDarezc_8_0.npz"
  "clean_smile_candidate_08_CELEBVHQ_rLY0ubQwxes_26_0.npz"
  "clean_smile_candidate_10_CELEBVTEXT_NDmnK5fgpH8_18_0.npz"
  "clean_smile_candidate_12_CELEBVTEXT_iW3OkJWMz6I_24_0.npz"
  "clean_smile_candidate_03_CELEBVHQ__0tf2n3rlJU_0.npz"
)
for rel in "${required[@]}"; do
  if [[ ! -f "${SOURCE_DIR}/${rel}" ]]; then
    printf '[ERROR] required candidate missing: %s\n' "${SOURCE_DIR}/${rel}" >&2
    exit 1
  fi
done

python - "${SOURCE_DIR}" "${NPZ_DIR}" "${MANIFEST}" "${EXPR_GAINS}" "${JAW_MODES}" "${LIMIT_COMBOS}" <<'PY'
import json
import re
import sys
from pathlib import Path

import numpy as np

source_dir = Path(sys.argv[1])
npz_dir = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
expr_gains = [float(x) for x in sys.argv[4].split()]
jaw_modes = sys.argv[5].split()
limit = int(sys.argv[6])

npz_dir.mkdir(parents=True, exist_ok=True)

candidates = [
    {
        "label": "rank01_closed_subtle",
        "source": "clean_smile_candidate_01_CELEBVTEXT_3KlGlvQ7Jzg_3_0.npz",
        "sample_id": "CELEBVTEXT_3KlGlvQ7Jzg_3_0",
        "role": "closed_candidate",
        "prior_visual": "subtle closed-mouth smile; best rule score but weaker than rank3",
    },
    {
        "label": "rank04_cheerful_closed",
        "source": "clean_smile_candidate_04_CELEBVTEXT_fy2SVmBBK5U_1_0.npz",
        "sample_id": "CELEBVTEXT_fy2SVmBBK5U_1_0",
        "role": "closed_candidate",
        "prior_visual": "consistent cheerful expression with low jaw",
    },
    {
        "label": "rank05_soft_smile",
        "source": "clean_smile_candidate_05_CELEBVHQ_83qspDarezc_8_0.npz",
        "sample_id": "CELEBVHQ_83qspDarezc_8_0",
        "role": "closed_candidate",
        "prior_visual": "weak but natural candidate",
    },
    {
        "label": "rank08_gentle_dimple",
        "source": "clean_smile_candidate_08_CELEBVHQ_rLY0ubQwxes_26_0.npz",
        "sample_id": "CELEBVHQ_rLY0ubQwxes_26_0",
        "role": "closed_candidate",
        "prior_visual": "gentle dimple/left cheek candidate",
    },
    {
        "label": "rank10_wide_closed",
        "source": "clean_smile_candidate_10_CELEBVTEXT_NDmnK5fgpH8_18_0.npz",
        "sample_id": "CELEBVTEXT_NDmnK5fgpH8_18_0",
        "role": "closed_candidate",
        "prior_visual": "wide-smile text but low jaw metrics",
    },
    {
        "label": "rank12_low_jaw_subtle",
        "source": "clean_smile_candidate_12_CELEBVTEXT_iW3OkJWMz6I_24_0.npz",
        "sample_id": "CELEBVTEXT_iW3OkJWMz6I_24_0",
        "role": "closed_candidate",
        "prior_visual": "lowest jaw candidate; may be weak",
    },
    {
        "label": "rank03_negative_forced_grin",
        "source": "clean_smile_candidate_03_CELEBVHQ__0tf2n3rlJU_0.npz",
        "sample_id": "CELEBVHQ__0tf2n3rlJU_0",
        "role": "negative_control",
        "prior_visual": "user rejected: exposed teeth / forced grin / open teeth smile",
    },
]


def load_motion(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    key = "motion" if "motion" in data else "motion_raw" if "motion_raw" in data else data.files[0]
    motion = np.asarray(data[key], dtype=np.float32)
    if motion.ndim == 1:
        motion = motion[None, :]
    if motion.shape[1] < 56:
        raise ValueError(f"expected [T,>=56], got {motion.shape}")
    return motion[:, :56]


def jaw_scale(mode: str) -> float:
    if mode == "zero":
        return 0.0
    if mode.startswith("original_"):
        return float(mode.split("_", 1)[1])
    raise ValueError(f"unknown jaw mode: {mode}")


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


items = []
counter = 0
for cand in candidates:
    source_path = source_dir / cand["source"]
    src = load_motion(source_path)
    src_expr = src[:, :50]
    src_jaw = src[:, 53:56]
    src_head = src[:, 50:53]
    for gain in expr_gains:
        for jaw_mode in jaw_modes:
            counter += 1
            if limit and counter > limit:
                break
            scale = jaw_scale(jaw_mode)
            motion = np.zeros_like(src, dtype=np.float32)
            motion[:, :50] = src_expr * gain
            motion[:, 50:53] = 0.0
            motion[:, 53:56] = src_jaw * scale

            case_name = f"{counter:03d}_{cand['label']}_gain{gain:.1f}_{jaw_mode}"
            case_name = slug(case_name)
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
            expr_norm = np.linalg.norm(motion[:, :50], axis=1)
            jaw_norm = np.linalg.norm(motion[:, 53:56], axis=1)
            source_jaw_norm = np.linalg.norm(src_jaw, axis=1)
            source_head_norm = np.linalg.norm(src_head, axis=1)
            items.append({
                **cand,
                "case_name": case_name,
                "source_npz": str(source_path),
                "motion_npz": str(out_npz),
                "expr_gain": gain,
                "jaw_mode": jaw_mode,
                "jaw_scale": scale,
                "expr_norm_max": float(expr_norm.max(initial=0.0)),
                "jaw_norm_max": float(jaw_norm.max(initial=0.0)),
                "source_jaw_norm_max": float(source_jaw_norm.max(initial=0.0)),
                "source_head_norm_max": float(source_head_norm.max(initial=0.0)),
                "video": f"fastavatar_video/{case_name}.mp4",
                "render_log": f"{case_name}/render_wrapper.log",
                "manual_verdict": "TODO inspect grid/video",
            })
        if limit and counter >= limit:
            break
    if limit and counter >= limit:
        break

payload = {
    "output_dir": str(manifest_path.parent),
    "expr_gains": expr_gains,
    "jaw_modes": jaw_modes,
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
      printf '[ClosedSmileAblation] skipping existing video %s\n' "${case_name}"
      printf '%s\tskipped_existing\t%s\t%s\n' \
        "${case_name}" "${VIDEO_DIR}/${case_name}.mp4" "${OUTPUT_DIR}/${case_name}/render_wrapper.log" >> "${RENDER_STATUS}"
      continue
    fi
    printf '[ClosedSmileAblation] rendering %s\n' "${case_name}"
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
  printf '[ClosedSmileAblation] RENDER=0, skipping FastAvatar render and grid generation\n'
  printf 'case\tstatus\tvideo\tlog\n' > "${RENDER_STATUS}"
fi

python - "${MANIFEST}" "${REPORT}" "${GRID}" "${RENDER_STATUS}" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
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


def cell(text, n=120):
    text = " ".join(str(text).replace("|", "/").split())
    return text[: n - 3] + "..." if len(text) > n else text


by_candidate = defaultdict(list)
for item in items:
    by_candidate[item["label"]].append(item)

status_counts = Counter(item["render_status"] for item in items)
closed = [x for x in items if x["role"] == "closed_candidate"]
negative = [x for x in items if x["role"] == "negative_control"]
priority = [
    x for x in closed
    if x["jaw_mode"] in {"zero", "original_0.25"} and x["expr_gain"] in {0.6, 0.8, 1.0}
]
priority.sort(key=lambda x: (
    x["render_status"] not in {"ok", "skipped_existing"},
    x["jaw_norm_max"],
    abs(x["expr_gain"] - 0.8),
))
best_rule = priority[0] if priority else (closed[0] if closed else items[0])

lines = [
    "# P7/P8 Closed-Mouth Smile Ablation",
    "",
    f"Output directory: `{report.parent}`",
    f"Grid: `{grid.name}`" if grid.exists() else "Grid: not generated yet (`RENDER=0` or render incomplete)",
    f"Manifest: `{manifest.get('output_dir', report.parent)}/closed_smile_ablation_manifest.json`",
    f"Render status: `{render_status_path}`",
    "",
    "## Setup",
    "",
    f"- expr_gain values: `{', '.join(str(x) for x in manifest['expr_gains'])}`",
    f"- jaw_mode values: `{', '.join(manifest['jaw_modes'])}`",
    f"- total combinations: `{manifest['total_combos']}`",
    f"- render status counts: `{dict(status_counts)}`",
    "- head pose is forced neutral in generated ablation motions.",
    "- `rank03_negative_forced_grin` is included only as a negative control because user review rejected it as exposed-teeth / forced-grin.",
    "",
    "## Candidate Summary",
    "",
    "| label | role | source sample | combinations | source npz | prior visual note |",
    "|---|---|---|---:|---|---|",
]
for label, rows in by_candidate.items():
    first = rows[0]
    lines.append(
        f"| `{label}` | `{first['role']}` | `{first['sample_id']}` | {len(rows)} | "
        f"`{first['source_npz']}` | {cell(first['prior_visual'], 100)} |"
    )

lines += [
    "",
    "## Ablation Table",
    "",
    "| case | role | expr_gain | jaw_mode | jaw max | expr max | render | video | note |",
    "|---|---|---:|---|---:|---:|---|---|---|",
]
for item in items:
    lines.append(
        f"| `{item['case_name']}` | `{item['role']}` | {item['expr_gain']:.1f} | "
        f"`{item['jaw_mode']}` | {item['jaw_norm_max']:.4f} | {item['expr_norm_max']:.4f} | "
        f"`{item['render_status']}` | [video]({item['video']}) | {item['manual_verdict']} |"
    )

rank3_zero = [
    x for x in negative
    if x["jaw_mode"] == "zero" and x["render_status"] in {"ok", "skipped_existing"}
]

lines += [
    "",
    "## Questions To Answer From The Grid",
    "",
    f"- Rule-based best natural closed-mouth candidate before manual grid review: `{best_rule['case_name']}`.",
    f"- Rule-based default basis candidate: `{best_rule['sample_id']}`.",
    f"- Rule-based candidate npz: `{best_rule['motion_npz']}`.",
    "- Preferred default smile_gain should come from the weakest natural visible setting, usually `0.8` if visible or `1.0` if `0.8` is too weak.",
    "- Mark combinations with exposed teeth, large mouth opening, lip tightening, or grimace/forced grin after inspecting the grid.",
    "- If rank3 remains unnatural with `jaw_mode=zero`, exclude it from default smile and keep it only as an open_teeth_smile / grin-style control.",
    "",
    "## Manual Visual Review",
    "",
    "- TODO inspect `closed_smile_ablation_grid.png` and videos.",
    "- Which candidate + expr_gain + jaw_mode is most natural: TODO.",
    "- Which combinations expose teeth / look strange / look like grimace: TODO.",
    "- rank3 with jaw zero verdict: TODO.",
    "- Recommended default clean smile basis: TODO.",
    "- Recommended default smile_gain: TODO.",
    "- Whether to keep a separate open_teeth_smile / laugh primitive: likely yes if rank3 remains useful as expressive control, but not as default smile.",
]
if rank3_zero:
    lines.append("")
    lines.append("Rendered rank3 jaw-zero controls:")
    for item in rank3_zero:
        lines.append(f"- `{item['case_name']}` -> [video]({item['video']})")

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[ClosedSmileAblation] report={report}")
PY

printf '[ClosedSmileAblation] manifest=%s\n' "${MANIFEST}"
printf '[ClosedSmileAblation] report=%s\n' "${REPORT}"
if [[ -f "${GRID}" ]]; then
  printf '[ClosedSmileAblation] grid=%s\n' "${GRID}"
fi
