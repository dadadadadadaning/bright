#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/shared-nvme/BRIGHT-master}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/eval_analysis/model_eval_config.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
EVAL_DESTROYED_CHECKPOINT="${EVAL_DESTROYED_CHECKPOINT:-1}"
FORCE="${FORCE:-0}"

cd "${PROJECT_ROOT}" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/bda_benchmark:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

mapfile -t MODEL_NAMES < <("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
for model in cfg["models"]:
    print(model["name"])
PY
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  RESOLVED="$("${PYTHON_BIN}" - "${CONFIG_PATH}" "${MODEL_NAME}" "${EVAL_DESTROYED_CHECKPOINT}" <<'PY'
import glob
import json
import os
import shlex
import sys

config_path, model_name, eval_destroyed = sys.argv[1:4]
with open(config_path, encoding="utf-8") as f:
    cfg = json.load(f)
model = next(item for item in cfg["models"] if item["name"] == model_name)
checkpoint_names = set(cfg["primary_checkpoint_priority"] + [cfg["destroyed_checkpoint"]])

run_dirs = []
for pattern in model["experiment_candidates"]:
    for candidate in glob.glob(pattern):
        if not os.path.isdir(candidate):
            continue
        if any(os.path.isfile(os.path.join(candidate, name)) for name in checkpoint_names):
            run_dirs.append(candidate)
        for root, _dirs, files in os.walk(candidate):
            if checkpoint_names.intersection(files):
                run_dirs.append(root)

if not run_dirs:
    print("STATUS=NOT_FOUND")
    print("CANDIDATES=" + shlex.quote(" | ".join(model["experiment_candidates"])))
    raise SystemExit

run_dir = max(set(run_dirs), key=os.path.getmtime)
primary = next(
    (name for name in cfg["primary_checkpoint_priority"]
     if os.path.isfile(os.path.join(run_dir, name))),
    "",
)
checkpoints = []
if primary:
    checkpoints.append(primary)
destroyed = cfg["destroyed_checkpoint"]
if eval_destroyed == "1" and destroyed != primary and os.path.isfile(os.path.join(run_dir, destroyed)):
    checkpoints.append(destroyed)

print("STATUS=OK")
print("RUN_DIR=" + shlex.quote(run_dir))
print("MODEL_TYPE=" + shlex.quote(model["model_type"]))
print("ARGS=" + shlex.quote(" ".join(str(value) for value in model["args"])))
print("CHECKPOINTS=" + shlex.quote(" ".join(checkpoints)))
print("PRIMARY_WARNING=" + shlex.quote("" if primary else "no primary checkpoint found"))
print("DESTROYED_WARNING=" + shlex.quote(
    "" if eval_destroyed != "1" or os.path.isfile(os.path.join(run_dir, destroyed))
    else f"missing optional checkpoint: {destroyed}"
))
for key in ("dataset_path", "data_list_path", "output_root", "image_size", "batch_size", "num_workers"):
    print(key.upper() + "=" + shlex.quote(str(cfg[key])))
PY
)"

  eval "${RESOLVED}"
  if [[ "${STATUS:-}" != "OK" ]]; then
    echo "WARNING: ${MODEL_NAME}: no experiment directory found"
    echo "  candidates: ${CANDIDATES:-unknown}"
    continue
  fi

  if [[ -z "${CHECKPOINTS:-}" ]]; then
    echo "WARNING: ${MODEL_NAME}: no primary/destroyed checkpoint found under ${RUN_DIR}"
    continue
  fi
  if [[ -n "${PRIMARY_WARNING:-}" ]]; then
    echo "WARNING: ${MODEL_NAME}: ${PRIMARY_WARNING}"
  fi
  if [[ -n "${DESTROYED_WARNING:-}" ]]; then
    echo "WARNING: ${MODEL_NAME}: ${DESTROYED_WARNING}"
  fi

  echo
  echo "============================================================"
  echo "Model: ${MODEL_NAME}"
  echo "Run dir: ${RUN_DIR}"
  echo "Checkpoints: ${CHECKPOINTS}"
  echo "Model args: ${ARGS}"
  echo "Key config:"
  echo "  $(echo "${ARGS}" | grep -oE -- '--mddm_variant [^ ]+|--dg_large_beta [^ ]+|--use_hdc|--hdc_mode [^ ]+|--hdc_alpha [^ ]+|--use_damage_exist_head' | tr '\n' ' ')"
  echo "============================================================"

  read -r -a MODEL_ARGS <<< "${ARGS}"
  read -r -a CKPT_NAMES <<< "${CHECKPOINTS}"

  for CKPT_NAME in "${CKPT_NAMES[@]}"; do
    CKPT_PATH="${RUN_DIR}/${CKPT_NAME}"
    if [[ ! -f "${CKPT_PATH}" ]]; then
      echo "WARNING: missing checkpoint: ${CKPT_PATH}"
      continue
    fi

    CKPT_TAG="${CKPT_NAME%.pth}"
    SAVE_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${CKPT_TAG}"
    if [[ "${FORCE}" != "1" && -f "${SAVE_DIR}/summary_official.json" ]]; then
      echo "SKIP existing result: ${SAVE_DIR}"
      continue
    fi
    if [[ -d "${SAVE_DIR}" && "${FORCE}" != "1" ]]; then
      SAVE_DIR="${SAVE_DIR}_$(date +%Y%m%d_%H%M%S)"
      echo "Existing incomplete directory detected; writing to: ${SAVE_DIR}"
    fi
    mkdir -p "${SAVE_DIR}"

    echo "RUN ${MODEL_NAME}/${CKPT_TAG}"
    "${PYTHON_BIN}" full_evaluate_unified.py \
      --eval_protocol official \
      --model_type "${MODEL_TYPE}" \
      --model_path "${CKPT_PATH}" \
      --dataset_path "${DATASET_PATH}" \
      --data_list_path "${DATA_LIST_PATH}" \
      --image_size "${IMAGE_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --save_dir "${SAVE_DIR}" \
      "${MODEL_ARGS[@]}" \
      2>&1 | tee "${SAVE_DIR}/eval.log"

    STATUS_CODE=${PIPESTATUS[0]}
    if [[ ${STATUS_CODE} -ne 0 ]]; then
      echo "WARNING: evaluation failed for ${MODEL_NAME}/${CKPT_TAG} with code ${STATUS_CODE}"
    fi
  done
done

echo
echo "Evaluation loop finished."
echo "Generate cross-model tables with:"
echo "${PYTHON_BIN} scripts/eval/summarize_eventwise_official_results.py --root \"${OUTPUT_ROOT}\" --output_dir \"${PROJECT_ROOT}/eval_analysis\""
