#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|split_audit|old450|new697|both) ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|split_audit|old450|new697|both" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
OLD_VAL_LIST="${OLD_VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
# The CVPRW26 instance challenge reuses 697 images from the original BRIGHT
# standard-ML test split as its labelled validation set.  Keep the historical
# NEW_VAL_LIST variable for compatibility, but never treat it as the semantic
# model-selection validation set.
NEW_VAL_LIST="${NEW_VAL_LIST:-${REPO_ROOT}/cvprw26/data/splits/val_set.txt}"
OLD_TRAIN_LIST="${OLD_TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
OLD_TEST_LIST="${OLD_TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
CVPRW26_TRAIN_LIST="${CVPRW26_TRAIN_LIST:-${REPO_ROOT}/cvprw26/data/splits/train_set.txt}"
CVPRW26_HOLDOUT_LIST="${CVPRW26_HOLDOUT_LIST:-${REPO_ROOT}/cvprw26/data/splits/holdout_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-${STORAGE_ROOT}/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
P3_CKPT="${P3_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
T0_CKPT="${T0_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_T1_ISOLATED/seed3407/t0_original_objective/last.pth}"
T1_CKPT="${T1_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_T1_ISOLATED/seed3407/t1_component_balanced/last.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_T1_SEMANTIC_REAUDIT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ "${MODE}" == "preflight" ]]; then
  bash -n run/run_SPI_T1_SEMANTIC_REAUDIT.sh
  python -m py_compile script/standard_ML/spi_t1/semantic_reaudit.py
  echo "SPI pure semantic re-audit preflight passed."
  exit 0
fi

if [[ "${MODE}" == "split_audit" ]]; then
  for split_file in \
    "${OLD_TRAIN_LIST}" "${OLD_VAL_LIST}" "${OLD_TEST_LIST}" \
    "${CVPRW26_TRAIN_LIST}" "${NEW_VAL_LIST}" "${CVPRW26_HOLDOUT_LIST}"
  do
    test -s "${split_file}" || { echo "Missing split list: ${split_file}" >&2; exit 1; }
  done
  SPLIT_FILES="${OLD_TRAIN_LIST}|${OLD_VAL_LIST}|${OLD_TEST_LIST}|${CVPRW26_TRAIN_LIST}|${NEW_VAL_LIST}|${CVPRW26_HOLDOUT_LIST}" \
  python - <<'PY'
import os
from pathlib import Path

labels = ("old_train", "old_val", "old_test", "cv_train", "cv_val", "cv_holdout")
paths = os.environ["SPLIT_FILES"].split("|")
sets = {}
for label, raw_path in zip(labels, paths):
    values = {line.strip() for line in Path(raw_path).read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    sets[label] = values
    print(f"{label:12s}: {len(values):4d}  {raw_path}")
print()
for old in ("old_train", "old_val", "old_test"):
    for new in ("cv_train", "cv_val", "cv_holdout"):
        print(f"{old:12s} intersect {new:12s}: {len(sets[old] & sets[new])}")
print("\nINTERPRETATION:")
if sets["cv_val"] <= sets["old_test"]:
    print("- cv_val is contained in the original standard-ML TEST split.")
    print("- Do not use cv_val to tune semantic models or select checkpoints.")
else:
    print("- cv_val is not fully contained in old_test; investigate split provenance manually.")
PY
  exit 0
fi

for required in "${DATASET_ROOT}" "${SAR_STATS_JSON}" "${P3_CKPT}" "${T0_CKPT}" "${T1_CKPT}"; do
  test -e "${required}" || { echo "Missing semantic re-audit input: ${required}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_ROOT}"

run_split() {
  local split_name="$1"
  local val_list="$2"
  local expected_images="$3"
  local expected_p3_valid_miou="${4:-}"
  local original_standard_test_accessed="${5:-false}"
  test -s "${val_list}" || { echo "Missing validation list: ${val_list}" >&2; exit 1; }
  local reproduction_args=()
  if [[ -n "${expected_p3_valid_miou}" ]]; then
    reproduction_args+=(--expected_p3_valid_miou "${expected_p3_valid_miou}")
  fi
  if [[ "${original_standard_test_accessed}" == "true" ]]; then
    reproduction_args+=(--original_standard_test_accessed)
  fi
  python -u -m bda_benchmark.script.standard_ML.spi_t1.semantic_reaudit \
    --p3_checkpoint "${P3_CKPT}" \
    --t0_checkpoint "${T0_CKPT}" \
    --t1_checkpoint "${T1_CKPT}" \
    --dataset_path "${DATASET_ROOT}" \
    --val_list "${val_list}" \
    --sar_stats_json "${SAR_STATS_JSON}" \
    --split_name "${split_name}" \
    --expected_images "${expected_images}" \
    --output_json "${OUTPUT_ROOT}/${split_name}_seed3407.json" \
    --device cuda:0 \
    --num_workers "${NUM_WORKERS:-4}" \
    --progress_every "${PROGRESS_EVERY:-20}" \
    "${reproduction_args[@]}"
}

if [[ "${MODE}" == "old450" || "${MODE}" == "both" ]]; then
  run_split standard_ml_val450 "${OLD_VAL_LIST}" 450 0.605019
fi
if [[ "${MODE}" == "new697" || "${MODE}" == "both" ]]; then
  echo "WARNING: CVPRW26 val697 belongs to the original standard-ML test split."
  echo "Use this result for frozen external diagnosis only, never for tuning or checkpoint selection."
  run_split cvprw26_instance_val697_external_only "${NEW_VAL_LIST}" 697 "" true
fi
