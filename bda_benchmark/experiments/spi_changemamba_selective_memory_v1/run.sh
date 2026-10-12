#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-k3_structural}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BDA_ROOT="$(cd "${HERE}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BDA_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${BDA_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
P3_CKPT="${P3_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
CM_ROOT="${CM_ROOT:-/root/shared-nvme/BRIGHT-master/ChangeMamba-master}"
CM_CFG="${CM_CFG:-${CM_ROOT}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml}"
CM_CKPT="${CM_CKPT:-${CM_ROOT}/results/checkpoints_B0_full/BRIGHT/ChangeMamba-MMBDA_B0_cat_1783259203.4570115/best_model.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_CHANGEMAMBA_SELECTIVE_MEMORY_V1}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_CHANGEMAMBA_SELECTIVE_MEMORY_V1}"
TRAIN_RECORDS="${TRAIN_RECORDS:-${OUTPUT_ROOT}/train_records.pt}"
VAL_RECORDS="${VAL_RECORDS:-${OUTPUT_ROOT}/val450_records.pt}"
SEED="${SEED:-3407}"
STEPS="${STEPS:-5000}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${CM_ROOT}:${PYTHONPATH:-}"

check_variant() {
  case "$1" in k1_local|k2_multiscale|k3_structural) ;; *) echo "variant must be k1_local|k2_multiscale|k3_structural" >&2; exit 2;; esac
}
metric_for() { echo "${OUTPUT_ROOT}/${1}_key_step${STEPS}_seed${SEED}.pth"; }
memory_for() { echo "${OUTPUT_ROOT}/${1}_memory_step${STEPS}_seed${SEED}.pt"; }
audit_for() { echo "${RESULT_ROOT}/${1}_audit_step${STEPS}.json"; }
utility_for() { echo "${OUTPUT_ROOT}/${1}_${2}_utility_step${STEPS}.pt"; }
verifier_for() { echo "${OUTPUT_ROOT}/${1}_${2}_seed${SEED}.pth"; }
val_for() { echo "${RESULT_ROOT}/${1}_${2}_val450.json"; }

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    "${HERE}/model.py" "${HERE}/core.py" "${HERE}/build_records.py" \
    "${HERE}/train_key.py" "${HERE}/build_memory.py" "${HERE}/audit_retrieval.py" \
    "${HERE}/build_utility.py" "${HERE}/verifier_core.py" "${HERE}/train_verifier.py" \
    "${HERE}/evaluate_verifier.py" "${HERE}/summarize.py"
  "${PYTHON_BIN}" -m pytest "${HERE}/test_core.py" -q
  test -s "${P3_CKPT}" || { echo "Missing P3 checkpoint: ${P3_CKPT}" >&2; exit 3; }
  test -d "${CM_ROOT}/changedetection" || { echo "Missing ChangeMamba root: ${CM_ROOT}" >&2; exit 3; }
  test -s "${CM_CFG}" || { echo "Missing ChangeMamba config: ${CM_CFG}" >&2; exit 3; }
  test -s "${CM_CKPT}" || { echo "Missing ChangeMamba checkpoint: ${CM_CKPT}" >&2; exit 3; }
  case "$(basename "${TRAIN_LIST}") $(basename "${VAL_LIST}")" in *test*) echo "Test access forbidden" >&2; exit 4;; esac
  echo "ChangeMamba Selective Memory V1 preflight passed. Test access: FORBIDDEN."
}

build_records() {
  local split="$1" list="$2" output="$3" expected="$4" expected_miou="$5"
  if [[ -s "${output}" && "${FORCE:-0}" != "1" ]]; then echo "Records exist; skipped: ${output}"; return; fi
  mkdir -p "$(dirname "${output}")"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_changemamba_selective_memory_v1.build_records \
    --dataset_path "${DATASET_PATH}" --data_list "${list}" --split "${split}" \
    --p3_ckpt "${P3_CKPT}" --changemamba_root "${CM_ROOT}" \
    --changemamba_ckpt "${CM_CKPT}" --cfg "${CM_CFG}" --sar_stats_json "${SAR_STATS_JSON}" \
    --output "${output}" --region_mode "${REGION_MODE:-slico}" \
    --segments "${SEGMENTS:-400}" --compactness "${COMPACTNESS:-10}" \
    --minimum_area "${MIN_AREA:-4}" --minimum_building "${MIN_BUILDING:-0.50}" \
    --batch_size "${BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-6}" \
    --expected_images "${expected}" --expected_p3_miou "${expected_miou}" \
    --p3_tolerance_pp "${P3_TOLERANCE_PP:-0.02}" --device "${DEVICE:-cuda:0}"
}

train_key() {
  local variant="$1" metric; check_variant "${variant}"; metric="$(metric_for "${variant}")"
  test -s "${TRAIN_RECORDS}" || { echo "Run build_train first" >&2; exit 3; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_changemamba_selective_memory_v1.train_key \
    --records "${TRAIN_RECORDS}" --variant "${variant}" --output "${metric}" \
    --steps "${STEPS}" --batch_size "${KEY_BATCH_SIZE:-96}" --seed "${SEED}" --device "${DEVICE:-cuda:0}"
}

build_memory() {
  local variant="$1"; check_variant "${variant}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_changemamba_selective_memory_v1.build_memory \
    --records "${TRAIN_RECORDS}" --metric "$(metric_for "${variant}")" \
    --output "$(memory_for "${variant}")" --per_class_event "${PER_CLASS_EVENT:-128}" \
    --hard_fraction "${HARD_FRACTION:-0.30}" --seed "${SEED}" --device "${DEVICE:-cuda:0}"
}

audit_variant() {
  local variant="$1"; check_variant "${variant}"; mkdir -p "${RESULT_ROOT}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_changemamba_selective_memory_v1.audit_retrieval \
    --records "${VAL_RECORDS}" --metric "$(metric_for "${variant}")" \
    --memory "$(memory_for "${variant}")" --output_json "$(audit_for "${variant}")" \
    --topk "${TOPK:-8}" --max_per_event "${MAX_PER_EVENT:-2}" \
    --exclude_same_image --exclude_same_event --device "${DEVICE:-cuda:0}"
}

build_utility() {
  local variant="$1" split="$2" records output; check_variant "${variant}"
  if [[ "${split}" == train ]]; then records="${TRAIN_RECORDS}"; else records="${VAL_RECORDS}"; fi
  output="$(utility_for "${variant}" "${split}")"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_changemamba_selective_memory_v1.build_utility \
    --records "${records}" --metric "$(metric_for "${variant}")" \
    --memory "$(memory_for "${variant}")" --output "${output}" \
    --topk "${TOPK:-8}" --max_per_event "${MAX_PER_EVENT:-2}" --device "${DEVICE:-cuda:0}" \
    --min_p3_entropy "${MIN_P3_ENTROPY:-0.30}" \
    --max_abs_p3_occurrence_margin "${MAX_ABS_P3_MARGIN:-0.60}"
}

train_and_eval_verifier() {
  local token_variant="$1" verifier_variant="$2" checkpoint result
  checkpoint="$(verifier_for "${token_variant}" "${verifier_variant}")"
  result="$(val_for "${token_variant}" "${verifier_variant}")"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_changemamba_selective_memory_v1.train_verifier \
    --utility "$(utility_for "${token_variant}" train)" --variant "${verifier_variant}" \
    --output "${checkpoint}" --steps "${VERIFIER_STEPS:-1500}" --folds "${FOLDS:-5}" \
    --maximum_acceptance "${MAX_ACCEPTANCE:-0.70}" --seed "${SEED}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_changemamba_selective_memory_v1.evaluate_verifier \
    --utility "$(utility_for "${token_variant}" validation)" --checkpoint "${checkpoint}" \
    --output_json "${result}"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke)
    preflight
    mkdir -p "${OUTPUT_ROOT}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_changemamba_selective_memory_v1.build_records \
      --dataset_path "${DATASET_PATH}" --data_list "${VAL_LIST}" --split validation \
      --p3_ckpt "${P3_CKPT}" --changemamba_root "${CM_ROOT}" \
      --changemamba_ckpt "${CM_CKPT}" --cfg "${CM_CFG}" --sar_stats_json "${SAR_STATS_JSON}" \
      --output "${OUTPUT_ROOT}/smoke2_records.pt" --max_images 2 --expected_images 2 \
      --batch_size 1 --num_workers 0 --device "${DEVICE:-cuda:0}"
    ;;
  build_train) preflight; build_records train "${TRAIN_LIST}" "${TRAIN_RECORDS}" "${EXPECTED_TRAIN_IMAGES:-2918}" -1 ;;
  build_val) preflight; build_records validation "${VAL_LIST}" "${VAL_RECORDS}" "${EXPECTED_VAL_IMAGES:-450}" "${EXPECTED_P3_MIOU:-0.6050195963527841}" ;;
  train) preflight; train_key "${VARIANT}" ;;
  memory) preflight; build_memory "${VARIANT}" ;;
  audit) preflight; audit_variant "${VARIANT}" ;;
  stage_token)
    preflight
    build_records train "${TRAIN_LIST}" "${TRAIN_RECORDS}" "${EXPECTED_TRAIN_IMAGES:-2918}" -1
    build_records validation "${VAL_LIST}" "${VAL_RECORDS}" "${EXPECTED_VAL_IMAGES:-450}" "${EXPECTED_P3_MIOU:-0.6050195963527841}"
    train_key "${VARIANT}"; build_memory "${VARIANT}"; audit_variant "${VARIANT}"
    ;;
  stage_all_tokens)
    preflight
    build_records train "${TRAIN_LIST}" "${TRAIN_RECORDS}" "${EXPECTED_TRAIN_IMAGES:-2918}" -1
    build_records validation "${VAL_LIST}" "${VAL_RECORDS}" "${EXPECTED_VAL_IMAGES:-450}" "${EXPECTED_P3_MIOU:-0.6050195963527841}"
    for v in k1_local k2_multiscale k3_structural; do train_key "$v"; build_memory "$v"; audit_variant "$v"; done
    ;;
  stage_safe_k3)
    preflight
    build_utility k3_structural train; build_utility k3_structural validation
    train_and_eval_verifier k3_structural v1_direct
    train_and_eval_verifier k3_structural v2_memory
    mkdir -p "${RESULT_ROOT}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_changemamba_selective_memory_v1.summarize \
      --k1 "$(audit_for k1_local)" --k2 "$(audit_for k2_multiscale)" --k3 "$(audit_for k3_structural)" \
      --v1 "$(val_for k3_structural v1_direct)" --v2 "$(val_for k3_structural v2_memory)" \
      --output "${RESULT_ROOT}/final_admission.json"
    ;;
  summarize)
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_changemamba_selective_memory_v1.summarize \
      --k1 "$(audit_for k1_local)" --k2 "$(audit_for k2_multiscale)" --k3 "$(audit_for k3_structural)" \
      --output "${RESULT_ROOT}/token_admission.json"
    ;;
  *) echo "Choose preflight|smoke|build_train|build_val|train <variant>|memory <variant>|audit <variant>|stage_token <variant>|stage_all_tokens|stage_safe_k3|summarize" >&2; exit 2;;
esac
