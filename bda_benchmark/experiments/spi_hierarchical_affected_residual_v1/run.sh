#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BENCHMARK_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${BENCHMARK_ROOT}:${PYTHONPATH:-}"

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-${BENCHMARK_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
OUTCOME_JSON="${OUTCOME_JSON:-/root/data/BRIGHT-eval-results/SPI_CLUSTER_OUTCOME_UTILITY_E21_V1/cluster_outcome_e21_loeo_seed${SEED}.json}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_HIERARCHICAL_AFFECTED_RESIDUAL_V1}"
CACHE_ROOT="${CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_HIERARCHICAL_AFFECTED_RESIDUAL_V1/seed${SEED}}"
ORACLE_JSON="${RESULT_ROOT}/ao_e0_reachable_oracle_seed${SEED}.json"
DETAIL_JSON="${RESULT_ROOT}/ao_d0_detail_seed${SEED}.json"
TRAIN_CACHE="${CACHE_ROOT}/ao_e1_meta_train.pt"
CALIBRATION_CACHE="${CACHE_ROOT}/ao_e1_meta_calibration.pt"
EVIDENCE_JSON="${RESULT_ROOT}/ao_e1_evidence_seed${SEED}.json"

echo "Hierarchical AO Python: ${PYTHON_BIN}"

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/model.py" \
    "${SCRIPT_DIR}/audit_oracle.py" "${SCRIPT_DIR}/audit_detail.py" \
    "${SCRIPT_DIR}/build_evidence_cache.py" "${SCRIPT_DIR}/audit_evidence.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  echo "Hierarchical AO preflight passed"
}

oracle_prerequisite() {
  test -s "${ORACLE_JSON}" || { echo "Run AO-E0 Oracle first: missing ${ORACLE_JSON}" >&2; exit 1; }
  "${PYTHON_BIN}" -c "import json; p=json.load(open('${ORACLE_JSON}')); assert p.get('decision') == 'ADMIT_AO_E1_EVIDENCE_AUDIT', p.get('decision'); print('AO-E0 Oracle prerequisite passed')"
}

prerequisites() {
  for path in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" \
    "${SPLIT_ROOT}/split_manifest.json" "${OUTCOME_JSON}"; do
    test -e "${path}" || { echo "Missing input: ${path}" >&2; exit 1; }
  done
  "${PYTHON_BIN}" -c "import json; p=json.load(open('${OUTCOME_JSON}')); assert p.get('decision') == 'STOP_CLUSTER_OUTCOME_E21_AFTER_LOEO', p.get('decision'); print('Cluster Outcome stop prerequisite passed')"
}

count_for() {
  "${PYTHON_BIN}" -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['$1'])"
}

run_oracle() {
  prerequisites
  mkdir -p "${RESULT_ROOT}"
  local split="${1:-meta_calibration}"
  local maximum="${2:-0}"
  local output="${3:-${ORACLE_JSON}}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_hierarchical_affected_residual_v1.audit_oracle \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${SPLIT_ROOT}/${split}.txt" --sar-stats-json "${SAR_STATS_JSON}" \
    --output "${output}" --split "${split}" --expected-images "$(count_for "${split}")" \
    --max-images "${maximum}" --device "${DEVICE:-cuda:0}" \
    --num-workers "${NUM_WORKERS:-1}" --minimum-building-area "${MINIMUM_BUILDING_AREA:-32}" \
    --minimum-gain-pp "${MINIMUM_GAIN_PP:-0.30}" \
    --minimum-damaged-f1-gain-pp "${MINIMUM_DAMAGED_F1_GAIN_PP:-0.50}" \
    --minimum-nonnegative-events "${MINIMUM_NONNEGATIVE_EVENTS:-10}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
    --progress-every "${PROGRESS_EVERY:-10}" --seed "${SEED}"
}

run_detail() {
  prerequisites
  mkdir -p "${RESULT_ROOT}"
  local split="${1:-meta_calibration}"
  local maximum="${2:-0}"
  local oracle_json="${3:-${ORACLE_JSON}}"
  local output="${4:-${DETAIL_JSON}}"
  if test "${maximum}" -le 0; then
    oracle_prerequisite
  else
    test -s "${oracle_json}" || { echo "Missing smoke Oracle: ${oracle_json}" >&2; exit 1; }
  fi
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_hierarchical_affected_residual_v1.audit_detail \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${SPLIT_ROOT}/${split}.txt" --sar-stats-json "${SAR_STATS_JSON}" \
    --oracle-json "${oracle_json}" --output "${output}" \
    --expected-images "$(count_for "${split}")" --max-images "${maximum}" \
    --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-1}" \
    --minimum-building-area "${MINIMUM_BUILDING_AREA:-32}" \
    --core-radius "${CORE_RADIUS:-2}" --progress-every "${PROGRESS_EVERY:-10}"
}

build_evidence() {
  local split="$1"
  local maximum="${2:-0}"
  local detail_json="${3:-${DETAIL_JSON}}"
  local output="$4"
  mkdir -p "${CACHE_ROOT}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_hierarchical_affected_residual_v1.build_evidence_cache \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${SPLIT_ROOT}/${split}.txt" --sar-stats-json "${SAR_STATS_JSON}" \
    --detail-json "${detail_json}" --output "${output}" --split "${split}" \
    --expected-images "$(count_for "${split}")" --max-images "${maximum}" \
    --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-1}" \
    --minimum-building-area "${MINIMUM_BUILDING_AREA:-32}" \
    --feature-channels "${FEATURE_CHANNELS:-32}" \
    --maximum-candidates-per-image "${MAXIMUM_CANDIDATES_PER_IMAGE:-2048}" \
    --progress-every "${PROGRESS_EVERY:-10}" --checkpoint-every "${STATE_EVERY:-50}" \
    --seed "${SEED}"
}

run_evidence_audit() {
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_hierarchical_affected_residual_v1.audit_evidence \
    --train-cache "${TRAIN_CACHE}" --calibration-cache "${CALIBRATION_CACHE}" \
    --detail-json "${DETAIL_JSON}" --output "${EVIDENCE_JSON}" \
    --minimum-pixel-auroc "${MINIMUM_PIXEL_AUROC:-0.65}" \
    --minimum-instance-auroc "${MINIMUM_INSTANCE_AUROC:-0.60}" \
    --minimum-impact-auroc "${MINIMUM_IMPACT_AUROC:-0.65}" \
    --minimum-event-macro "${MINIMUM_EVENT_MACRO:-0.60}" \
    --minimum-correct-events "${MINIMUM_CORRECT_EVENTS:-10}" \
    --minimum-control-gain "${MINIMUM_CONTROL_GAIN:-0.02}" \
    --permutations "${PERMUTATIONS:-1000}" --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke)
    preflight
    smoke_oracle="${RESULT_ROOT}/ao_e0_smoke_seed${SEED}.json"
    smoke_detail="${RESULT_ROOT}/ao_d0_smoke_seed${SEED}.json"
    smoke_cache="${CACHE_ROOT}/ao_e1_smoke.pt"
    run_oracle meta_train "${SMOKE_IMAGES:-20}" "${smoke_oracle}"
    ORACLE_JSON="${smoke_oracle}" run_detail meta_train "${SMOKE_IMAGES:-20}" "${smoke_oracle}" "${smoke_detail}"
    build_evidence meta_train "${SMOKE_IMAGES:-20}" "${smoke_detail}" "${smoke_cache}"
    echo "AO pipeline smoke passed. Meta-calibration/confirm/Val450/test were not accessed."
    ;;
  oracle)
    preflight
    run_oracle meta_calibration 0 "${ORACLE_JSON}"
    ;;
  detail)
    preflight
    run_detail meta_calibration 0 "${ORACLE_JSON}" "${DETAIL_JSON}"
    ;;
  evidence)
    preflight
    prerequisites
    test -s "${DETAIL_JSON}" || { echo "Run detail first: missing ${DETAIL_JSON}" >&2; exit 1; }
    build_evidence meta_train 0 "${DETAIL_JSON}" "${TRAIN_CACHE}"
    build_evidence meta_calibration 0 "${DETAIL_JSON}" "${CALIBRATION_CACHE}"
    run_evidence_audit
    ;;
  auto)
    preflight
    test -s "${ORACLE_JSON}" || run_oracle meta_calibration 0 "${ORACLE_JSON}"
    test -s "${DETAIL_JSON}" || run_detail meta_calibration 0 "${ORACLE_JSON}" "${DETAIL_JSON}"
    test -s "${TRAIN_CACHE}" || build_evidence meta_train 0 "${DETAIL_JSON}" "${TRAIN_CACHE}"
    test -s "${CALIBRATION_CACHE}" || build_evidence meta_calibration 0 "${DETAIL_JSON}" "${CALIBRATION_CACHE}"
    run_evidence_audit
    ;;
  *) echo "usage: $0 {preflight|smoke|oracle|detail|evidence|auto}"; exit 2 ;;
esac
