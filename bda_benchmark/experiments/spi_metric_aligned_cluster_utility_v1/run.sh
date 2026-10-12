#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|audit|train|auto|confirm) ;;
  *) echo "Choose preflight|audit|train|auto|confirm" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_metric_aligned_cluster_utility_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python
echo "MACU Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/core.py" \
    "${SCRIPT_DIR}/build_targets.py" "${SCRIPT_DIR}/audit_oracle.py" \
    "${SCRIPT_DIR}/train_verifier.py" "${SCRIPT_DIR}/build_confirm_cache.py" \
    "${SCRIPT_DIR}/evaluate_confirm.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "MACU preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
D1_CACHE_ROOT="${D1_CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed${SEED}}"
D1_RESULT="${D1_RESULT:-/root/data/BRIGHT-eval-results/SPI_CLUSTER_DETAIL_D1_V1/cluster_detail_d1_e0_seed${SEED}.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_METRIC_ALIGNED_CLUSTER_UTILITY_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_METRIC_ALIGNED_CLUSTER_UTILITY_V1}"
TRAIN_CACHE="${D1_CACHE_ROOT}/meta_train.pt"
CAL_CACHE="${D1_CACHE_ROOT}/meta_calibration.pt"
TRAIN_TARGETS="${OUTPUT_ROOT}/targets/meta_train_targets.pt"
CAL_TARGETS="${OUTPUT_ROOT}/targets/meta_calibration_targets.pt"
TARGET_MANIFEST="${OUTPUT_ROOT}/targets/manifest.json"
ORACLE_JSON="${RESULT_ROOT}/macu_m0_seed${SEED}.json"
MODEL_TYPE="${MODEL_TYPE:-linear}"
VERIFIER_DIR="${OUTPUT_ROOT}/${MODEL_TYPE}_step${VERIFIER_STEPS:-1000}"
SCREEN_JSON="${VERIFIER_DIR}/calibration_screen.json"
VERIFIER_CKPT="${VERIFIER_DIR}/verifier_frozen.pth"
CONFIRM_CACHE="${D1_CACHE_ROOT}/meta_confirm.pt"
CONFIRM_JSON="${RESULT_ROOT}/macu_e1_confirm_seed${SEED}.json"

for path in "${TRAIN_CACHE}" "${CAL_CACHE}" "${D1_RESULT}"; do
  test -s "${path}" || { echo "Missing input: ${path}" >&2; exit 1; }
done
"${PYTHON_BIN}" -c "import json; p=json.load(open('${D1_RESULT}')); assert p.get('decision') == 'STOP_CLUSTER_DETAIL_D1_NO_HIGH_RESOLUTION_SIGNAL'; assert not p.get('meta_confirm_accessed'); assert not p.get('val450_accessed'); print('D1 negative-control prerequisite passed')"
mkdir -p "${OUTPUT_ROOT}/targets" "${RESULT_ROOT}"

run_audit() {
  if [[ ! -s "${TARGET_MANIFEST}" ]]; then
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.build_targets \
      --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" \
      --output-dir "${OUTPUT_ROOT}/targets" --epsilon-pp "${UTILITY_EPSILON_PP:-0.0}"
  fi
  if [[ ! -s "${ORACLE_JSON}" ]]; then
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.audit_oracle \
      --calibration-cache "${CAL_CACHE}" --calibration-targets "${CAL_TARGETS}" \
      --target-manifest "${TARGET_MANIFEST}" --output "${ORACLE_JSON}" \
      --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed "${SEED}"
  fi
}

run_train() {
  test -s "${ORACLE_JSON}" || { echo "Run audit first: missing ${ORACLE_JSON}" >&2; exit 1; }
  "${PYTHON_BIN}" -c "import json; p=json.load(open('${ORACLE_JSON}')); assert p.get('decision') == 'ADMIT_METRIC_ALIGNED_LINEAR_VERIFIER_E1', p.get('decision'); print('MACU-M0 admission passed')"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.train_verifier \
    --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" \
    --train-targets "${TRAIN_TARGETS}" --calibration-targets "${CAL_TARGETS}" \
    --oracle-json "${ORACLE_JSON}" --output-dir "${VERIFIER_DIR}" \
    --groups "${FEATURE_GROUPS:-spatial,stage2_role,rgb_edge,sar_texture,structure_relation}" \
    --model-type "${MODEL_TYPE}" --hidden "${VERIFIER_HIDDEN:-32}" \
    --steps "${VERIFIER_STEPS:-1000}" --batch-size "${VERIFIER_BATCH_SIZE:-2048}" \
    --learning-rate "${LEARNING_RATE:-1e-3}" --epsilon-pp "${UTILITY_EPSILON_PP:-0.0}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed "${SEED}"
}

case "${MODE}" in
  audit) run_audit ;;
  train) run_train ;;
  auto)
    run_audit
    decision="$(${PYTHON_BIN} -c "import json; print(json.load(open('${ORACLE_JSON}'))['decision'])")"
    if [[ "${decision}" == "ADMIT_METRIC_ALIGNED_LINEAR_VERIFIER_E1" ]]; then
      run_train
    else
      echo "Automatic stop after MACU-M0: ${decision}. Verifier was not trained."
    fi
    ;;
  confirm)
    DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
    P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
    SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
    SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
    for path in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" \
      "${SPLIT_ROOT}/split_manifest.json" "${SPLIT_ROOT}/meta_confirm.txt" \
      "${TRAIN_CACHE}" "${TRAIN_TARGETS}" "${SCREEN_JSON}" "${VERIFIER_CKPT}"; do
      test -e "${path}" || { echo "Missing confirm input: ${path}" >&2; exit 1; }
    done
    "${PYTHON_BIN}" -c "import json; p=json.load(open('${SCREEN_JSON}')); assert p.get('decision') == 'FREEZE_MACU_E1_FOR_META_CONFIRM', p.get('decision'); assert abs(float(p['selected']['threshold']) - 0.4699999988079071) < 1e-8; print('Frozen MACU verifier and threshold=0.47 prerequisite passed')"
    test ! -e "${CONFIRM_JSON}" || { echo "Refusing repeated meta_confirm evaluation: ${CONFIRM_JSON}" >&2; exit 1; }
    CONFIRM_LOCK="${CONFIRM_JSON}.lock"
    mkdir "${CONFIRM_LOCK}" 2>/dev/null || { echo "Another meta_confirm run holds: ${CONFIRM_LOCK}" >&2; exit 1; }
    trap 'rmdir "${CONFIRM_LOCK}" 2>/dev/null || true' EXIT
    CONFIRM_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_confirm'])")"
    if [[ ! -s "${CONFIRM_CACHE}" ]]; then
      "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.build_confirm_cache \
        --train-cache "${TRAIN_CACHE}" --checkpoint "${P3_CKPT}" \
        --dataset-path "${DATASET_ROOT}" --confirm-list "${SPLIT_ROOT}/meta_confirm.txt" \
        --sar-stats-json "${SAR_STATS_JSON}" --output "${CONFIRM_CACHE}" \
        --expected-images "${CONFIRM_COUNT}" --device "${DEVICE:-cuda:0}" \
        --num-workers "${NUM_WORKERS:-1}" --progress-every "${PROGRESS_EVERY:-10}" \
        --checkpoint-every "${STATE_EVERY:-50}"
    fi
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.evaluate_confirm \
      --train-cache "${TRAIN_CACHE}" --confirm-cache "${CONFIRM_CACHE}" \
      --train-targets "${TRAIN_TARGETS}" --verifier-checkpoint "${VERIFIER_CKPT}" \
      --calibration-screen "${SCREEN_JSON}" --output "${CONFIRM_JSON}" \
      --minimum-gain-pp 0.30 --minimum-oracle-recovery 0.25 \
      --minimum-nonnegative-events 10 \
      --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed "${SEED}"
    echo "One-time MACU meta_confirm complete. Val450/official instances/test were not accessed."
    ;;
esac
