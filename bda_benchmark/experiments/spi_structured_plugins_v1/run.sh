#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|train-strip) ;;
  *) echo "Choose preflight|smoke|train-strip" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_structured_plugins_v1"
PLUGIN_DIR="${BDA_ROOT}/model/structured_plugins"

# A frequent deployment mistake is copying only this experiment directory.
# Fail with the complete missing-file list before Python emits an opaque import
# error.  Keep this list aligned with SERVER_SYNC_FILES.txt.
REQUIRED_CODE=(
  "${PLUGIN_DIR}/__init__.py"
  "${PLUGIN_DIR}/blocks.py"
  "${PLUGIN_DIR}/rgid_head.py"
  "${PLUGIN_DIR}/test_blocks.py"
  "${SCRIPT_DIR}/train_strip_rgid.py"
  "${BDA_ROOT}/experiments/spi_selective_rgid_e0_v1/runtime.py"
)
missing_code=()
for path in "${REQUIRED_CODE[@]}"; do
  [[ -s "${path}" ]] || missing_code+=("${path}")
done
if (( ${#missing_code[@]} > 0 )); then
  echo "Structured plug-ins installation is incomplete." >&2
  echo "Missing server files:" >&2
  printf '  - %s\n' "${missing_code[@]}" >&2
  echo "Copy every path listed in ${SCRIPT_DIR}/SERVER_SYNC_FILES.txt, preserving directories." >&2
  exit 1
fi

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile \
    "${PLUGIN_DIR}/blocks.py" \
    "${PLUGIN_DIR}/rgid_head.py" \
    "${SCRIPT_DIR}/train_strip_rgid.py"
  "${PYTHON_BIN}" -m pytest -q "${PLUGIN_DIR}/test_blocks.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "Structured plug-ins V1 preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
STEPS="${STEPS:-500}"
if [[ "${MODE}" == "smoke" ]]; then STEPS="${SMOKE_STEPS:-2}"; fi
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/BRIGHT-checkpoints/SPI_STRUCTURED_PLUGINS_V1/strip_rgid_step${STEPS}_seed${SEED}}"

for path in "${DATASET_ROOT}" "${P3_CKPT}" "${SPLIT_ROOT}/meta_train.txt"; do
  test -e "${path}" || { echo "Missing input: ${path}" >&2; exit 1; }
done

"${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_structured_plugins_v1.train_strip_rgid \
  --checkpoint "${P3_CKPT}" \
  --dataset-path "${DATASET_ROOT}" \
  --train-list "${SPLIT_ROOT}/meta_train.txt" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE:-cuda:0}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --accumulation "${ACCUMULATION_STEPS:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --calibration-samples "${CALIBRATION_SAMPLES:-256}" \
  --strip-kernel "${STRIP_KERNEL:-19}" \
  --seed "${SEED}"

echo "Strip-RGID training complete. Meta-calibration/confirm/Val450/test were not accessed."
