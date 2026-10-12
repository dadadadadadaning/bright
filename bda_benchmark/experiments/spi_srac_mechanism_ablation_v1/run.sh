#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
SOURCE="${2:-c4_structural}"
POOL="${3:-p3_attn_tail}"
RAG_VARIANT="${4:-x1_label_geometry}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BDA_ROOT="$(cd "${HERE}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BDA_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${BDA_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
P3_CKPT="${P3_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
CM_ROOT="${CM_ROOT:-/root/data/BRIGHT-master/ChangeMamba-master}"
CM_CFG="${CM_CFG:-${CM_ROOT}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml}"
B2_CKPT="${B2_CKPT:-${CM_ROOT}/results/checkpoints_B2_DiffFusion/B2_ChangeMamba_BRIGHT_DiffFusion_seed3407_C512_I80000_BS2/ChangeMamba-MMBDA_B2_DiffFusion_1783263533.424174/best_model.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_SRAC_MECHANISM_ABLATION_V1}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_SRAC_MECHANISM_ABLATION_V1}"
TRAIN_RECORDS="${TRAIN_RECORDS:-${OUTPUT_ROOT}/train_records.pt}"
VAL_RECORDS="${VAL_RECORDS:-${OUTPUT_ROOT}/val450_records.pt}"
SEED="${SEED:-3407}"
STEPS="${STEPS:-5000}"
TAG="${SOURCE}__${POOL}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${CM_ROOT}:${PYTHONPATH:-}"

check_source() { case "$1" in c0_base|c1_diff|c2_enhanced|c3_dual|c4_structural) ;; *) echo "bad source variant: $1" >&2; exit 2;; esac; }
check_pool() { case "$1" in p0_mean|p1_stats|p2_attn_skip|p3_attn_tail) ;; *) echo "bad pool variant: $1" >&2; exit 2;; esac; }
check_rag() { case "$1" in x0_keys|x1_label_geometry|x2_label_geometry_event) ;; *) echo "bad RAG variant: $1" >&2; exit 2;; esac; }

encoder_for() { echo "${OUTPUT_ROOT}/${1}_encoder_step${STEPS}_seed${SEED}.pth"; }
memory_for() { echo "${OUTPUT_ROOT}/${1}_memory_step${STEPS}_seed${SEED}.pt"; }
train_neighbours_for() { echo "${OUTPUT_ROOT}/${1}_train_neighbours_step${STEPS}.pt"; }
val_neighbours_for() { echo "${OUTPUT_ROOT}/${1}_val_neighbours_step${STEPS}.pt"; }
rag_for() { echo "${OUTPUT_ROOT}/${1}_${2}_step${STEPS}_seed${SEED}.pth"; }

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    "${HERE}/b2_features.py" "${HERE}/core.py" "${HERE}/build_records.py" \
    "${HERE}/train_encoder.py" "${HERE}/build_memory.py" "${HERE}/build_neighbour_cache.py" \
    "${HERE}/train_rag.py" "${HERE}/audit_evidence.py" "${HERE}/audit.py" \
    "${HERE}/build_utility.py" "${HERE}/router_core.py" "${HERE}/train_router.py" \
    "${HERE}/evaluate_router.py" "${HERE}/summarize.py"
  "${PYTHON_BIN}" -m pytest "${HERE}/test_core.py" -q
  test -s "${P3_CKPT}" || { echo "Missing P3 checkpoint: ${P3_CKPT}" >&2; exit 3; }
  test -s "${B2_CKPT}" || { echo "Missing B2 checkpoint: ${B2_CKPT}" >&2; exit 3; }
  test -s "${CM_CFG}" || { echo "Missing ChangeMamba config: ${CM_CFG}" >&2; exit 3; }
  case "$(basename "${TRAIN_LIST}") $(basename "${VAL_LIST}")" in *test*) echo "Test access forbidden" >&2; exit 4;; esac
  echo "SRAC mechanism ablation V1 preflight passed. Test access: FORBIDDEN."
}

build_records() {
  local split="$1" list="$2" output="$3" expected="$4" expected_miou="$5"
  if [[ -s "${output}" && "${FORCE:-0}" != "1" ]]; then echo "Records exist; skipped: ${output}"; return; fi
  mkdir -p "$(dirname "${output}")"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.build_records \
    --dataset_path "${DATASET_PATH}" --data_list "${list}" --split "${split}" \
    --p3_ckpt "${P3_CKPT}" --changemamba_root "${CM_ROOT}" \
    --changemamba_ckpt "${B2_CKPT}" --cfg "${CM_CFG}" --output "${output}" \
    --region_mode "${REGION_MODE:-slico}" --segments "${SEGMENTS:-400}" \
    --compactness "${COMPACTNESS:-10}" --max_region_tokens "${MAX_REGION_TOKENS:-12}" \
    --minimum_area "${MIN_AREA:-4}" --minimum_building "${MIN_BUILDING:-0.50}" \
    --batch_size "${BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-6}" \
    --expected_images "${expected}" --expected_p3_miou "${expected_miou}" --device "${DEVICE:-cuda:0}"
}

stage_evidence() {
  check_source "${SOURCE}"; check_pool "${POOL}"; mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}"
  local encoder memory neighbours
  encoder="$(encoder_for "${TAG}")"; memory="$(memory_for "${TAG}")"; neighbours="$(val_neighbours_for "${TAG}")"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.train_encoder \
    --records "${TRAIN_RECORDS}" --output "${encoder}" --source_variant "${SOURCE}" \
    --pool_variant "${POOL}" --steps "${STEPS}" --seed "${SEED}" --device "${DEVICE:-cuda:0}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.build_memory \
    --records "${TRAIN_RECORDS}" --encoder "${encoder}" --output "${memory}" \
    --seed "${SEED}" --device "${DEVICE:-cuda:0}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.build_neighbour_cache \
    --records "${VAL_RECORDS}" --encoder "${encoder}" --memory "${memory}" --output "${neighbours}" \
    --exclude_same_event --topk "${TOPK:-8}" --max_per_event "${MAX_PER_EVENT:-2}" \
    --seed "${SEED}" --device "${DEVICE:-cuda:0}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.audit_evidence \
    --records "${VAL_RECORDS}" --cache "${neighbours}" --output_json "${RESULT_ROOT}/${TAG}_evidence.json"
}

stage_rag() {
  check_source "${SOURCE}"; check_pool "${POOL}"; check_rag "${RAG_VARIANT}"
  local encoder memory train_cache val_cache rag
  encoder="$(encoder_for "${TAG}")"; memory="$(memory_for "${TAG}")"
  train_cache="$(train_neighbours_for "${TAG}")"; val_cache="$(val_neighbours_for "${TAG}")"
  rag="$(rag_for "${TAG}" "${RAG_VARIANT}")"
  test -s "${encoder}" || { echo "Run stage_evidence first: ${encoder}" >&2; exit 5; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.build_neighbour_cache \
    --records "${TRAIN_RECORDS}" --encoder "${encoder}" --memory "${memory}" --output "${train_cache}" \
    --exclude_same_image --exclude_same_event --topk "${TOPK:-8}" --max_per_event "${MAX_PER_EVENT:-2}" --device "${DEVICE:-cuda:0}"
  if [[ ! -s "${val_cache}" ]]; then
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.build_neighbour_cache \
      --records "${VAL_RECORDS}" --encoder "${encoder}" --memory "${memory}" --output "${val_cache}" \
      --exclude_same_event --topk "${TOPK:-8}" --max_per_event "${MAX_PER_EVENT:-2}" --device "${DEVICE:-cuda:0}"
  fi
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.train_rag \
    --cache "${train_cache}" --output "${rag}" --variant "${RAG_VARIANT}" \
    --steps "${RAG_STEPS:-5000}" --seed "${SEED}" --device "${DEVICE:-cuda:0}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.audit \
    --records "${VAL_RECORDS}" --cache "${val_cache}" --rag "${rag}" \
    --output_json "${RESULT_ROOT}/${TAG}_rag_${RAG_VARIANT}.json" --device "${DEVICE:-cuda:0}"
}

stage_router() {
  local train_cache val_cache rag train_utility val_utility variant checkpoint
  train_cache="$(train_neighbours_for "${TAG}")"; val_cache="$(val_neighbours_for "${TAG}")"
  rag="$(rag_for "${TAG}" "${RAG_VARIANT}")"
  train_utility="${OUTPUT_ROOT}/${TAG}_${RAG_VARIANT}_train_utility.pt"
  val_utility="${OUTPUT_ROOT}/${TAG}_${RAG_VARIANT}_val_utility.pt"
  for split in train validation; do
    if [[ "${split}" == train ]]; then records="${TRAIN_RECORDS}"; cache="${train_cache}"; output="${train_utility}"; else records="${VAL_RECORDS}"; cache="${val_cache}"; output="${val_utility}"; fi
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.build_utility \
      --records "${records}" --cache "${cache}" --rag "${rag}" --output "${output}" --device "${DEVICE:-cuda:0}"
  done
  for variant in v1_direct v2_rag; do
    checkpoint="${OUTPUT_ROOT}/${TAG}_${RAG_VARIANT}_${variant}_seed${SEED}.pth"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.train_router \
      --utility "${train_utility}" --variant "${variant}" --output "${checkpoint}" --seed "${SEED}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.evaluate_router \
      --utility "${val_utility}" --checkpoint "${checkpoint}" \
      --output_json "${RESULT_ROOT}/${TAG}_${RAG_VARIANT}_${variant}_val450.json"
  done
}

case "${MODE}" in
  preflight) preflight ;;
  smoke)
    preflight; mkdir -p "${OUTPUT_ROOT}"
    FORCE=1 "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.build_records \
      --dataset_path "${DATASET_PATH}" --data_list "${VAL_LIST}" --split validation \
      --p3_ckpt "${P3_CKPT}" --changemamba_root "${CM_ROOT}" --changemamba_ckpt "${B2_CKPT}" \
      --cfg "${CM_CFG}" --output "${OUTPUT_ROOT}/smoke2_records.pt" --max_images 2 \
      --expected_images 2 --batch_size 1 --num_workers 0 --device "${DEVICE:-cuda:0}" ;;
  build_train) preflight; build_records train "${TRAIN_LIST}" "${TRAIN_RECORDS}" "${EXPECTED_TRAIN_IMAGES:-2918}" -1 ;;
  build_val) preflight; build_records validation "${VAL_LIST}" "${VAL_RECORDS}" "${EXPECTED_VAL_IMAGES:-450}" "${EXPECTED_P3_MIOU:-0.6050195963527841}" ;;
  evidence) preflight; stage_evidence ;;
  rag) preflight; stage_rag ;;
  router) preflight; stage_router ;;
  random_control)
    preflight; check_source "${SOURCE}"; check_pool "${POOL}"; check_rag "${RAG_VARIANT}"
    encoder="$(encoder_for "${TAG}")"; memory="$(memory_for "${TAG}")"
    random_cache="${OUTPUT_ROOT}/${TAG}_val_neighbours_random_labels.pt"
    rag="$(rag_for "${TAG}" "${RAG_VARIANT}")"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.build_neighbour_cache \
      --records "${VAL_RECORDS}" --encoder "${encoder}" --memory "${memory}" --output "${random_cache}" \
      --exclude_same_event --shuffle_labels --seed "${SEED}" --device "${DEVICE:-cuda:0}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.audit \
      --records "${VAL_RECORDS}" --cache "${random_cache}" --rag "${rag}" \
      --output_json "${RESULT_ROOT}/${TAG}_rag_${RAG_VARIANT}_random_labels.json" --device "${DEVICE:-cuda:0}" ;;
  summarize)
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.summarize \
      --result_root "${RESULT_ROOT}" --output_json "${RESULT_ROOT}/summary.json" ;;
  *) echo "usage: $0 preflight|smoke|build_train|build_val|evidence|rag|router|random_control|summarize [source] [pool] [rag]" >&2; exit 2 ;;
esac
