#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BDA_ROOT="$(cd "${HERE}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BDA_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${BDA_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
if [[ -n "${CM_ROOT:-}" ]]; then
  CM_ROOT="${CM_ROOT}"
elif [[ -d "${PROJECT_ROOT}/ChangeMamba-master/changedetection" ]]; then
  CM_ROOT="${PROJECT_ROOT}/ChangeMamba-master"
elif [[ -d "${BDA_ROOT}/experiments/changedetection" ]]; then
  # Compatibility with the user's server mirror, where the ChangeMamba
  # package was placed below bda_benchmark/experiments.
  CM_ROOT="${BDA_ROOT}/experiments"
else
  CM_ROOT="${PROJECT_ROOT}/ChangeMamba-master"
fi
CM_CFG="${CM_CFG:-${CM_ROOT}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml}"
B2_CKPT="${B2_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/chagemamba/best_model.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/CHANGEMAMBA_DEEP_REGION_RETRIEVAL_V1}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/CHANGEMAMBA_DEEP_REGION_RETRIEVAL_V1}"
SEED="${SEED:-3407}"
STEPS="${STEPS:-5000}"
RAG_STEPS="${RAG_STEPS:-5000}"
DEVICE="${DEVICE:-cuda:0}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${CM_ROOT}:${PYTHONPATH:-}"

records_path() { echo "${OUTPUT_ROOT}/${1}_${2}_records.pt"; }
tag_for() { echo "${1}__${2}__${3}"; }
encoder_path() { echo "${OUTPUT_ROOT}/$(tag_for "$1" "$2" "$3")_encoder_step${STEPS}_seed${SEED}.pth"; }
memory_path() { echo "${OUTPUT_ROOT}/$(tag_for "$1" "$2" "$3")_memory_step${STEPS}_seed${SEED}.pt"; }
val_cache_path() { echo "${OUTPUT_ROOT}/$(tag_for "$1" "$2" "$3")_val_neighbours.pt"; }
train_cache_path() { echo "${OUTPUT_ROOT}/$(tag_for "$1" "$2" "$3")_train_neighbours.pt"; }

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    "${HERE}/build_records.py" "${HERE}/check_checkpoint.py" \
    "${HERE}/make_control_cache.py" "${HERE}/summarize.py"
  "${PYTHON_BIN}" -m pytest "${HERE}/test_contract.py" -q
  test -d "${CM_ROOT}/changedetection" || { echo "Invalid CM_ROOT (missing changedetection/): ${CM_ROOT}" >&2; exit 3; }
  test -s "${CM_ROOT}/changedetection/models/ChangeMambaMMBDA_B2_DiffFusion.py" || {
    echo "Missing B2 model source below CM_ROOT: ${CM_ROOT}" >&2; exit 3;
  }
  test -s "${B2_CKPT}" || { echo "Missing B2 checkpoint: ${B2_CKPT}" >&2; exit 3; }
  test -s "${CM_CFG}" || { echo "Missing ChangeMamba config: ${CM_CFG}" >&2; exit 3; }
  "${PYTHON_BIN}" -m bda_benchmark.experiments.changemamba_deep_region_retrieval_v1.check_checkpoint \
    --checkpoint "${B2_CKPT}"
  case "$(basename "${TRAIN_LIST}") $(basename "${VAL_LIST}")" in
    *test*) echo "Test access forbidden" >&2; exit 4 ;;
  esac
  echo "ChangeMamba Deep Region Retrieval V1 preflight passed. Test access: FORBIDDEN."
}

build_one() {
  local region="$1" split="$2" list="$3" expected="$4"
  local output
  output="$(records_path "${region}" "${split}")"
  if [[ -s "${output}" && "${FORCE:-0}" != "1" ]]; then
    echo "Records exist; skipped: ${output}"
    return
  fi
  mkdir -p "${OUTPUT_ROOT}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.changemamba_deep_region_retrieval_v1.build_records \
    --dataset_path "${DATASET_PATH}" --data_list "${list}" --split "${split}" \
    --changemamba_root "${CM_ROOT}" --changemamba_ckpt "${B2_CKPT}" --cfg "${CM_CFG}" \
    --region_mode "${region}" --output "${output}" --expected_images "${expected}" \
    --segments "${SEGMENTS:-400}" --compactness "${COMPACTNESS:-10}" \
    --minimum_area "${MIN_AREA:-4}" --minimum_building "${MIN_BUILDING:-0.50}" \
    --max_region_tokens "${MAX_REGION_TOKENS:-12}" --batch_size "${BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-6}" --device "${DEVICE}"
}

evidence_one() {
  local region="$1" source="$2" pool="$3"
  local tag train_records val_records encoder memory val_cache
  tag="$(tag_for "${region}" "${source}" "${pool}")"
  train_records="$(records_path "${region}" train)"
  val_records="$(records_path "${region}" validation)"
  encoder="$(encoder_path "${region}" "${source}" "${pool}")"
  memory="$(memory_path "${region}" "${source}" "${pool}")"
  val_cache="$(val_cache_path "${region}" "${source}" "${pool}")"
  test -s "${train_records}" || { echo "Missing ${train_records}; run build_records" >&2; exit 5; }
  test -s "${val_records}" || { echo "Missing ${val_records}; run build_records" >&2; exit 5; }
  mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.train_encoder \
    --records "${train_records}" --output "${encoder}" --source_variant "${source}" \
    --pool_variant "${pool}" --steps "${STEPS}" --seed "${SEED}" --device "${DEVICE}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.build_memory \
    --records "${train_records}" --encoder "${encoder}" --output "${memory}" \
    --seed "${SEED}" --device "${DEVICE}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.build_neighbour_cache \
    --records "${val_records}" --encoder "${encoder}" --memory "${memory}" --output "${val_cache}" \
    --exclude_same_event --topk "${TOPK:-8}" --max_per_event "${MAX_PER_EVENT:-2}" \
    --seed "${SEED}" --device "${DEVICE}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.audit_evidence \
    --records "${val_records}" --cache "${val_cache}" \
    --output_json "${RESULT_ROOT}/${tag}_evidence.json"
}

rag_primary() {
  local region=slico source=c4_structural pool=p3_attn_tail tag
  local train_records val_records encoder memory train_cache val_cache rag
  tag="$(tag_for "${region}" "${source}" "${pool}")"
  train_records="$(records_path "${region}" train)"
  val_records="$(records_path "${region}" validation)"
  encoder="$(encoder_path "${region}" "${source}" "${pool}")"
  memory="$(memory_path "${region}" "${source}" "${pool}")"
  train_cache="$(train_cache_path "${region}" "${source}" "${pool}")"
  val_cache="$(val_cache_path "${region}" "${source}" "${pool}")"
  rag="${OUTPUT_ROOT}/${tag}_x1_label_geometry_step${RAG_STEPS}_seed${SEED}.pth"
  test -s "${encoder}" || { echo "Run primary evidence first" >&2; exit 6; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.build_neighbour_cache \
    --records "${train_records}" --encoder "${encoder}" --memory "${memory}" --output "${train_cache}" \
    --exclude_same_image --exclude_same_event --topk "${TOPK:-8}" \
    --max_per_event "${MAX_PER_EVENT:-2}" --seed "${SEED}" --device "${DEVICE}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.train_rag \
    --cache "${train_cache}" --output "${rag}" --variant x1_label_geometry \
    --steps "${RAG_STEPS}" --seed "${SEED}" --device "${DEVICE}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.audit \
    --records "${val_records}" --cache "${val_cache}" --rag "${rag}" \
    --output_json "${RESULT_ROOT}/${tag}_rag.json" --device "${DEVICE}"
}

controls() {
  local region=slico source=c4_structural pool=p3_attn_tail tag val_records val_cache rag control cache
  tag="$(tag_for "${region}" "${source}" "${pool}")"
  val_records="$(records_path "${region}" validation)"
  val_cache="$(val_cache_path "${region}" "${source}" "${pool}")"
  rag="${OUTPUT_ROOT}/${tag}_x1_label_geometry_step${RAG_STEPS}_seed${SEED}.pth"
  test -s "${rag}" || { echo "Run rag_primary first" >&2; exit 7; }
  for control in shuffled_labels wrong_labels random_neighbours; do
    cache="${OUTPUT_ROOT}/${tag}_val_${control}.pt"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.changemamba_deep_region_retrieval_v1.make_control_cache \
      --input "${val_cache}" --output "${cache}" --control "${control}" --seed "${SEED}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.audit \
      --records "${val_records}" --cache "${cache}" --rag "${rag}" \
      --output_json "${RESULT_ROOT}/${tag}_rag_${control}.json" --device "${DEVICE}"
  done
}

summarize() {
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.changemamba_deep_region_retrieval_v1.summarize \
    --records "$(records_path slico validation)" --result_root "${RESULT_ROOT}" \
    --output "${RESULT_ROOT}/strict_summary.json"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke)
    preflight
    FORCE=1 "${PYTHON_BIN}" -u -m bda_benchmark.experiments.changemamba_deep_region_retrieval_v1.build_records \
      --dataset_path "${DATASET_PATH}" --data_list "${VAL_LIST}" --split validation \
      --changemamba_root "${CM_ROOT}" --changemamba_ckpt "${B2_CKPT}" --cfg "${CM_CFG}" \
      --region_mode slico --output "${OUTPUT_ROOT}/smoke_records.pt" --max_images 2 \
      --expected_images 2 --batch_size 1 --num_workers 0 --device "${DEVICE}" ;;
  build_records)
    preflight
    build_one grid train "${TRAIN_LIST}" "${EXPECTED_TRAIN_IMAGES:-2918}"
    build_one grid validation "${VAL_LIST}" "${EXPECTED_VAL_IMAGES:-450}"
    build_one slico train "${TRAIN_LIST}" "${EXPECTED_TRAIN_IMAGES:-2918}"
    build_one slico validation "${VAL_LIST}" "${EXPECTED_VAL_IMAGES:-450}" ;;
  evidence_controls)
    preflight
    evidence_one grid c4_structural p3_attn_tail
    evidence_one slico c0_base p0_mean
    evidence_one slico c4_structural p0_mean
    evidence_one slico c4_structural p3_attn_tail ;;
  rag_primary) preflight; rag_primary ;;
  controls) preflight; controls ;;
  summarize) summarize ;;
  all)
    preflight
    build_one grid train "${TRAIN_LIST}" "${EXPECTED_TRAIN_IMAGES:-2918}"
    build_one grid validation "${VAL_LIST}" "${EXPECTED_VAL_IMAGES:-450}"
    build_one slico train "${TRAIN_LIST}" "${EXPECTED_TRAIN_IMAGES:-2918}"
    build_one slico validation "${VAL_LIST}" "${EXPECTED_VAL_IMAGES:-450}"
    evidence_one grid c4_structural p3_attn_tail
    evidence_one slico c0_base p0_mean
    evidence_one slico c4_structural p0_mean
    evidence_one slico c4_structural p3_attn_tail
    rag_primary
    controls
    summarize ;;
  *) echo "usage: $0 preflight|smoke|build_records|evidence_controls|rag_primary|controls|summarize|all" >&2; exit 2 ;;
esac
