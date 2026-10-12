#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-preflight}"
ARG="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
SEED="${SEED:-3407}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset}"
TRAIN_LIST="${TRAIN_LIST:-$DATASET_PATH/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-$DATASET_PATH/splitname/standard_ML/val_set.txt}"
A1_CKPT="${A1_CKPT:-$STORAGE_ROOT/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth}"
V2_VARIANT="${V2_VARIANT:-p_sc_residual}"
V2_ROOT="${V2_ROOT:-$STORAGE_ROOT/BRIGHT-checkpoints/DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2}"
V2_BASE="$V2_ROOT/${V2_VARIANT}_seed${SEED}"
STAGE1_CKPT="${STAGE1_CKPT:-$V2_BASE/stage1_change/latest.pth}"
STAGE2_CKPT="${STAGE2_CKPT:-$V2_BASE/stage2_feedback/latest.pth}"
OUT_ROOT="${CHAIN_OUT_ROOT:-$STORAGE_ROOT/BRIGHT-checkpoints/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1}"
EVAL_ROOT="${CHAIN_EVAL_ROOT:-$STORAGE_ROOT/BRIGHT-eval-results/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1}"
LOG_ROOT="${CHAIN_LOG_ROOT:-$STORAGE_ROOT/BRIGHT-train-logs/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1}"
CACHE_ROOT="$OUT_ROOT/cache/${V2_VARIANT}_seed${SEED}"
TRAIN_CACHE="$CACHE_ROOT/stage1_train.pt"
VAL_CACHE="$CACHE_ROOT/stage1_val.pt"
VAL_FB_CACHE="$CACHE_ROOT/stage2_val.pt"
TAIL_ROOT="$OUT_ROOT/B_tail"
PROTO_ROOT="$OUT_ROOT/B_multi_proto"
EXPERT_ROOT="$OUT_ROOT/C_expert"
ORACLE_ROOT="$EVAL_ROOT/D_prototype"
MEM_ROOT="$EVAL_ROOT/E_memory"
mkdir -p "$OUT_ROOT" "$EVAL_ROOT" "$LOG_ROOT" "$CACHE_ROOT" "$TAIL_ROOT" "$PROTO_ROOT" "$EXPERT_ROOT" "$ORACLE_ROOT" "$MEM_ROOT"

py() { python -u -m "bda_benchmark.experiments.damageformer_causal_research_chain_v1.$1" "${@:2}"; }

preflight() {
  python -m py_compile experiments/damageformer_causal_research_chain_v1/*.py
  python -m unittest experiments.damageformer_causal_research_chain_v1.test_core -v
  test -s "$TRAIN_LIST" && test -s "$VAL_LIST" && test -s "$A1_CKPT"
  [[ "$(basename "$TRAIN_LIST")" != *test* && "$(basename "$VAL_LIST")" != *test* ]] || { echo "TEST forbidden" >&2; exit 4; }
  echo "CAUSAL CHAIN PREFLIGHT PASSED"
}

export_cache() {
  local ckpt="$1" list="$2" split="$3" out="$4"
  test -s "$ckpt" || { echo "Missing V2 checkpoint: $ckpt" >&2; exit 5; }
  [[ -s "$out" && "${FORCE_CACHE:-0}" != "1" ]] && { echo "Reuse cache: $out"; return; }
  py export_v2_cache --checkpoint "$ckpt" --a1_ckpt "$A1_CKPT" --dataset_path "$DATASET_PATH" \
    --list_file "$list" --split "$split" --dataset_mode val --output "$out" \
    --batch_size "${CACHE_BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-4}" --device "$DEVICE"
}

ensure_stage1_cache() {
  export_cache "$STAGE1_CKPT" "$TRAIN_LIST" train "$TRAIN_CACHE"
  export_cache "$STAGE1_CKPT" "$VAL_LIST" val "$VAL_CACHE"
}

case "$MODE" in
  preflight) preflight ;;
  cache_stage1) preflight; ensure_stage1_cache ;;
  cache_stage2_val) preflight; export_cache "$STAGE2_CKPT" "$VAL_LIST" val "$VAL_FB_CACHE" ;;
  audit_region)
    ensure_stage1_cache
    py region_audit --cache "$VAL_CACHE" --output "$EVAL_ROOT/A_region_evidence.json"
    ;;
  compare_feedback)
    ensure_stage1_cache; export_cache "$STAGE2_CKPT" "$VAL_LIST" val "$VAL_FB_CACHE"
    py compare_feedback --base_cache "$VAL_CACHE" --feedback_cache "$VAL_FB_CACHE" --output "$EVAL_ROOT/A_feedback_rescue_harm.json"
    ;;
  tail_probe)
    ensure_stage1_cache
    METHOD="${ARG:-ce}"
    py train_tail_probe --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" \
      --output_dir "$TAIL_ROOT/${METHOD}_seed${SEED}" --method "$METHOD" \
      --steps "${TAIL_STEPS:-2000}" --eval_every "${TAIL_EVAL_EVERY:-200}" --device "$DEVICE" --seed "$SEED"
    ;;
  multi_proto)
    ensure_stage1_cache
    K="${ARG:-2}"
    py train_multi_prototype --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" \
      --output_dir "$PROTO_ROOT/k${K}_seed${SEED}" --prototypes "$K" \
      --steps "${PROTO_STEPS:-3000}" --eval_every "${PROTO_EVAL_EVERY:-250}" --device "$DEVICE" --seed "$SEED"
    ;;
  expert_oracle)
    ensure_stage1_cache
    py expert_oracle --val_cache "$VAL_CACHE" --output "$EVAL_ROOT/C_expert_oracle.json" --candidate "${EXPERT_CANDIDATE:-union}"
    ;;
  expert_train)
    ensure_stage1_cache
    TAIL_CKPT="${TAIL_CKPT:-$PROTO_ROOT/best_tail.pth}"
    [[ -s "$TAIL_CKPT" ]] || TAIL_CKPT=""
    EXTRA=(); [[ -n "$TAIL_CKPT" ]] && EXTRA+=(--tail_ckpt "$TAIL_CKPT")
    py train_expert --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" \
      --output_dir "$EXPERT_ROOT/seed${SEED}" --candidate "${EXPERT_CANDIDATE:-union}" \
      --steps "${EXPERT_STEPS:-3000}" --eval_every "${EXPERT_EVAL_EVERY:-250}" --device "$DEVICE" --seed "$SEED" "${EXTRA[@]}"
    ;;
  proto_oracle)
    ensure_stage1_cache
    K="${ARG:-1}"
    TAIL_CKPT="${TAIL_CKPT:-$PROTO_ROOT/best_tail.pth}"
    EXTRA=(); [[ -s "$TAIL_CKPT" ]] && EXTRA+=(--tail_ckpt "$TAIL_CKPT")
    py prototype_oracle --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" \
      --output "$ORACLE_ROOT/k${K}_exclude_event.json" --prototypes "$K" --exclude_same_event --device "$DEVICE" --seed "$SEED" "${EXTRA[@]}"
    ;;
  memory_oracle)
    ensure_stage1_cache
    CONTROL="${ARG:-none}"
    TAIL_CKPT="${TAIL_CKPT:-$PROTO_ROOT/best_tail.pth}"
    EXTRA=(); [[ -s "$TAIL_CKPT" ]] && EXTRA+=(--tail_ckpt "$TAIL_CKPT")
    py memory_oracle --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" \
      --output "$MEM_ROOT/${CONTROL}_exclude_event.json" --control "$CONTROL" --exclude_same_event \
      --coarse_k "${MEM_COARSE_K:-50}" --topk "${MEM_TOPK:-20}" --local_weight "${MEM_LOCAL_WEIGHT:-0.35}" \
      --device "$DEVICE" --seed "$SEED" "${EXTRA[@]}"
    ;;
  memory_verifier)
    ensure_stage1_cache
    TAIL_CKPT="${TAIL_CKPT:-$PROTO_ROOT/best_tail.pth}"
    EXTRA=(); [[ -s "$TAIL_CKPT" ]] && EXTRA+=(--tail_ckpt "$TAIL_CKPT")
    py memory_verifier --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" \
      --output_dir "$OUT_ROOT/E_memory_verifier/seed${SEED}" --steps "${VERIFIER_STEPS:-3000}" \
      --coarse_k "${MEM_COARSE_K:-50}" --topk "${MEM_TOPK:-20}" --local_weight "${MEM_LOCAL_WEIGHT:-0.35}" \
      --device "$DEVICE" --seed "$SEED" "${EXTRA[@]}"
    ;;
  status)
    pgrep -af 'damageformer_causal_research_chain_v1|damageformer_building_internal_change_partition_v2' || true
    find "$OUT_ROOT" "$EVAL_ROOT" -maxdepth 4 -type f \( -name summary.json -o -name '*.json' \) -print 2>/dev/null | sort || true
    ;;
  *)
    echo "Usage: $0 {preflight|cache_stage1|cache_stage2_val|audit_region|compare_feedback|tail_probe METHOD|multi_proto K|expert_oracle|expert_train|proto_oracle K|memory_oracle CONTROL|memory_verifier|status}" >&2
    exit 2
    ;;
esac
