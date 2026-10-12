#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
M=bda_benchmark.experiments.damageformer_srac_fourstage_v2
mode="${1:-preflight}"
if [[ "$mode" == preflight ]]; then
  python -m unittest "$M.test_contracts" "$M.test_losses" "$M.test_fourstage" -v
  for stage in train_regions build_memory build_neighbors train_retrieval evaluate; do python -m "$M.$stage" --help; done
  echo 'Local contracts passed; full GPU/data smoke still required. Test forbidden.'
  exit 0
fi
case "$mode" in smoke_all|pipeline|regions|build_memory|build_neighbors|train_heads|evaluate) ;; *) echo 'Use preflight|smoke_all|regions|build_memory|build_neighbors|train_heads|evaluate|pipeline'; exit 2;; esac
: "${RUN_ROOT:?Set a NEW dedicated RUN_ROOT (not an old experiment directory)}"
COMMON=(--dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
 --train_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
 --val_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
 --workers "${WORKERS:-4}" --device "${DEVICE:-cuda:0}")
EXTRA=(); region_steps=50000; head_steps=5000; region_eval=1000; head_eval=500
if [[ "$mode" == smoke_all ]]; then EXTRA=(--smoke); region_steps=20; head_steps=20; region_eval=20; head_eval=20; fi
FRONT="$RUN_ROOT/regions/best.pth"
CACHE="$RUN_ROOT/train_memory"
if [[ "${DIAGNOSTIC_STEPS:-0}" != 0 && "$mode" != regions ]]; then
 echo 'Diagnostic early stop only supported in regions mode'; exit 2
fi
regions() {
 INIT_ARGS=()
 if [[ "${INIT_KIND:-swin}" == a1 ]]; then
  test -s "${A1_CKPT:?Set audited A1 checkpoint path}"
  INIT_ARGS=(--a1_checkpoint "$A1_CKPT" --base_lr "${BASE_LR:-1e-5}" --new_lr "${NEW_LR:-1e-4}")
 elif [[ "${INIT_KIND:-swin}" == swin ]]; then
  test -s "${SWIN_PRETRAIN:?Set real torchvision Swin-T pretrain file; NOT A1/iter3 task weights}"
  INIT_ARGS=(--pretrain "$SWIN_PRETRAIN")
 else echo 'Unknown INIT_KIND'; exit 2; fi
 python -u -m "$M.train_regions" "${COMMON[@]}" "${EXTRA[@]}" \
  "${INIT_ARGS[@]}" --output "$RUN_ROOT/regions" \
  --steps "$region_steps" --eval_every "$region_eval" \
  --batch "${BATCH:-2}" --accum "${ACCUM:-8}" --seed "${SEED:-3407}" --amp "${AMP:-bf16}" \
  --diagnostic_steps "${DIAGNOSTIC_STEPS:-0}"
}
build_memory() {
 python -u -m "$M.build_memory" "${COMMON[@]}" "${EXTRA[@]}" \
  --checkpoint "$FRONT" --output "$CACHE" --per_bucket "${PER_BUCKET:-64}" --seed "${SEED:-3407}"
}
train_heads() {
 for head in nomemory memory; do
  python -u -m "$M.train_retrieval" "${COMMON[@]}" "${EXTRA[@]}" \
   --checkpoint "$FRONT" --cache "$CACHE" --neighbors "$RUN_ROOT/neighbors" --mode "$head" --output "$RUN_ROOT/$head" \
   --steps "$head_steps" --eval_every "$head_eval" --seed "${SEED:-3407}"
 done
}
build_neighbors() {
 python -u -m "$M.build_neighbors" --checkpoint "$FRONT" --cache "$CACHE" --output "$RUN_ROOT/neighbors"
}
evaluate() {
 python -u -m "$M.evaluate" "${COMMON[@]}" "${EXTRA[@]}" \
  --checkpoint "$FRONT" --output "$RUN_ROOT/evaluation/region_only"
 python -u -m "$M.evaluate" "${COMMON[@]}" "${EXTRA[@]}" \
  --checkpoint "$FRONT" --cache "$CACHE" --head "$RUN_ROOT/nomemory/best.pth" \
  --output "$RUN_ROOT/evaluation/nomemory"
 for intervention in normal random shuffle_labels off no_metadata; do
  python -u -m "$M.evaluate" "${COMMON[@]}" "${EXTRA[@]}" \
   --checkpoint "$FRONT" --cache "$CACHE" --head "$RUN_ROOT/memory/best.pth" \
   --intervention "$intervention" --output "$RUN_ROOT/evaluation/memory_$intervention"
 done
}
case "$mode" in
 smoke_all|pipeline)
  if [[ -e "$RUN_ROOT" ]]; then echo 'Pipeline requires a completely NEW RUN_ROOT'; exit 2; fi
  regions; build_memory; build_neighbors; train_heads; evaluate ;;
 *) "$mode" ;;
esac
echo "COMPLETE $mode. Train/Val only."
