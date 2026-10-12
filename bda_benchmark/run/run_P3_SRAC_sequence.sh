#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-sr1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/P3_SRAC_CONTROLLED}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/P3_SRAC_CONTROLLED}"
mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

bash run/run_P3_SRAC_controlled_5090.sh preflight
SKIP_PREFLIGHT=1 bash run/run_P3_SRAC_controlled_5090.sh guard \
  2>&1 | tee "${LOG_ROOT}/P3_SR0_guard_seed${SEED:-3407}.log"

case "${MODE}" in
  e3a)
    seed="${SEED:-3407}"
    run_tag="${RUN_TAG:-v1}"
    h6_checkpoint="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
    test -s "${h6_checkpoint}" || {
      echo "Missing frozen H6-A2 native-SAR source: ${h6_checkpoint}" >&2
      exit 1
    }
    baseline_name="P3_SR1PAA_affinity_baseline_seed${seed}_${run_tag}"
    train_name="P3_SR1PAA_affinity500_seed${seed}_${run_tag}"
    eval_name="P3_SR1PAA_affinity500_independent_val_seed${seed}_${run_tag}"
    train_dir="${OUTPUT_ROOT}/${train_name}"

    echo "[E3A-A0] untrained building-conditioned affinity baseline"
    SKIP_PREFLIGHT=1 VARIANT=sr1paa H6_A2_CKPT="${h6_checkpoint}" \
      OPTIMIZER_STEPS=0 EXP_NAME="${baseline_name}" \
      bash run/run_P3_SRAC_controlled_5090.sh screen \
      2>&1 | tee "${LOG_ROOT}/${baseline_name}.log"

    echo "[E3A-A1] affinity-only smoke; P3 output remains exact"
    SKIP_PREFLIGHT=1 VARIANT=sr1paa H6_A2_CKPT="${h6_checkpoint}" \
      REGION_LOSS_WEIGHT=1.0 \
      EXP_NAME="P3_SR1PAA_smoke20_seed${seed}_${run_tag}" \
      bash run/run_P3_SRAC_controlled_5090.sh smoke \
      2>&1 | tee "${LOG_ROOT}/P3_SR1PAA_smoke20_seed${seed}_${run_tag}.log"

    echo "[E3A-A2] 500-step affinity-only training"
    SKIP_PREFLIGHT=1 VARIANT=sr1paa H6_A2_CKPT="${h6_checkpoint}" \
      REGION_LOSS_WEIGHT=1.0 OPTIMIZER_STEPS=500 EVAL_EVERY=100 SAVE_EVERY=500 \
      EXP_NAME="${train_name}" \
      bash run/run_P3_SRAC_controlled_5090.sh screen \
      2>&1 | tee "${LOG_ROOT}/${train_name}.log"

    echo "[E3A-A3] strict independent validation of latest affinity weights"
    SKIP_PREFLIGHT=1 VARIANT=sr1paa H6_A2_CKPT="${h6_checkpoint}" \
      SRAC_CKPT="${train_dir}/latest_resume.pth" EVAL_NAME="${eval_name}" \
      bash run/run_P3_SRAC_controlled_5090.sh eval_val \
      2>&1 | tee "${LOG_ROOT}/${eval_name}.log"

    python script/standard_ML/summarize_P3_SRAC_affinity.py \
      --baseline "${OUTPUT_ROOT}/${baseline_name}/validation_metrics.json" \
      --candidate "${OUTPUT_ROOT}/${eval_name}/validation_metrics.json" \
      --output "${train_dir}/affinity_gate.json"
    echo "E3A completed. P3 was unchanged; no region propagation, retrieval, or Test was run."
    ;;
  e2i)
    seed="${SEED:-3407}"
    run_tag="${RUN_TAG:-v1}"
    baseline_name="P3_SR1PAI_partition_baseline_seed${seed}_${run_tag}"
    stage_a_name="P3_SR1PAI_partition500_seed${seed}_${run_tag}"
    stage_a_eval="P3_SR1PAI_partition500_independent_val_seed${seed}_${run_tag}"
    stage_a_dir="${OUTPUT_ROOT}/${stage_a_name}"
    partition_gate="${stage_a_dir}/partition_gate.json"

    echo "[E2I-A0] untrained three-iteration partition baseline"
    SKIP_PREFLIGHT=1 VARIANT=sr1pai OPTIMIZER_STEPS=0 \
      EXP_NAME="${baseline_name}" \
      bash run/run_P3_SRAC_controlled_5090.sh screen \
      2>&1 | tee "${LOG_ROOT}/${baseline_name}.log"

    echo "[E2I-A1] iterative partition-only smoke"
    SKIP_PREFLIGHT=1 VARIANT=sr1pai REGION_LOSS_WEIGHT=1.0 \
      SRAC_BUILDING_WEIGHT=1.0 SRAC_BOUNDARY_WEIGHT=0.5 \
      SRAC_ENTROPY_WEIGHT=0.02 SRAC_ENTROPY_TARGET=0.60 \
      EXP_NAME="P3_SR1PAI_smoke20_seed${seed}_${run_tag}" \
      bash run/run_P3_SRAC_controlled_5090.sh smoke \
      2>&1 | tee "${LOG_ROOT}/P3_SR1PAI_smoke20_seed${seed}_${run_tag}.log"

    echo "[E2I-A2] 500-step three-iteration partition pretraining"
    SKIP_PREFLIGHT=1 VARIANT=sr1pai REGION_LOSS_WEIGHT=1.0 \
      SRAC_BUILDING_WEIGHT=1.0 SRAC_BOUNDARY_WEIGHT=0.5 \
      SRAC_ENTROPY_WEIGHT=0.02 SRAC_ENTROPY_TARGET=0.60 \
      OPTIMIZER_STEPS=500 EVAL_EVERY=100 SAVE_EVERY=500 \
      EXP_NAME="${stage_a_name}" \
      bash run/run_P3_SRAC_controlled_5090.sh screen \
      2>&1 | tee "${LOG_ROOT}/${stage_a_name}.log"

    echo "[E2I-A3] strict validation of completed iterative partition"
    SKIP_PREFLIGHT=1 VARIANT=sr1pai \
      SRAC_CKPT="${stage_a_dir}/latest_resume.pth" \
      EVAL_NAME="${stage_a_eval}" \
      bash run/run_P3_SRAC_controlled_5090.sh eval_val \
      2>&1 | tee "${LOG_ROOT}/${stage_a_eval}.log"

    python script/standard_ML/summarize_P3_SRAC_partition.py \
      --baseline "${OUTPUT_ROOT}/${baseline_name}/validation_metrics.json" \
      --candidate "${OUTPUT_ROOT}/${stage_a_eval}/validation_metrics.json" \
      --output "${partition_gate}"
    partition_passed="$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["passed"]))' "${partition_gate}")"
    if [[ "${partition_passed}" != "1" ]]; then
      echo "E2I stops: iterative partition gate failed. Projection remains closed; Test was never accessed."
      exit 0
    fi

    stage_b_name="P3_SR1PIB_screen1000_seed${seed}_${run_tag}"
    echo "[E2I-B1] iterative boundary-aware safe-projection smoke"
    SKIP_PREFLIGHT=1 VARIANT=sr1pib \
      SR1PAI_CKPT="${stage_a_dir}/latest_resume.pth" \
      REGION_LOSS_WEIGHT=1.0 SRAC_BUILDING_WEIGHT=0.10 \
      SRAC_BOUNDARY_WEIGHT=0.05 SRAC_ENTROPY_WEIGHT=0.01 \
      SRAC_ENTROPY_TARGET=0.60 \
      EXP_NAME="P3_SR1PIB_smoke20_seed${seed}_${run_tag}" \
      bash run/run_P3_SRAC_controlled_5090.sh smoke \
      2>&1 | tee "${LOG_ROOT}/P3_SR1PIB_smoke20_seed${seed}_${run_tag}.log"

    echo "[E2I-B2] 1000-step iterative safe projection"
    SKIP_PREFLIGHT=1 VARIANT=sr1pib \
      SR1PAI_CKPT="${stage_a_dir}/latest_resume.pth" \
      REGION_LOSS_WEIGHT=1.0 SRAC_BUILDING_WEIGHT=0.10 \
      SRAC_BOUNDARY_WEIGHT=0.05 SRAC_ENTROPY_WEIGHT=0.01 \
      SRAC_ENTROPY_TARGET=0.60 \
      OPTIMIZER_STEPS=1000 EVAL_EVERY=250 SAVE_EVERY=1000 \
      EXP_NAME="${stage_b_name}" \
      bash run/run_P3_SRAC_controlled_5090.sh screen \
      2>&1 | tee "${LOG_ROOT}/${stage_b_name}.log"

    python script/standard_ML/summarize_P3_SRAC.py \
      --baseline "${OUTPUT_ROOT}/P3_SR0_guard_seed${seed}/validation_metrics.json" \
      --candidate "${OUTPUT_ROOT}/${stage_b_name}/best_event_worst.pth" \
      --output "${OUTPUT_ROOT}/${stage_b_name}/advancement_gate.json"
    echo "E2I completed through SR1PIB validation screening. Test was never accessed."
    ;;
  e2)
    seed="${SEED:-3407}"
    run_tag="${RUN_TAG:-v1}"
    baseline_name="P3_SR1PA_partition_baseline_seed${seed}_${run_tag}"
    stage_a_name="P3_SR1PA_partition500_seed${seed}_${run_tag}"
    stage_a_eval="P3_SR1PA_partition500_independent_val_seed${seed}_${run_tag}"
    stage_a_dir="${OUTPUT_ROOT}/${stage_a_name}"
    partition_gate="${stage_a_dir}/partition_gate.json"

    echo "[E2-A0] untrained boundary-partition validation baseline"
    SKIP_PREFLIGHT=1 VARIANT=sr1pa OPTIMIZER_STEPS=0 \
      EXP_NAME="${baseline_name}" \
      bash run/run_P3_SRAC_controlled_5090.sh screen \
      2>&1 | tee "${LOG_ROOT}/${baseline_name}.log"

    echo "[E2-A1] partition-only smoke"
    SKIP_PREFLIGHT=1 VARIANT=sr1pa REGION_LOSS_WEIGHT=1.0 \
      SRAC_BUILDING_WEIGHT=1.0 SRAC_BOUNDARY_WEIGHT=0.5 \
      SRAC_ENTROPY_WEIGHT=0.02 SRAC_ENTROPY_TARGET=0.60 \
      EXP_NAME="P3_SR1PA_smoke20_seed${seed}_${run_tag}" \
      bash run/run_P3_SRAC_controlled_5090.sh smoke \
      2>&1 | tee "${LOG_ROOT}/P3_SR1PA_smoke20_seed${seed}_${run_tag}.log"

    echo "[E2-A2] 500-step partition pretraining"
    SKIP_PREFLIGHT=1 VARIANT=sr1pa REGION_LOSS_WEIGHT=1.0 \
      SRAC_BUILDING_WEIGHT=1.0 SRAC_BOUNDARY_WEIGHT=0.5 \
      SRAC_ENTROPY_WEIGHT=0.02 SRAC_ENTROPY_TARGET=0.60 \
      OPTIMIZER_STEPS=500 EVAL_EVERY=100 SAVE_EVERY=500 \
      EXP_NAME="${stage_a_name}" \
      bash run/run_P3_SRAC_controlled_5090.sh screen \
      2>&1 | tee "${LOG_ROOT}/${stage_a_name}.log"

    echo "[E2-A3] strict validation of the completed partition"
    SKIP_PREFLIGHT=1 VARIANT=sr1pa \
      SRAC_CKPT="${stage_a_dir}/latest_resume.pth" \
      EVAL_NAME="${stage_a_eval}" \
      bash run/run_P3_SRAC_controlled_5090.sh eval_val \
      2>&1 | tee "${LOG_ROOT}/${stage_a_eval}.log"

    python script/standard_ML/summarize_P3_SRAC_partition.py \
      --baseline "${OUTPUT_ROOT}/${baseline_name}/validation_metrics.json" \
      --candidate "${OUTPUT_ROOT}/${stage_a_eval}/validation_metrics.json" \
      --output "${partition_gate}"
    partition_passed="$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["passed"]))' "${partition_gate}")"
    if [[ "${partition_passed}" != "1" ]]; then
      echo "E2 stops: SR1PA partition gate failed. Projection remains closed; Test was never accessed."
      exit 0
    fi

    stage_b_name="P3_SR1PB_screen1000_seed${seed}_${run_tag}"
    echo "[E2-B1] boundary-aware safe-projection smoke"
    SKIP_PREFLIGHT=1 VARIANT=sr1pb \
      SR1PA_CKPT="${stage_a_dir}/latest_resume.pth" \
      REGION_LOSS_WEIGHT=1.0 SRAC_BUILDING_WEIGHT=0.10 \
      SRAC_BOUNDARY_WEIGHT=0.05 SRAC_ENTROPY_WEIGHT=0.01 \
      SRAC_ENTROPY_TARGET=0.60 \
      EXP_NAME="P3_SR1PB_smoke20_seed${seed}_${run_tag}" \
      bash run/run_P3_SRAC_controlled_5090.sh smoke \
      2>&1 | tee "${LOG_ROOT}/P3_SR1PB_smoke20_seed${seed}_${run_tag}.log"

    echo "[E2-B2] 1000-step boundary-aware safe projection"
    SKIP_PREFLIGHT=1 VARIANT=sr1pb \
      SR1PA_CKPT="${stage_a_dir}/latest_resume.pth" \
      REGION_LOSS_WEIGHT=1.0 SRAC_BUILDING_WEIGHT=0.10 \
      SRAC_BOUNDARY_WEIGHT=0.05 SRAC_ENTROPY_WEIGHT=0.01 \
      SRAC_ENTROPY_TARGET=0.60 \
      OPTIMIZER_STEPS=1000 EVAL_EVERY=250 SAVE_EVERY=1000 \
      EXP_NAME="${stage_b_name}" \
      bash run/run_P3_SRAC_controlled_5090.sh screen \
      2>&1 | tee "${LOG_ROOT}/${stage_b_name}.log"

    python script/standard_ML/summarize_P3_SRAC.py \
      --baseline "${OUTPUT_ROOT}/P3_SR0_guard_seed${seed}/validation_metrics.json" \
      --candidate "${OUTPUT_ROOT}/${stage_b_name}/best_event_worst.pth" \
      --output "${OUTPUT_ROOT}/${stage_b_name}/advancement_gate.json"
    echo "E2 completed through SR1PB validation screening. Test was never accessed."
    ;;
  sr1|sr1f|sr1p|sr1pa|sr1pb|sr1pai|sr1pib|sr1paa|sr2)
    SKIP_PREFLIGHT=1 VARIANT="${MODE}" bash run/run_P3_SRAC_controlled_5090.sh smoke \
      2>&1 | tee "${LOG_ROOT}/P3_${MODE^^}_smoke20_seed${SEED:-3407}.log"
    SKIP_PREFLIGHT=1 VARIANT="${MODE}" bash run/run_P3_SRAC_controlled_5090.sh screen \
      2>&1 | tee "${LOG_ROOT}/P3_${MODE^^}_screen${OPTIMIZER_STEPS:-3000}_seed${SEED:-3407}.log"
    ;;
  *)
    echo "Expected e2|e2i|e3a|sr1|sr1f|sr1p|sr1pa|sr1pb|sr1pai|sr1pib|sr1paa|sr2. Run a projection stage only after its evidence audit passes." >&2
    exit 2
    ;;
esac

echo "P3-SRAC ${MODE} sequence completed. Test was never accessed."
