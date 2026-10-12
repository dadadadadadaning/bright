#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${RUN_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${ROOT}:${ROOT}/script/standard_ML:${PYTHONPATH:-}"

required=(
  "model/DamageFormer_S9Ic_ReliableGeometry.py"
  "model/DamageFormer_S9Id_CenterOffset.py"
  "model/instance_geometry/config.py"
  "model/instance_geometry/targets.py"
  "model/instance_geometry/losses.py"
  "model/instance_geometry/decode.py"
  "model/instance_geometry/metrics.py"
  "model/instance_geometry/s8_backbone.py"
  "model/instance_geometry/maskdino.py"
  "script/standard_ML/s9ic_targets_losses.py"
  "script/standard_ML/s9ic_g1_geometry.py"
  "script/standard_ML/s9ic_shared.py"
  "script/standard_ML/train_S9Ic_G1.py"
  "script/standard_ML/train_S9Ic_G2.py"
  "script/standard_ML/train_S9Ic_G3.py"
  "script/standard_ML/train_S9Id.py"
  "script/standard_ML/eval_S9Ic_G1_val.py"
  "script/standard_ML/smoke_S9Ic_G1.py"
)

for file in "${required[@]}"; do
  [[ -f "${file}" ]] || { echo "MISSING ${file}"; exit 2; }
  echo "PASS ${file}"
done

python -m py_compile "${required[@]}"
bash -n \
  run/run_S9Ic_G1_smoke.sh \
  run/run_S9Ic_G1_train.sh \
  run/run_S9Ic_G1_eval_val.sh

python tests/test_s9ic_targets_losses.py
python tests/test_s9ic_edt_center.py
python tests/test_s9ic_g1_geometry.py
python tests/test_s9ic_ablation_switches.py
python tests/test_s9id_center_offset.py
python tests/test_s9_center_metrics.py
python tests/test_s9_instance_metrics.py
python tests/test_s9_maskdino_adapter.py
python tests/test_s9_s8_frozen_wrapper.py
python tests/test_s9ic_protocol.py

echo "S9-Ic-G1 environment/static checks: PASS"
