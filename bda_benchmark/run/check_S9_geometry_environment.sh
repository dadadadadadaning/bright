#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
cd "${ROOT}"
files=(
  model/DamageFormer_S9I_CenterBoundary.py
  script/standard_ML/s9_instance_common.py
  script/standard_ML/s9_geometry_common.py
  script/standard_ML/eval_S9A2b_conditional_watershed.py
  script/standard_ML/train_S9I_center_boundary.py
  script/standard_ML/eval_S9I_center_boundary.py
  script/standard_ML/inspect_S9I_feature.py
)
for file in "${files[@]}"; do
  [[ -f "${file}" ]] || { echo "MISSING: ${file}"; exit 2; }
done
python -m py_compile "${files[@]}"
bash -n run/run_S9A2b_conditional_watershed_val.sh
bash -n run/run_S9I_center_boundary_train.sh
bash -n run/run_S9I_center_boundary_eval_val.sh
python - <<'PY'
import numpy, scipy, skimage, torch
print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("skimage:", skimage.__version__)
print("S9 geometry imports: PASS")
PY
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python script/standard_ML/inspect_S9I_feature.py
