#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
cd "${PROJECT_ROOT}"

required=(
  "script/standard_ML/s9_instance_common.py"
  "script/standard_ML/eval_S9A_predicted_instance.py"
  "run/run_S9A_predicted_instance_val.sh"
)

for file in "${required[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing S9-A file: ${file}"
    exit 2
  fi
done

python -m py_compile \
  script/standard_ML/s9_instance_common.py \
  script/standard_ML/eval_S9A_predicted_instance.py

python - <<'PY'
import inspect
import numpy
import scipy
import torch

from model.DamageFormer_S8_MICB_DSMIL import (
    BuildingBagSpec,
    DamageFormer_S8_MICB_DSMIL,
)
from script.standard_ML.train_S8_MICB_DSMIL import (
    build_training_bag_specs,
)
from script.standard_ML.s9_instance_common import (
    exact_s8_instances_from_labels,
    exact_s8_predicted_instances,
)

print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("model:", DamageFormer_S8_MICB_DSMIL)
print("BuildingBagSpec signature:", inspect.signature(BuildingBagSpec))
print("BuildingBagSpec annotations:", getattr(BuildingBagSpec, "__annotations__", {}))
print("bag builder:", build_training_bag_specs)
print("exact GT instances:", exact_s8_instances_from_labels)
print("exact predicted instances:", exact_s8_predicted_instances)
print("S9-A v3 environment: PASS")
PY
