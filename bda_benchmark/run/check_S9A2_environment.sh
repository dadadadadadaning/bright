#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
cd "${PROJECT_ROOT}"

required=(
  "script/standard_ML/s9_instance_common.py"
  "script/standard_ML/s9_watershed_common.py"
  "script/standard_ML/eval_S9A2_probability_watershed.py"
  "run/run_S9A2_probability_watershed_val.sh"
)

for file in "${required[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing S9-A2 file: ${file}"
    exit 2
  fi
done

python -m py_compile \
  script/standard_ML/s9_instance_common.py \
  script/standard_ML/s9_watershed_common.py \
  script/standard_ML/eval_S9A2_probability_watershed.py

python - <<'PY'
import numpy
import scipy
import skimage
import torch

from model.DamageFormer_S8_MICB_DSMIL import (
    BuildingBagSpec,
    DamageFormer_S8_MICB_DSMIL,
)
from script.standard_ML.s9_watershed_common import (
    labels_to_instances_and_specs,
    locked_watershed_grid,
    probability_seed_watershed_labels,
)

print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("skimage:", skimage.__version__)
print("model:", DamageFormer_S8_MICB_DSMIL)
print("BuildingBagSpec:", BuildingBagSpec)
print("grid size:", len(locked_watershed_grid()))
print("watershed:", probability_seed_watershed_labels)
print("spec converter:", labels_to_instances_and_specs)
print("S9-A2 environment: PASS")
PY
