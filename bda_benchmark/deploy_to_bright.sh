#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:-/root/shared-nvme/BRIGHT-master/bda_benchmark}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$TARGET/model" "$TARGET/script/standard_ML"
cp -v "$HERE/model/DamageFormer_Hierarchical_RGBSAR.py" "$TARGET/model/"
cp -v "$HERE/script/standard_ML/train_Hierarchical_RGBSAR.py" "$TARGET/script/standard_ML/"
cp -v "$HERE/script/standard_ML/eval_Hierarchical_RGBSAR.py" "$TARGET/script/standard_ML/"
cp -v "$HERE/script/standard_ML/check_Hierarchical_RGBSAR.py" "$TARGET/script/standard_ML/"

python -m py_compile \
  "$TARGET/model/DamageFormer_Hierarchical_RGBSAR.py" \
  "$TARGET/script/standard_ML/train_Hierarchical_RGBSAR.py" \
  "$TARGET/script/standard_ML/eval_Hierarchical_RGBSAR.py" \
  "$TARGET/script/standard_ML/check_Hierarchical_RGBSAR.py"

echo "DEPLOYMENT PASS: $TARGET"
echo "Next: cd $TARGET && python script/standard_ML/check_Hierarchical_RGBSAR.py"
