#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-/root/data/BRIGHT-master/bda_benchmark}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${TARGET}/experiments/damageformer_building_context_v2"

mkdir -p "${DEST}"
cp -a "${HERE}/." "${DEST}/"
rm -f "${DEST}/install.sh"

chmod +x \
  "${DEST}/run.sh" \
  "${DEST}/run_building_variants.sh" \
  "${DEST}/run_everything.sh"

echo "Installed to ${DEST}"
echo "Next:"
echo "  cd ${TARGET}"
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "  conda activate bright4090"
echo "  bash experiments/damageformer_building_context_v2/run.sh preflight"
