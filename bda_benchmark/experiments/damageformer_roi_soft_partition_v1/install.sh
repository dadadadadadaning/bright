#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-/root/data/BRIGHT-master/bda_benchmark}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${TARGET}/experiments/damageformer_roi_soft_partition_v1"

mkdir -p "${DEST}"
cp -a "${HERE}/." "${DEST}/"
rm -f "${DEST}/install.sh"

chmod +x "${DEST}/run.sh" "${DEST}/inspect_result.py"

echo "Installed: ${DEST}"
echo "Next:"
echo "  cd ${TARGET}"
echo "  bash experiments/damageformer_roi_soft_partition_v1/run.sh preflight"
