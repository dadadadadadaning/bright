#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
BDA_ROOT="${1:-/root/data/BRIGHT-master/bda_benchmark}"
DEST="${BDA_ROOT}/experiments/damageformer_building_internal_change_partition_v2"
mkdir -p "${DEST}"
find "${DEST}" -mindepth 1 -maxdepth 1 -type f -delete
cp -a "${SRC}"/*.py "${SRC}"/*.sh "${SRC}"/*.md "${DEST}/" 2>/dev/null || true
chmod +x "${DEST}"/*.sh 2>/dev/null || true
echo "Installed V2 experiment to: ${DEST}"
echo "Next: cd ${BDA_ROOT} && bash experiments/damageformer_building_internal_change_partition_v2/run.sh preflight"
