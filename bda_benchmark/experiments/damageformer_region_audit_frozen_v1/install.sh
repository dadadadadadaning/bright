#!/usr/bin/env bash
set -euo pipefail
TARGET="${1:-/root/data/BRIGHT-master/bda_benchmark}"; HERE="$(cd "$(dirname "$0")" && pwd)"; DEST="${TARGET}/experiments/damageformer_region_audit_frozen_v1"
mkdir -p "$DEST"; cp -a "${HERE}/." "$DEST/"; rm -f "$DEST/install.sh"; chmod +x "$DEST/run.sh"; echo "Installed: $DEST"
