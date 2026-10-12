#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${1:-/root/data/BRIGHT-master/bda_benchmark}"
DEST="$PROJECT/experiments/damageformer_causal_research_chain_v1"
mkdir -p "$DEST"
cp -a "$SRC"/. "$DEST"/
echo "Installed to $DEST"
