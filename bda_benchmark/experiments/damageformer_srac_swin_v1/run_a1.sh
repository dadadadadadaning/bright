#!/usr/bin/env bash
set -euo pipefail
export INIT_KIND=a1
exec bash "$(dirname "$0")/run.sh" "${1:-preflight}"
