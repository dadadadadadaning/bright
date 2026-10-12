#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

python -m pip install \
  -e . \
  --no-deps \
  --no-build-isolation

python - <<'PY'
import rgid
print("Installed RGID from:", rgid.__file__)

normalized = rgid.__file__.replace("\\", "/")
if "experiments/rgid_s9ic_clean/rgid" not in normalized:
    raise RuntimeError(
        "RGID imported from wrong path: " + rgid.__file__
    )
PY
