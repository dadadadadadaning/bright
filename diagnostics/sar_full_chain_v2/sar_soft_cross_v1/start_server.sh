#!/usr/bin/env bash
set -euo pipefail
TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
REPO="${BRIGHT_REPO:-/root/shared-nvme/BRIGHT-master}"
BASE=/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_SOFT_CROSS_V1
export BRIGHT_REPO="$REPO" BRIGHT_REPO_ROOT="$REPO" PYTHONDONTWRITEBYTECODE=1
test -x "$PYTHON"
test -s "$REPO/bda_benchmark/experiments/damageformer_srac_building_route_v1/train.py"
"$PYTHON" - "$TOOL" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
files = json.loads((root / 'SHA256SUMS.json').read_text())
for name, expected in files.items():
    if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit('工具校验失败：' + name)
print('工具校验通过')
PY
SOURCE_ARGS=()
if [[ -n "${SAR_SOURCE_ROOT:-}" ]]; then SOURCE_ARGS+=(--sar-root "$SAR_SOURCE_ROOT"); fi
SAR_ROOT="$("$PYTHON" "$TOOL/run.py" source "${SOURCE_ARGS[@]}")"
printf '固定本次SAR来源：%s\n' "$SAR_ROOT"
"$PYTHON" -m unittest discover -s "$TOOL" -p 'test_*.py' -q
mkdir -p "$BASE"
RUN="$BASE/run_$(date -u +%Y%m%d_%H%M%S)_${BASHPID}"
test ! -e "$RUN"
test ! -e "$RUN.log"
"$PYTHON" "$TOOL/run.py" plan --repo-root "$REPO" --sar-root "$SAR_ROOT" --output "$RUN" > "$RUN.plan.json"
nohup "$PYTHON" -u "$TOOL/run.py" run --repo-root "$REPO" --sar-root "$SAR_ROOT" \
  --output "$RUN" > "$RUN.log" 2>&1 < /dev/null &
printf '后台PID=%s\n工具目录=%s\n结果目录=%s\n日志=%s.log\n' "$!" "$TOOL" "$RUN" "$RUN"
printf '查看进度：tail -n 60 -f "%s.log"\n' "$RUN"
printf '读取报告："%s" "%s/run.py" report --root "%s"\n' "$PYTHON" "$TOOL" "$RUN"
printf '先等SAR四组完整通过，再核对来源、等待GPU、smoke、八头匹配训练和独立Val450。\n'
