#!/usr/bin/env bash
set -euo pipefail
TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
REPO="${SAR_REPO:-/root/shared-nvme/BRIGHT-master}"
PROTOCOL="${SAR_PROTOCOL:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json}"
BASE=/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_Q_MATCHED_V1
RUN="$BASE/run_$(date -u +%Y%m%d_%H%M%S)_${BASHPID}"
test -x "$PYTHON"
test -f "$PROTOCOL"
test -f "$REPO/bda_benchmark/experiments/damageformer_srac_q_factorial_v1/train_q.py"
export BRIGHT_REPO_ROOT="$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" - "$TOOL" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'SHA256SUMS.json').read_text())
for name, expected in manifest.items():
    if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit('工具校验失败：' + name)
print('工具校验通过')
PY
"$PYTHON" -m unittest discover -s "$TOOL" -p 'test_*.py' -q
mkdir -p "$BASE"
test ! -e "$RUN"
test ! -e "$RUN.log"
"$PYTHON" "$TOOL/pipeline.py" plan --repo-root "$REPO" --protocol "$PROTOCOL" --output "$RUN" > "$RUN.plan.json"
nohup "$PYTHON" -u "$TOOL/pipeline.py" run --repo-root "$REPO" --protocol "$PROTOCOL" \
  --output "$RUN" --amp bf16 --wait-gpu-idle > "$RUN.log" 2>&1 < /dev/null &
printf '后台PID=%s\n工具目录=%s\n结果目录=%s\n日志=%s.log\n' "$!" "$TOOL" "$RUN" "$RUN"
printf '查看进度：tail -n 60 -f "%s.log"\n' "$RUN"
printf '读取报告："%s" "%s/pipeline.py" report --root "%s"\n' "$PYTHON" "$TOOL" "$RUN"
printf '启动不等于训练通过。GPU 忙时等待；四组 smoke 均通过后开始正式训练。\n'
