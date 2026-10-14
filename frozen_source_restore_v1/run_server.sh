#!/usr/bin/env bash
# Check/restore immutable E0 inputs before starting the existing four-arm package.
set -euo pipefail
BRIGHT_REPAIR_TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIGHT_REPAIR_PY="${BRIGHT_REPAIR_PY:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
BRIGHT_REPAIR_MODE="${1:-audit}"
if (( $# > 0 )); then shift; fi
case "$BRIGHT_REPAIR_MODE" in
  audit|start) BRIGHT_REPAIR_ACTION=audit ;;
  restore|restore-start) BRIGHT_REPAIR_ACTION=restore ;;
  *) echo '用法：bash run_server.sh audit|restore|restore-start|start [--backup 路径]'; exit 2 ;;
esac
test -x "$BRIGHT_REPAIR_PY"
BRIGHT_REPORT_BASE=/root/shared-nvme/BRIGHT-master/New/source_restore_reports
mkdir -p "$BRIGHT_REPORT_BASE"
BRIGHT_REPAIR_OUT="$(mktemp -d "$BRIGHT_REPORT_BASE/run_$(date +%Y%m%d_%H%M%S)_XXXXXXXX")"
printf '依赖检查报告：%s/report.json\n' "$BRIGHT_REPAIR_OUT"
"$BRIGHT_REPAIR_PY" "$BRIGHT_REPAIR_TOOL/restore_sources.py" "$BRIGHT_REPAIR_ACTION" \
  --report "$BRIGHT_REPAIR_OUT/report.json" "$@"
if [[ "$BRIGHT_REPAIR_MODE" == start || "$BRIGHT_REPAIR_MODE" == restore-start ]]; then
  BRIGHT_TOOL_RECORD=/root/shared-nvme/BRIGHT-master/New/soft_memory_tool_path_independent.txt
  test -s "$BRIGHT_TOOL_RECORD" || { echo "找不到训练工具路径记录：$BRIGHT_TOOL_RECORD"; exit 2; }
  BRIGHT_TRAIN_TOOL="$(cat "$BRIGHT_TOOL_RECORD")"
  test -s "$BRIGHT_TRAIN_TOOL/run.py"
  test -s "$BRIGHT_TRAIN_TOOL/start_server.sh"
  printf 'E0 来源文件字节校验全部通过；启动新的四组运行：%s\n' "$BRIGHT_TRAIN_TOOL"
  bash "$BRIGHT_TRAIN_TOOL/start_server.sh" full --baseline-policy independent
else
  printf '依赖处理完成；本次没有启动 CUDA 或训练。\n'
fi
