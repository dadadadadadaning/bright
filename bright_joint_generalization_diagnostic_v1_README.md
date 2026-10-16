# 四组联合实验的固定 Train／Val 诊断

这个脚本只读取已完成的 Q-rebased A／B／C／D 四组训练、原始 fix2 代码、数据和案例库，不更新权重，不运行 Test。它不是新的正式成绩，也不能单靠一轮曲线证明过拟合。

它从原训练日志的第 1–500 次更新中选前 450 张**不同且已经参与过训练**的 Train 图像。500、1500、5000 步的检查点都在这**同一批 Train450**与同一份 Val450 上，以推理模式重算原训练损失 `region_nll + Lovasz`、两项损失分量、前景三类 mIoU 和四类 IoU。Train450 是固定诊断子集，不代表整个 Train2918；报告同时记录这些图像在三个检查点前各参与过几次更新。

脚本调用原始未修改的 fix2 Runtime、Reader、loss 和数据加载器。它保留 Train 查询的同图／同事件案例排除，核对原始检查点、代码和输入文件哈希；完整模式的 Val450 四类 IoU 必须在 **0.02 个百分点**内复现对应的原报告，否则停止。报告还记录原始与重算的 confusion 是否完全相同及各格绝对差值；这项更严格的检查单独报告，不替代预定的 IoU 容差。500 和 1500 步仍是中间检查点，5000 步才是原实验正式终点。

将单个 `bright_joint_generalization_diagnostic_v1.py` 放在服务器 `/root/shared-nvme/BRIGHT-master/New/`。先检查只读计划（无需 Torch、CUDA 或数据前向）：

```bash
BRIGHT_PY=/root/shared-nvme/conda-envs/bright4090/bin/python
BRIGHT_DIAG=/root/shared-nvme/BRIGHT-master/New/bright_joint_generalization_diagnostic_v1.py
"$BRIGHT_PY" "$BRIGHT_DIAG" --plan
```

再跑真实 CUDA 小检查：第 500 步、两张固定 Train 与两张 Val 图像。使用全新的输出目录；`--smoke` 不是正式 Val 重放。

```bash
set -o pipefail
BRIGHT_DIAG_PARENT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_GENERALIZATION_DIAGNOSTIC_V1
mkdir -p "$BRIGHT_DIAG_PARENT"
BRIGHT_SMOKE="$BRIGHT_DIAG_PARENT/smoke_$(date +%Y%m%d_%H%M%S)_$$"
"$BRIGHT_PY" -u "$BRIGHT_DIAG" --smoke --output "$BRIGHT_SMOKE" 2>&1 | tee "$BRIGHT_SMOKE.log"
```

小检查通过后，再启动完整的 3 ×（Train450＋Val450）诊断。它需要重复 2700 张图像的冻结前端前向，可能运行较久；以下命令后台执行，不要求保持终端打开：

```bash
BRIGHT_DIAG_PARENT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_GENERALIZATION_DIAGNOSTIC_V1
mkdir -p "$BRIGHT_DIAG_PARENT"
BRIGHT_FULL="$BRIGHT_DIAG_PARENT/full_$(date +%Y%m%d_%H%M%S)_$$"
nohup "$BRIGHT_PY" -u "$BRIGHT_DIAG" --run --output "$BRIGHT_FULL" > "$BRIGHT_FULL.log" 2>&1 &
printf 'PID=%s\n结果=%s\n日志=%s\n' "$!" "$BRIGHT_FULL" "$BRIGHT_FULL.log"
```

打印进度与结果：

```bash
tail -n 40 "$BRIGHT_FULL.log"
"$BRIGHT_PY" "$BRIGHT_DIAG" --report --output "$BRIGHT_FULL"
```

日志每 10 张打印一次进度，每个步骤输出 `Step | Split | Path | mean loss (NLL + Lovasz) | foreground mIoU | class IoUs`，并打印四类混淆矩阵（行是真值，列是预测）。每张图像的损失、有效标签像素数和混淆矩阵也存放在新结果目录的 JSONL 中；所有原实验文件保持不变。若出错，查看新目录下的 `traceback.txt`，不要把部分结果记为完成。

解读重点是同一路径在同一批图像上的 500→1500→5000 变化：Train 损失持续下降而 Val 损失上升、IoU 下降，支持泛化差距；Train 和 Val 都恶化，更应查优化；损失改善但 Damaged IoU 下降，应查目标与类别取舍。单个种子、事后挑出的 Train450 和 500 步峰值都不足以证明稳定增益；后续还需预定短训练预算并重复种子。
