# P3-HR 分层 Occurrence--Severity 受控实验

## 隔离边界

- 不修改或覆盖 P3、OABH、S2O、Memory 代码和 checkpoint；
- P3 编码器、定位、融合及四类头全部冻结；
- HR0/HR1/HR2 没有 Test 入口；
- 所有新 checkpoint 只保存层级模块的增量参数。

## 变体

```text
HR0：零残差，逐元素复现P3
HR1：P3 fused_roi -> Occurrence residual + Severity residual
HR2：Stage2+3 Relation-FPN -> Occurrence residual
     P3 fused_roi          -> Severity residual
```

最终保持 P3 Background logit和总 Building 概率质量，软分解为：

```text
P(Intact|B)    = 1 - P(Affected|B)
P(Damaged|B)   = P(Affected|B) * (1-P(Destroyed|Affected))
P(Destroyed|B) = P(Affected|B) * P(Destroyed|Affected)
```

## 同步文件

```text
bda_benchmark/model/DamageFormer_P3_HierarchicalResidual.py
bda_benchmark/script/standard_ML/p3_hierarchical_residual_runtime.py
bda_benchmark/script/standard_ML/train_P3_HierarchicalResidual.py
bda_benchmark/run/run_P3_HierarchicalResidual.sh
bda_benchmark/tests/test_p3_hierarchical_residual.py
```

## 服务器运行

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_P3_HierarchicalResidual.sh preflight
```

先运行 HR0：

```bash
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/P3_HIERARCHICAL_RESIDUAL
mkdir -p "$LOG_ROOT"

bash run/run_P3_HierarchicalResidual.sh guard \
  2>&1 | tee "$LOG_ROOT/P3_HR0_guard_seed3407.log"
```

必须看到：

```text
Decision: ADMIT_TO_P3_HR1_500_STEP_SCREEN
Test accessed: False
```

再运行 HR1：

```bash
LOG="$LOG_ROOT/P3_HR1_screen500_seed3407.log"
nohup env SEED=3407 \
  bash run/run_P3_HierarchicalResidual.sh hr1 \
  > "$LOG" 2>&1 &
echo "PID: $!"
echo "LOG: $LOG"
tail -F "$LOG"
```

HR1 是必须完成的“没有新证据”对照。无论通过还是失败，都要先记录结论，然后
才运行预先声明的 HR2：

```text
HR1_COMPLETE_NO_GAIN_PROCEED_TO_PREDECLARED_HR2_NO_TEST
或
HR1_PASSED_PROCEED_TO_PREDECLARED_HR2_NO_TEST
```

才运行 HR2：

```bash
LOG="$LOG_ROOT/P3_HR2_screen500_seed3407.log"
nohup env SEED=3407 \
  bash run/run_P3_HierarchicalResidual.sh hr2 \
  > "$LOG" 2>&1 &
echo "PID: $!"
echo "LOG: $LOG"
tail -F "$LOG"
```

不要同时运行 HR1 与 HR2，也不要跳过 HR1。HR1 失败并不禁止 HR2，因为 HR2
预先声明的唯一新增变量正是 Stage2+3 新证据；但 HR2 必须独立从 P3 初始化，
不能加载 HR1 权重。只有 HR2 自身通过完整 Gate，才允许延长训练。
