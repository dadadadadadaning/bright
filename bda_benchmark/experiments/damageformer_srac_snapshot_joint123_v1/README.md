# 中途快照上的 Memory / J1 / J2 / J3

这是新隔离协议，不是“50,000 步父流程已完成”。原实验、旧完成检查、模型公式、快照文件均不修改。
显式允许从已审计的 formal Train/Val 中途快照出发，冻结 A1、Q、适配器和区域 Embedding，重新建库并训练联合分类器。
记录 `parent_checkpoint_step`、`declared_parent_steps`、`parent_run_complete`，不伪造任何完成标记。
新产物使用不同格式且 `reusable_as_parent=False`，不会被旧完成父流程接收。

## 实验问题

| 组 | 输入 | 回答的问题 |
|---|---|---|
| J1 | A1 像素证据 + 当前区域 + 可学习控制槽位 | 不查历史区域时，新联合头达到什么水平？ |
| J2 | 相同像素/区域 + 余弦检索的 Train 支持区域 | J2−J1：实际历史区域是否有增益？ |
| J3 | 相同像素/区域 + 学习检索投影选出的 Train 支持区域 | J3−J2：仅改变邻居选择是否有增益？ |

J2/J3 使用同一个 Reader 结构、相同 bank 和支持标签；区别是选出的邻居 ID。前端/Q 不继续学习。
不是把最终 Memory 结果强行加到 A1 分类结果；三组从 A1 等价初始化开始，训练像素证据与上下文的联合分类器。
J1/J2/J3 在一个 Python 进程内各自独立优化。

## 与旧流程的差异

仅修改产物准入协议：接受带 `snapshot.json` 且严格 SHA 一致的正式中途快照。
模型、跨事件 Top-8 检索、区域回投、损失和 J1/J2/J3 预算沿用旧实验。
选择已审计的 13,000 步快照，不等于使用暂停时尚未保存的内存权重，也不是重新训练 A1。
若采用不同快照，必须新建 Memory、投影和全部头，不能混用。

与当前 selective-readout 的 P/Q-read/Content-read 是不同问题和不同训练协议：
本实验为确定性 1024 整图、batch=1、无增强、每组 5,000 次优化器更新，不能把两个实验分数之差直接称为 Memory 增益。

## 检查与保存

- 父快照、A1、源码、Train/Val 划分、bank、metric、probe 均绑定哈希。
- 仅 Train2918 建库；Train 标签形成区域支持分布，不输入验证查询的 Q 或检索选择。
- bank 采用原事件×主类别随机优先级抽样，每桶最多 64，未新增过滤或调参。
- J3 使用同一 bank 中纯度 >=0.8 的 Train 区域拟合 128×128 检索投影，固定 2,000 步，不用 Val 选投影。
- 若某类不具备两个事件的可用支持，明确失败，不自动降纯度或制造支持。
- 初始化检查三头 A1 恒等；正式初始/每500步完整 Val450 核验 A1/direct_region，误差门槛 1e-5。
- 原逐像素恒等检查 1e-6 与精确混淆矩阵检查保留，不因失败自动放宽。
- 前端、bank、投影均检查冻结指纹；原边界审计的异常不因此被认定修复。
- 仅保存头的 best/last（共6份）和一个 metric_last。含头优化器，但无精确断点续训入口。
- Train 缓存有2918个FP16 Token文件，典型1024区域/128维配置约1GB量级，另有bank和分类头。A1与快照只引用，不复制。
- 输出目录存在即拒绝覆盖；不删除旧结果、不停止任何其他进程。

## 上传

将本目录整个上传到：

```text
/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_snapshot_joint123_v1
```

依赖原项目已有 `srac_transformer_v1`、`srac_joint123_v1`、`srac_qmetric_v1` 等，不应删除它们。

## 暂停父实验：保留进程、内存和显存

在服务器运行。只针对核对过的父 Python；不是 SIGTERM，不删除权重。
DataLoader 子进程可能短暂预取至队列填满。进程/服务器重启后不能用 SIGCONT 恢复。

```bash
PARENT_PID=1964323
PARENT_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_TRANSFORMER_V1/formal50000_seed3407_20261008_173616/front
PARENT_COMMAND="$(ps -ww -p "$PARENT_PID" -o args= || true)"
if [[ "$PARENT_COMMAND" == *" -m bda_benchmark.experiments.damageformer_srac_transformer_v1.train_front "* \
   && "$PARENT_COMMAND" == *" --output $PARENT_OUTPUT "* ]]; then
    kill -STOP "$PARENT_PID"
    ps -p "$PARENT_PID" -o pid,stat,etime,args
else
    echo "父进程不存在或身份变化，未发送信号。"
fi
```

`STAT` 含 `T` 表示暂停，仍占用显存。恢复时重新核对相同模块/输出/PID，把上面的 `-STOP` 换为 `-CONT`。
若需要释放显存而结束父任务，应另外决定；当前代码不承诺从最新内存状态恢复。

## 一次启动完整快照流水线

不要复用旧 `PARENT_RUN` 的入口，也不要把新目录创建在父实验或快照目录里面。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export SNAPSHOT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_CAUSE_AUDIT_V1/diag_20261009_100438/snapshot/front_snapshot.pth
export DEVICE=cuda:0
export WORKERS=4
export SEED=3407
export SNAP_J123_AMP=none
export SNAP_RUN_TAG="$(date +%Y%m%d_%H%M%S)"
export SNAP_J123_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_SNAPSHOT_JOINT123_V1/run_${SNAP_RUN_TAG}"
export SNAP_J123_SCRIPT=/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_snapshot_joint123_v1/run.sh
mkdir -p /root/shared-nvme/BRIGHT-train-logs/SRAC_SNAPSHOT_JOINT123_V1
export SNAP_J123_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_SNAPSHOT_JOINT123_V1/run_${SNAP_RUN_TAG}.log"

nohup bash -c '
set -euo pipefail
test -x "$PYTHON"
test -s "$SNAP_J123_SCRIPT"
bash "$SNAP_J123_SCRIPT" preflight
bash "$SNAP_J123_SCRIPT" pipeline
' > "$SNAP_J123_LOG" 2>&1 < /dev/null &

echo "启动PID=$!"
echo "结果=$SNAP_J123_ROOT"
echo "日志=$SNAP_J123_LOG"
tail -F "$SNAP_J123_LOG"
```

顺序：CPU检查 → 4图建库smoke → Train2918完整bank → metric2000 → J123真实20步probe → J123正式各5000步。
probe 使用正式bank/metric，正式训练重新初始化头，不从20步probe续训。任一步失败后续立即停止。
Memory建库是提取/保存，不是再训练前端。J3额外有2000步Train检索投影训练。

成功后：

```bash
bash "$SNAP_J123_SCRIPT" report
```

分阶段入口：`check` / `smoke_memory` / `memory` / `metric` / `probe` / `train`。
若某阶段失败，先读日志；已完成阶段可核验后复用，失败目录保留，不能盲目重跑覆盖，也不要伪造完成标记。
`train` 要求同一快照/bank/metric/source的probe完整通过。

## 验证边界

本地17项CPU合约和入口/语法检查通过；另以合成输入核验建库和检索投影梯度。
尚未在真实BRIGHT数据或服务器CUDA上验证。smoke仅验证执行，不代表有效提升。
最终看同一步 J2−J1、J3−J2、Damaged P/R/F1 和错误转移，不跨协议挑最佳分数作因果结论。
