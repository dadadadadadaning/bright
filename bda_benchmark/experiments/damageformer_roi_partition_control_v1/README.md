# ROI-constrained partition controls V1

独立实验目录；不修改旧实验，不暂停/杀死已有任务，不覆盖任何现有输出。

## 四个实验的目的

| ID | 处理 | 问题 |
|---|---|---|
| s0 | 原Q | 同协议基准，等价于region_content的roi完整读出 |
| s1 | 一次Q*exp(-lambda*建筑概率差)后归一化 | 固定K下，柔性ROI相容约束是否有效 |
| s2 | 每个区域按预测B>=threshold拆为建筑/背景 | 硬隔离汇总是否有效 |
| c2 | 每个区域按seed格子x中线进行确定空间二分 | 是否只是更细分组有帮助 |

全部使用同一原图融合+Soft ROI特征，固定同一个V2前端快照、原Q与建筑概率。
只训练四个独立且同初始化的完整区域表示/分类器。没有Memory、GT分区、额外损失、
像素残差融合、Swin更新或Q生成器更新。前端仍内部运行旧调制支路用于原输出重放，
但四组读出不消费调制特征。不是节省前端计算的优化版本。

S1的建筑区域均值由原Q计算，重加权只做一次；lambda=0复现s0。
S2的注意力、mean/std及回投都使用2k+group；背景仍分类四类，不把低B位置强制判背景。
严格隔离指特征网格上的汇总/回投，不保证最终双线性上采样边界或编码器感受野隔离。
C2与S2同为最多2K槽位，不保证相同实际非空区域数，不可称为完全预算匹配。
不是实例分割；相邻建筑仍可共享同一建筑子组。

## 上传与运行

上传本目录全部文件到服务器项目的同名experiments目录。依赖服务器已有四阶段V2、
region_content_v1及其依赖；不需要新安装CUDA扩展。
必须使用与front_snapshot匹配的旧前端代码，源码哈希不符不可跳过。
不要使用A1、Memory或新读出权重作为FRONT_CKPT。

```bash
cd /root/shared-nvme/BRIGHT-master
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth
test -s "$FRONT_CKPT" || { echo '快照不存在，请先确认路径'; exit 1; }
export BATCH=2 ACCUM=8 WORKERS=4 DEVICE=cuda:0 SEED=3407
export LR=1e-4 SOFT_STRENGTH=1 ROI_THRESHOLD=0.5
export VARIANTS="s0 s1 s2 c2"

conda run --no-capture-output -p /root/shared-nvme/conda-envs/bright4090 \
  bash bda_benchmark/experiments/damageformer_roi_partition_control_v1/run.sh preflight

export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/ROI_PARTITION_CONTROL_V1/smoke_$(date +%Y%m%d_%H%M%S)"
conda run --no-capture-output -p /root/shared-nvme/conda-envs/bright4090 \
  bash bda_benchmark/experiments/damageformer_roi_partition_control_v1/run.sh smoke
```

smoke默认20优化器步，每头EBS16，Val前2张；看到完成标记与summary complete才完成，不能判性能。
正式训练前确认GPU空闲；不会自动暂停当前region_content实验。不支持DDP多卡。

```bash
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/ROI_PARTITION_CONTROL_V1/formal5000_seed3407_$(date +%Y%m%d_%H%M%S)"
export STEPS=5000 EVAL_EVERY=500
mkdir -p /root/shared-nvme/BRIGHT-train-logs/ROI_PARTITION_CONTROL_V1
export TRAIN_LOG="/root/shared-nvme/BRIGHT-train-logs/ROI_PARTITION_CONTROL_V1/$(basename "$OUTPUT_ROOT").log"
nohup conda run --no-capture-output -p /root/shared-nvme/conda-envs/bright4090 \
  bash bda_benchmark/experiments/damageformer_roi_partition_control_v1/run.sh train \
  > "$TRAIN_LOG" 2>&1 &
echo "PID=$!"
echo "结果=$OUTPUT_ROOT"
echo "日志=$TRAIN_LOG"
```

四头同一进程：每个batch只提取一次前端，各头顺序前向/反向，各有优化器。
每头5000更新、80000训练crops，共用同批次；不是4组平分5000步。
初始化完整Val450用于旧模型复现，所以训练日志出现step1前会先evaluate450。
AdamW lr1e-4 wd.005，100步warmup+cosine，clip1；所有头相同NLL+Lovasz。
如显存不足，可BATCH=1 ACCUM=16；不要改EBS或仅一组预算。不自动断点续训。

## 输出与报告

```bash
conda run --no-capture-output -p /root/shared-nvme/conda-envs/bright4090 \
  python -m bda_benchmark.experiments.damageformer_roi_partition_control_v1.report --root "$OUTPUT_ROOT"
```

保存每组best+last，总计8个**读出小权重**，而非8个761MiB前端；没有优化器快照、
完整前端复制或特征缓存。保留原FRONT_CKPT，读出不能单独恢复前端。
临时原子保存产生一个.tmp文件；结果JSON记录原快照SHA、原前端源码SHA、新实验SHA。
ROUTING每次Val报告allocated、soft/hard occupied、tiny mass、entropy和预测二分混合度。
predicted_binary_mixing_at_05使用固定0.5诊断阈值，不是GT纯度；当训练threshold改变时更不能据此宣称绝对隔离。
soft/hard实际数量可能不同，S2>C2后还要检查粒度/占用数差异。
每类混淆矩阵保存在validation.jsonl；完整配对rescue/harm、事件/尺寸审计本版未实现。
原old_pixel/old_region仅固定参考，不是同容量新像素头。不存在Test入口。

## 本地验证边界

提供8项CPU测试：S0旧完整读出前向/梯度等价、S1 lambda0/零权重、S2 batch隔离、
C2与B无关、同初始化与冻结梯度、全背景/全建筑/单点/边角、真实小尺寸前端重放与BN不变、
真实损失优化器更新与权重序列化。
还需要服务器真实数据CUDA smoke，不可用CPU测试替代真实性能验证。

## 结果解释

- S1>S0：相同区域预算的软ROI约束有正向证据。
- S2>S0：硬分组可用，但还须C2与实际区域数量审计。
- S2≈C2：不能把收益全部归因于建筑语义。
- 都无提升：当前特征/Q可能不需要更强隔离，不能继续以更强约束必定更好为前提。
- 有明确改善后重复种子，再考虑学习Q；本版所有Q变换均no_grad，没有训练分区网络。
