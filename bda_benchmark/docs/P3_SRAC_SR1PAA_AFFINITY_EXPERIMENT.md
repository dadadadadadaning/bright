# P3-SR1PAA：建筑条件损毁亲和度实验

## 实验问题

SR1PAA只回答一个问题：冻结P3的建筑几何后，冻结H6-A2原生SAR的
Stage2/3能否判断相邻位置是否属于同一损毁区域。

它不修改P3分类、不生成检索记忆、不进行区域传播，也不访问测试集。

## 前向路径

```text
灾前RGB ──> 冻结P3 ──> RGB特征 + 建筑概率 + 原始P3预测
灾后SAR第1通道 ──> 冻结H6-A2 sar_encoder ──> S2 + S3

RGB特征、S2、S3、建筑概率、RGB边缘
                  ↓
            32维局部上下文
                  ↓
每个像素与8邻居构造：
[中心, 邻居, |中心-邻居|, 中心×邻居, 两侧建筑概率, 两侧RGB边缘]
                  ↓
             共享Pair Head
                  ↓
8通道Affinity Logits（同一损毁区域=1，不同损毁区域=0）

最终输出始终等于冻结P3。
```

## 训练监督

训练标签只用于构造建筑内部的邻接关系：

- 两个位置均为建筑且损毁等级相同：正样本；
- 两个位置均为建筑但损毁等级不同：负样本；
- Background或Ignore参与的关系：不计算内部亲和度损失。

正负关系分别计算BCE后等权平均，避免大量同类邻接淹没稀少的损毁边界。

## 验证门槛

只有同时满足以下条件，才允许开发“亲和度构图/区域传播”阶段：

- Affinity loss相对随机基线至少下降0.02；
- Balanced accuracy不低于0.65；
- Damaged边界召回不低于0.60；
- Same/Different概率间隔不低于0.15；
- 验证集同时包含正关系、负关系和Damaged边界关系。

门槛失败时保持P3不变，不运行传播、检索或Test。

## 服务器运行

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

export H6_A2_CKPT=/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth

VARIANT=sr1paa bash run/run_P3_SRAC_controlled_5090.sh preflight

LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/P3_SRAC_CONTROLLED
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/P3_SR1PAA_E3A_seed3407_v1.log"

nohup env \
  H6_A2_CKPT="$H6_A2_CKPT" \
  SEED=3407 \
  RUN_TAG=v1 \
  bash run/run_P3_SRAC_sequence.sh e3a \
  > "$LOG" 2>&1 &

tail -F "$LOG"
```

最终准入报告位于：

```text
/root/shared-nvme/BRIGHT-checkpoints/P3_SRAC_CONTROLLED/
P3_SR1PAA_affinity500_seed3407_v1/affinity_gate.json
```
