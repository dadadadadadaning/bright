# S2 修改簇级跨模态证据与效用验证器

## 1. 为什么做这一项实验

已有 Cluster Oracle 表明：按“同一 LocCC 内、同一 P3→S2 修改方向、8 连通”的修改簇选择 P3 或 S2，相对 Building Oracle 仍有 `+0.2225pp` mIoU，且 14/14 事件不下降。说明修改簇是有意义的选择单位。

SMP-E0 失败只说明把多个修改簇再次汇总为整栋建筑的面积、位置和概率统计无法预测效用。它没有检验每个修改簇自身的 RGB/SAR 表征、局部匹配和关系变化。

本实验回答：

> 在不看 GT 的推理阶段，冻结 P3 后提取的簇级 RGB/SAR 证据，能否区分 S2 的有益修改与有害修改？

## 2. 完整数据流

```text
冻结 P3
  ├─ RGB encoder Stage3/4 ─┐
  ├─ SAR encoder Stage3/4 ─┤
  └─ P3 → fixed S2 → 修改簇
                           │
          修改区 / 稳定锚点 / 局部边界 三类角色
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
 RGB/SAR角色表征       IM4CD式局部匹配       模态内关系差
      │                    │                    │
      └────────────────────┴────────────────────┘
                           │
                 E0 无参数、单通道审计
                           │
                  证据是否跨事件通过？
                    ┌──────┴──────┐
                    否             是
                    │              │
                   STOP       E1 两层小验证器
                                   │
                         接受该簇S2 / 回退P3
```

修改簇不是超像素，也不是重新分割建筑。它只由 P3 与 fixed S2 的实际差异形成，因此最终验证器不能修改未被 S2 改动的像素，也不能生成 P3/S2 之外的新类别。

## 3. 特征组与因果对照

| 组 | 内容 | 回答的问题 |
|---|---|---|
| spatial | 面积、位置、方向、P3 margin/entropy、S2 概率变化 | 简单统计基线 |
| rgb_role | RGB Stage3/4 的修改区、锚点、边界差异 | 灾前结构是否有用 |
| sar_role | SAR Stage3/4 的修改区、锚点、边界差异 | 灾后响应是否有用 |
| direct | 同位 RGB/SAR 直接特征差 | 朴素跨模态基线 |
| matching | 5×5 局部匹配残差与置信度 | 配准残差下的局部对应是否有用 |
| relation | 两个模态各自的邻域关系差 | 跨模态结构变化是否有用 |
| shuffled_* | 同建筑内等面积随机掩膜 | 收益是否真的来自修改簇位置 |

Stage3/4 先做固定的 384/768→32 通道分组平均，不训练投影层。这既降低局部匹配开销，又防止 E0 通过新增容量拟合标签。Stage3 是预注册主证据；Stage4 以独立的 `*_stage4` 组报告，只作诊断，不与 Stage3 拼接，也不能单独触发 E1 准入。

## 4. E0 准入规则

每个证据组采用 leave-one-event-out：评估事件 E 时，只能在排除 E 的 `meta_train` 样本上选择一个标量通道和正负方向，再冻结到 E 的 `meta_calibration` 样本。不同 P3→S2 修改方向允许使用各自通道，但样本不足时必须回退全局规则。

主证据需满足：

```text
Calibration AUROC             >= 0.60
Event-macro AUROC             >= 0.58
正确方向事件                  >= 10/14
正、负效用修改簇              各 >= 200
Permutation p                 < 0.05
相对预注册控制增益             >= 0.02 AUC
```

其中 `matching/relation` 必须同时超过 direct 和对应 shuffled 对照；单模态角色特征必须超过 spatial。只有 matching/relation 至少一个通过，或 RGB 与 SAR 两个角色组同时通过，才允许 E1。

## 5. E1 验证器

E1 输入仅包含 `spatial + E0通过的证据组`，网络为：

```text
Linear → GELU → LayerNorm → Linear(1)
```

默认隐藏维度 32，训练 1000 个优化步，每步随机采样 4096 个修改簇。损失使用类别平衡、事件平衡和 `sqrt(changed_pixels)` 效用权重。

验证器只执行：

```text
score >= threshold：接受这个修改簇的 S2
score <  threshold：这个修改簇回退 P3
```

阈值只能在 `meta_calibration` 选择，并限制最多拒绝 60% 的 fixed S2 修改簇，防止选择器通过全部回退 P3 获得虚假安全性。

E1 校准准入：相对 fixed S2 `>= +0.30pp` mIoU、Damaged F1 不下降、至少 10 个事件不下降。通过后才冻结模型并允许一次性 `meta_confirm`。

## 6. 数据隔离

```text
meta_train       选择E0通道、训练E1
meta_calibration E0准入、选择E1阈值
meta_confirm     E1完全冻结后只评估一次
Val450/Test      本实验不访问
```

`confirm` 会拒绝覆盖既有结果，避免反复查看 confirm 后调参。

## 7. 服务器运行

先同步 `SERVER_SYNC_FILES.txt` 中的文件，然后：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_s2_cluster_evidence_v1/run.sh preflight
bash experiments/spi_s2_cluster_evidence_v1/run.sh smoke
```

如果日志出现 `unknown cluster evidence groups: ['0']`，说明外部 shell
残留了 `EVIDENCE_GROUPS=0`。新版入口会自动恢复默认六组；旧版入口可在运行前
执行 `unset EVIDENCE_GROUPS`，或者显式设置完整的证据组列表。

完整 E0：

```bash
nohup env \
PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark \
NUM_WORKERS=1 FEATURE_CHANNELS=32 MATCHING_WINDOW=5 PROFILE_TIMING=1 \
PROGRESS_EVERY=10 STATE_EVERY=50 PERMUTATIONS=1000 SEED=7341 \
bash experiments/spi_s2_cluster_evidence_v1/run.sh audit \
> /root/data/BRIGHT-train-logs/SPI_S2_CLUSTER_EVIDENCE_E0_seed7341.log 2>&1 &
```

查看结果：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_S2_CLUSTER_EVIDENCE_E0_seed7341.log
grep -E 'Cluster Evidence cache|Cluster-E0|Admitted:|Decision:|Saved:' \
/root/data/BRIGHT-train-logs/SPI_S2_CLUSTER_EVIDENCE_E0_seed7341.log
```

仅当 E0 输出 `ADMIT_CLUSTER_E1_SINGLE_VERIFIER` 时训练 E1：

```bash
nohup env \
PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark \
VERIFIER_STEPS=1000 VERIFIER_BATCH_SIZE=4096 VERIFIER_HIDDEN=32 SEED=7341 \
bash experiments/spi_s2_cluster_evidence_v1/run.sh train \
> /root/data/BRIGHT-train-logs/SPI_S2_CLUSTER_EVIDENCE_E1_seed7341.log 2>&1 &
```

仅当 E1 输出 `FREEZE_CLUSTER_E1_FOR_META_CONFIRM` 时，执行一次确认：

```bash
nohup env \
PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark \
NUM_WORKERS=1 BOOTSTRAP_ITERATIONS=10000 SEED=7341 \
bash experiments/spi_s2_cluster_evidence_v1/run.sh confirm \
> /root/data/BRIGHT-train-logs/SPI_S2_CLUSTER_EVIDENCE_CONFIRM_seed7341.log 2>&1 &
```

## 8. Memory 的启用条件

缓存中的 `memory_key` 保存 RGB/SAR Stage3/4 的“修改区减锚点、修改区减边界”向量，供后续跨事件检索使用。但本实验不检索。

只有 E1 在 `meta_confirm` 通过，或至少一个 E0 表征组稳定通过但小验证器仍不足时，才开展 Cluster-Memory M0：查询事件必须从 Memory 排除，先比较 kNN 与 shuffled-memory，再决定是否引入检索增强分类器。若 E0 完全没有信号，Memory 只会检索更多不可分样本，应停止，而不是扩大库。
