# 建筑条件的因子化 RGB–SAR Token 与 Memory V1

## 1. 这次真正改了什么

旧版容易把所有信息直接拼成一个向量，包括 P3 概率、几何、配准质量和 RGB–SAR 特征。这样即使检索很准，也无法判断 Memory 是发现了新的跨模态证据，还是复制了 P3 已有判断。

本实验把信息拆成四个职责不同的部分：

```text
内容 Key：RGB结构 + SAR证据 + RGB–SAR关系
几何信息：只允许用于弱重排序
质量信息：只留给以后做可靠性门控
P3状态：只留给以后做 Utility Verifier
```

T1–T3 的检索 Key 明确禁止使用 P3 最终概率、entropy、margin、碎片度、区域面积和绝对坐标。T0 是故意保留的“全部拼接”对照。

## 2. 区域如何产生

```text
灾前 RGB
  ↓
SLICO 局部区域
  ↓
用 P3 soft 建筑图把跨越建筑边界的区域再切开
  ↓
只保留建筑重叠充足、GT纯度≥80%的训练 Memory 条目
```

SLICO 只是局部证据容器，不是建筑实例，也不是损毁类别。最终类别仍是 Intact、Damaged、Destroyed。

## 3. T0–T3逐级证明什么

| 变体 | Key内容 | 要回答的问题 |
|---|---|---|
| T0 `t0_concat` | 所有特征直接拼接 | 高分是否只是复制P3状态或几何捷径 |
| T1 `t1_factorized` | RGB + SAR + 未对齐关系 | 独立跨模态Key是否可行 |
| T2 `t2_aligned` | RGB + SAR + 局部容错关系 | 局部对齐是否真的改善检索 |
| T3 `t3_multiring` | T2 + Core/Ring-8/Ring-16 | 核心—外围变化是否补充损毁证据 |

只有 T3 相对 T2 有稳定提升后，才建议启用已经预留但未接入流程的 `BuildingDeepSetEncoder`；只有裸检索对 P3 错误具有互补性后，才训练回写 Verifier。

## 4. Memory如何建立

每个“类别×事件”最多保存128个条目：

- 约70%是覆盖特征空间的代表性样本；
- 最多30%是P3判错但GT可靠的困难样本；
- 检索时每个来源事件最多占Top-k中的2个位置，避免单一事件垄断答案。

Memory Value保存三类软标签和来源信息。第一阶段只审计检索，不修改P3像素输出，也不访问Test。

## 5. 服务器运行

从 `bda_benchmark` 目录运行：

```bash
bash experiments/spi_factorized_cross_modal_token_v1/run.sh preflight
```

先只运行最必要的 T1→T2→T3，不必先跑T0：

```bash
mkdir -p /root/data/BRIGHT-train-logs/SPI_FACTORIZED_CROSS_MODAL_TOKEN_V1

for variant in t1_factorized t2_aligned t3_multiring; do
  nohup env STEPS=5000 \
    bash experiments/spi_factorized_cross_modal_token_v1/run.sh stage "$variant" \
    > "/root/data/BRIGHT-train-logs/SPI_FACTORIZED_CROSS_MODAL_TOKEN_V1/${variant}.log" 2>&1
done
```

上面是顺序执行；前两个最大特征缓存只会建立一次。查看某一项：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_FACTORIZED_CROSS_MODAL_TOKEN_V1/t2_aligned.log
```

如果需要完整对照，可运行：

```bash
nohup env STEPS=5000 \
  bash experiments/spi_factorized_cross_modal_token_v1/run.sh stage_all \
  > /root/data/BRIGHT-train-logs/SPI_FACTORIZED_CROSS_MODAL_TOKEN_V1/stage_all.log 2>&1 &
```

## 6. 判断标准

第一阶段不要看像素mIoU，因为还没有回写P3。重点比较：

1. Memory Macro-F1与Direct Head Macro-F1；
2. 正确Memory与随机标签Memory差距；
3. `p3_wrong_memory_correct`：P3错时Memory能救回多少；
4. `p3_correct_memory_wrong`：若无门控直接采用会伤害多少；
5. 排除同事件Memory后是否仍稳定；
6. 每事件、每类别的Recall/F1，尤其Noto与Ukraine。

建议准入：

```text
T2 > T1：局部容错关系有正贡献
T3 > T2：Core/Ring关系有正贡献
Memory > Direct Head：检索有额外价值
P3错误中的Memory正确覆盖 ≥ 25%
正确Memory显著高于随机Memory
跨事件结果不存在单事件灾难性崩溃
```

若这些条件不满足，不应进入像素回写和Test。

全部完成后生成统一对比：

```bash
STEPS=5000 bash experiments/spi_factorized_cross_modal_token_v1/run.sh summarize
```

## 7. T3 Direct与Memory互补性审计

T3完成后无需重新训练，直接复用Val缓存、T3 Key和T3 Memory：

```bash
STEPS=5000 \
bash experiments/spi_factorized_cross_modal_token_v1/run.sh audit_complementarity
```

主要结果：

```text
BRIGHT-eval-results/SPI_FACTORIZED_CROSS_MODAL_TOKEN_V1/
t3_direct_memory_complementarity_step5000.json
```

供后续Utility Verifier使用的逐Token明细：

```text
BRIGHT-checkpoints/SPI_FACTORIZED_CROSS_MODAL_TOKEN_V1/
t3_direct_memory_complementarity_detail_step5000.pt
```

重点看：

```text
direct_only_correct：P3错时只有Direct救回
memory_only_correct：P3错时只有Memory救回
both_correct：两者都能救回
neither_correct：两者都无法救回
memory_incremental_over_direct_rate：Direct失败后Memory还能补多少
oracle_union_recovery_rate：Direct与Memory理想联合的恢复上限
```
