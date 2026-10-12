# H7 第一阶段：P3–A2 互补性与 Oracle 上限审计

## 目的

该阶段不训练新网络，也不访问测试集。它只回答一个问题：

> H6-A2 原生 SAR 专家是否包含足够多、足够稳定的 P3 纠错信息，值得继续实现 H7-CER 路由？

如果没有先证明互补上限，直接增加门控、Cross-Attention 或外部 SAR
主干，很难判断收益来自真正的 SAR 证据还是验证集随机波动。

## 严格输入协议

同一个 DataLoader 同时为两个模型提供同一张 RGB、SAR 和标签，保证空间变换完全一致。

- P3：使用未裁剪的显示 SAR，复制为三通道，再执行历史 ImageNet 归一化。
- H6-A2：使用训练集统计裁剪、归一化后的原生单通道 SAR。
- P3 权重通过固定 SHA256 和 `strict=True` 校验。
- A2 必须声明 `variant=a2`，并通过 `strict=True` 加载。
- 脚本只接受验证集参数，没有测试集入口。

## 输出

完整 JSON 包括：

1. P3、A2、标签感知 Oracle 的四类和有效三类指标；
2. P3/A2 都正确、仅 P3 正确、仅 A2 正确、都错误的像素数量；
3. 按背景、完好、受损、摧毁拆分的互补性；
4. 伪建筑级互补性；
5. 每个事件的 P3、A2、Oracle 指标和提升；
6. P3 与 A2 的置信度校准误差；
7. 自动决策结果。

这里的“伪建筑”是对每个真值损毁类别分别执行四连通域得到的近似实例。同类相邻建筑可能仍然合并，因此该统计用于判断趋势，不能替代官方实例标注。

## 自动决策门槛

默认同时满足下列条件才建议用当前 A2 实现 H7-CER：

- Oracle 有效三类 mIoU 相对 P3 提升不少于 0.5 pp；
- Oracle Damaged IoU 相对 P3 提升不少于 1.0 pp；
- 至少 50% 的事件获得正的 Oracle 有效三类 mIoU 收益。

通过时输出：

```text
IMPLEMENT_H7_CER_WITH_CURRENT_A2
```

未通过时输出：

```text
UPGRADE_SAR_EXPERT_BEFORE_ROUTING
```

后者表示应先进行 H7-S0/S1/S2 的 SAR 编码器筛选，而不是继续调 B1R 门控。

## 服务器运行顺序

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_H7_P3_A2_oracle.sh preflight
bash run/run_H7_P3_A2_oracle.sh smoke
bash run/run_H7_P3_A2_oracle.sh val
bash run/run_H7_P3_A2_oracle.sh report
```

结果默认写入：

```text
/root/shared-nvme/BRIGHT-eval-results/H7_P3_A2_oracle/
```

其中：

```text
P3_A2_oracle_validation_full.json
P3_A2_oracle_validation_full.events.tsv
P3_A2_event_disaster_comparison.json
P3_A2_event_comparison.tsv
P3_A2_disaster_type_comparison.tsv
```

`report` 不运行模型，可直接读取已有 Oracle JSON。新版 `val` 完成后也会自动调用它。

新版 Oracle JSON 还保存 `[真值类别, P3预测, A2预测]` 三维转移计数。这可以判断
具体哪一种修改方向真正有用，例如 `P3=intact -> A2=damaged` 是净纠错还是净破坏，
从而在实现 H7-CER 前锁定允许的类别转移，而不是再次使用全类别共享门控。

## 结果解释

- `a2_only_correct`：A2 能纠正 P3 的位置，是路由的正收益候选。
- `p3_only_correct`：采用 A2 会破坏 P3 的位置，是路由必须避开的风险。
- `a2_advantage_among_disagreements`：两专家不一致时 A2 实际正确的比例。
- `oracle`：知道真值后再选择专家的理论上限，不能作为可部署模型。
- `ECE`：置信度与真实正确率的偏差；如果 A2 的 ECE 很高，普通置信度门控不可靠。

所有 H7 结构、超参数和 checkpoint 仍然只由训练集/验证集决定。测试集在最终锁定候选后只运行一次。
