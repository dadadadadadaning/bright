# 事件鲁棒修改簇验证器：隔离主实验

## 为什么先做这个实验

“事件鲁棒P3 + 多强度S2 + 多动作Verifier”一次改变太多因素，无法知道提升来自哪里；其中`弱S2`也尚无Oracle证据。因此本目录只验证最小命题：

> 在冻结同一个P3、同一个fixed S2和同一套Metric Utility特征时，事件均衡、GroupDRO或V-REx能否让二动作线性Verifier迁移到未参与Verifier训练和阈值选择的完整事件？

它是Verifier级LOEO，不是端到端未见事件证明，因为冻结P3可能见过外层留出事件。

## 四个唯一变量

```text
G0: 普通簇级ERM
G1: Event-balanced ERM
G2: GroupDRO
G3: V-REx
```

特征、标签、模型容量、训练步数、阈值集合和最大拒绝比例完全一致。事件鲁棒目标只作用于训练损失。

## 外层事件隔离

每个fold选择一个完整事件`E`：

1. `meta_train`中删除事件`E`后训练Verifier；
2. `meta_calibration`中删除事件`E`后选择一个全局阈值；
3. 仅在`meta_calibration`的事件`E`上测试；
4. 轮流留出全部事件，组合成LOEO结果。

因此留出事件不会参与该fold的Verifier拟合、标准化统计或阈值选择。

## 数据隔离

- 读取：D1的`meta_train.pt`、`meta_calibration.pt`和MACU指标标签；
- 不读取：`meta_confirm`、Val450、官方实例、Official Test；
- 输出明确记录`P3 exposure to the held-out event is not controlled`，防止把Verifier级结果误写为端到端跨事件结果。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
bash experiments/spi_event_robust_cluster_verifier_v1/run.sh preflight

nohup env DEVICE=cuda:0 STEPS=1000 SEED=7341 \
bash experiments/spi_event_robust_cluster_verifier_v1/run.sh loeo \
> /root/data/BRIGHT-train-logs/SPI_EVENT_ROBUST_CLUSTER_VERIFIER_V1_seed7341.log 2>&1 &
```

第一轮不要调`groupdro_eta`或`vrex_lambda`。只有默认对照显示正方向但优化不稳，才允许在相同LOEO协议内预注册极小搜索范围。

## 准入

相对fixed S2要求：mIoU≥+0.30pp、Metric Oracle恢复≥25%、事件bootstrap CI下界>0、Damaged F1不下降、至少10个留出事件不下降、拒绝率≤70%。全部满足才建议冻结唯一目标进入尚未访问的`meta_confirm`。

若G1不优于G0，停止把“事件数量偏置”写成主要机制。若G2/G3不优于G1，保留简单事件均衡，不把GroupDRO/V-REx写入最终模型。若所有方案失败，说明当前特征而不是训练目标更可能是瓶颈。
