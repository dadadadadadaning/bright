# Utility Gate 500步结果判定与下一阶段运行方案

## 结果判定

本次流程正常完成：方向规则拟合处理2627张，Gate完成500个optimizer steps，内部校准处理291张；Val450、Test和官方实例均未访问。

固定S2相对P3提高0.7050pp mIoU，Damaged F1提高0.7746pp，是本次结果中的最佳可部署候选。方向规则仅提高0.2770pp，明显弱于完整固定S2。Utility Gate结果与P3完全相同，说明阈值0.5下几乎没有S2修改被接受。

训练日志显示changed区域平均Gate值从约0.12升至约0.285；这是模型在学习，但不足以跨过0.5决策阈值。更重要的是，selector AUROC未达到0.60，且事件稳定、前景安全和超过fixed S2均失败，所以不能把问题简单解释为阈值太高，也不能访问Val450。

`requires_grad=True`转float警告只发生在日志记录，未改变loss或梯度；本地代码已增加`.detach()`消除警告。

## 正式决策

1. 把S3基础Utility Gate登记为REJECT；
2. 不运行S4 Memory、S5 IM4CD/Relation、S6 RGID、S7超像素、S8联合；
3. 不把500步模型延长到3000步；
4. fixed S2保留为当前选择性上下文主候选；
5. 独立开始SAR-FSA与SAR-RCM编码器pilot。

如果需要研究Gate失败原因，只允许进行一次标记为post-hoc的阈值诊断，不能把结果当作预注册证据。先读取JSON中的AUROC、accepted fraction和正负样本均值；若AUROC明显低于0.60，停止阈值扫描。

实际完整统计为：changed均值0.31694、accepted fraction为0、S2-only-correct均值0.31403、P3-only-correct均值0.32438、AUROC 0.40899、AP 0.49383。代码检查确认正负标签没有写反：高分的语义确实是接受S2。将分数事后取反的理论AUROC约为0.59101，仍低于0.60。因此不再进行阈值扫描或取反部署。

固定S2共有699,348个修改，其中349,822个修正、280,522个伤害、69,004个中性，净修正69,300。方向规则只保留162,944个修改，伤害从280,522降至48,414，但修正也从349,822降至91,928；它提高了决定性精度，却丢失了大部分有效覆盖，并且只在2/14事件上不差于fixed S2。它可以作为安全性诊断对照，不能替代fixed S2主结果。

## 登记S1-S3

按照`experiments/spi_s2_evidence_ladder_v1/README_ZH.md`执行S1、S2、S3登记命令，然后运行：

```bash
bash experiments/spi_s2_evidence_ladder_v1/run.sh status
```

预期S3为REJECT，S4-S8不具备合法父结果。

## SAR适配器pilot

新增`experiments/spi_sar_adapter_v1`，以同一个P3 checkpoint和Train2918内部划分运行，和Utility Gate完全隔离。

```bash
cd /root/data/BRIGHT-master/bda_benchmark
bash experiments/spi_sar_adapter_v1/run.sh preflight

nohup env NUM_WORKERS=4 BATCH_SIZE=4 ACCUMULATION_STEPS=4 PILOT_STEPS=500 \
  bash experiments/spi_sar_adapter_v1/run.sh pilot \
  > /root/data/BRIGHT-train-logs/SPI_SAR_ADAPTER_V1_seed3407.log 2>&1 &
```

该命令依次运行FSA和RCM，分别审计，不进行组合，并且不访问Val450。只有单模块满足全部准入条件，才允许做完整训练；两者都失败则停止SAR适配器路线。
