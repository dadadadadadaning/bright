# SPI S2 Evidence V2

## 目的

在不修改P3、不访问Val450的条件下，判断固定S2的修改能否由以下证据区分为corrected或harmed：

- E1：模态内Relation Difference；
- E2：IM4CD式局部硬匹配；
- E3：真实固定RGB-SLIC与等预算Grid区域统计。

基础Utility Gate、FSA和RCM的参数均不进入本实验。

## 数据协议

- `gate_train.txt`：只用于选择标量通道/方向和训练Verifier；
- `gate_calibration.txt`：只用于内部准入；
- Val450、Test、官方实例：禁止；
- Train缓存每图、每类最多抽取2048个decisive修改；
- Calibration缓存保存全部changed修改，并保存可逆混淆矩阵信息。

注意：旧291张已经用于多次探索，本实验属于development。若产生候选方法，正式确认应另外冻结一个未查看的Train2918内部确认划分，然后才能考虑一次Val450。

缓存会复用以避免重复跑冻结P3。如果修改`EVIDENCE_GROUPS`、`MATCHING_MAX_SIDE`、`MATCHING_WINDOW`或`CELL_SIZE`，必须指定一个新的`CACHE_ROOT`，不能覆盖或混用旧缓存。

脚本内部不能使用Bash特殊变量`GROUPS`（root用户时其值为Unix组ID `0`）；当前版本使用`EVIDENCE_GROUP_LIST`保存解析后的证据组。若服务器报`unknown evidence groups: ['0']`，说明仍是旧版`run.sh`。

旧实验若在当前终端留下`EVIDENCE_GROUPS=0/1`，入口会自动解释为完整默认组，而不会把`0/1`当作证据名称。仍建议启动前执行`unset EVIDENCE_GROUPS`，保持实验环境清晰。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
bash experiments/spi_s2_evidence_v2/run.sh preflight
bash experiments/spi_s2_evidence_v2/run.sh smoke

nohup env NUM_WORKERS=4 MATCHING_MAX_SIDE=80 CELL_SIZE=4 \
  bash experiments/spi_s2_evidence_v2/run.sh audit \
  > /root/data/BRIGHT-train-logs/SPI_S2_EVIDENCE_V2_AUDIT_seed3407.log 2>&1 &
```

查看进度：

```bash
grep -E 'Evidence cache|E0 P3|AUC=|Checks:|Admitted:|Decision:|Saved:' \
  /root/data/BRIGHT-train-logs/SPI_S2_EVIDENCE_V2_AUDIT_seed3407.log
```

`audit`只做一次冻结P3前向，但分别保存并审计E1/E2/E3；它们不是串联模块。IM4CD局部匹配和CPU-SLIC会明显降低缓存速度，所以必须先通过10张真实数据的`smoke`。

只有`Admitted`中出现的单个组才能训练。例如：

```bash
nohup env VERIFIER_STEPS=500 VERIFIER_BATCH_SIZE=8192 \
  bash experiments/spi_s2_evidence_v2/run.sh relation \
  > /root/data/BRIGHT-train-logs/SPI_S2_RELATION_VERIFIER_V2_seed3407.log 2>&1 &
```

如果`relation`未被准入，训练脚本会主动报错并停止。`matching`和`slic`同理。禁止一次传入多个证据组。

## 审计结束后自动续跑

当前审计已经用`nohup`启动时，可以再启动一个轻量监控脚本。默认`RUN_POLICY=best`：按照无参数内部准入AUROC和事件宏AUROC，只训练排名第一的单一证据Verifier：

```bash
nohup env POLL_SECONDS=60 RUN_POLICY=best VERIFIER_STEPS=500 \
  bash experiments/spi_s2_evidence_v2/watch_then_run.sh 118662 \
  > /root/data/BRIGHT-train-logs/SPI_S2_EVIDENCE_V2_MONITOR_seed3407.log 2>&1 &
```

若需要完成论文用的独立分支对照，可使用`RUN_POLICY=all_admitted`。它会把所有通过无参数门槛的组逐个训练，但每次仍只有一个附加证据，不做联合：

```bash
nohup env POLL_SECONDS=60 RUN_POLICY=all_admitted VERIFIER_STEPS=500 \
  bash experiments/spi_s2_evidence_v2/watch_then_run.sh 118662 \
  > /root/data/BRIGHT-train-logs/SPI_S2_EVIDENCE_V2_MONITOR_seed3407.log 2>&1 &
```

监控脚本只有在审计日志存在成功结束标记、准入JSON协议正确且未访问禁用数据时才会继续。如果没有组通过会安全停止。即使两个分支独立通过，也只标记具备手工设计S8的资格，不会自动联合训练或访问Val450。

## 准入

Relation/Matching必须同时满足：AUROC≥0.60、事件宏AUROC≥0.58、至少10个事件方向正确、超过直接RGB-SAR差至少0.02 AUROC。

SLIC必须同时满足：AUROC≥0.60、事件宏AUROC≥0.58、至少10个事件方向正确、超过等请求节点预算Grid至少0.01 AUROC。

Verifier必须超过fixed S2至少0.10pp mIoU、减少至少10% harmed、AUROC≥0.60、至少60%事件不低于fixed S2、无前景IoU下降超过0.50pp，并且bootstrap下界大于0。

## FSA后续方案

FSA当前融合后适配结果约等于P3，不能直接延长训练。只保留三个新假设：

1. Stage3-SAR FSA：先对冻结Stage3做频率统计探针，探针通过才训练；
2. FSA+damage-head：保持编码器冻结，只解冻`fusion_layer_3/clf_2`的0.1倍学习率，用于判断冻结读出是否造成无效；
3. Building-masked FSA：只在P3建筑支持和高不确定区域应用，避免全场景纹理扰动。

三者必须分别预注册，不能与Evidence V2同时训练，也不能因为FSA安全无下降就宣称有效。
