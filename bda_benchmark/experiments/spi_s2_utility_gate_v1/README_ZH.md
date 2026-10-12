# SPI S2 Utility Gate V1

## 为什么做这个实验

Val450 的固定 S2 审计为：`changed=1,535,249`、`corrected=799,584`、
`harmed=552,463`、`neutral=183,202`、`net=247,121`。对应修改中的纠正率为
52.08%，伤害率为 35.99%，决定性修改精度为 59.14%。因此 S2 有真实净收益，
但存在足够多的有害修改，适合保留候选并学习接受/拒绝，而不适合继续蒸馏 S2。

## 实验变量

- U0：冻结 P3。
- U1：固定 LocCC-S2，不训练。
- U2：只接受训练集中全局净收益为正、决定性精度达标且跨事件稳定的修改方向。
- U3：轻量 Utility Gate，只在 P3 与 S2 的 argmax 不一致处选择 S2 或回退 P3。
- O1：读取标签的像素 Oracle，只作上界，绝不部署。

U3 的输入只有 P3/S2 概率、修改方向、置信度、区域纯度、margin、面积和区域类别
分布。第一轮故意不加入 Memory、IM4CD、RGID、图关系或可微超像素，以保证归因。

## 防泄漏流程

```text
Train2918 --事件分层--> gate_train + internal_calibration
       gate_train --> 方向审计/方向规则 + 500步门控训练
internal_calibration --> 六项自动准入检查
       全部通过 --> Val450 一次最终审计
       任一失败 --> 自动停止，不访问 Val450
```

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
bash experiments/spi_s2_utility_gate_v1/run.sh preflight
bash experiments/spi_s2_utility_gate_v1/run.sh smoke
nohup env NUM_WORKERS=4 PILOT_STEPS=500 ACCUMULATION_STEPS=4 \
  bash experiments/spi_s2_utility_gate_v1/run.sh auto \
  > /root/data/BRIGHT-train-logs/SPI_S2_UTILITY_GATE_V1_seed3407.log 2>&1 &
```

先看内部校准 JSON 的 `final.checks`。只有 U3 超过固定 S2、伤害至少下降 20%、
selector AUROC 达到 0.60、前景类别没有明显下降并具有跨事件稳定性时，控制器才运行
Val450。

## 标签图向量化实现

性能版本使用一张 `H×W` LocCC 标签图表示全部区域，并用 `bincount` 一次性计算
区域面积、类别投票、概率均值、纯度和 margin。它替代了“每个区域建立一张完整
`H×W` Mask 并逐区域扫描”的实现。方向变化也只计算一次，再复用到全局和事件统计。
实验定义、S2阈值、四连通规则、最小面积和输出协议均未改变。

同步到服务器后先运行完整预检，再用32张图检查吞吐和分阶段计时：

```bash
bash experiments/spi_s2_utility_gate_v1/run.sh preflight

POLICY_MAX_IMAGES=32 \
NUM_WORKERS=4 \
bash experiments/spi_s2_utility_gate_v1/run.sh fit-policy
```

日志会输出 `p3_and_transfer`、`loccc`、`s2_candidate`、`target` 和 `transition`
的秒/图统计。32张检查通过后，去掉 `POLICY_MAX_IMAGES` 再运行完整 `auto`。

## 后续模块如何接

若 U3 通过，把新证据追加到 `utility_features`，每次只增加一组：先 Memory verifier，
再 IM4CD/图关系可靠性，之后才考虑 selective RGID。可微超像素必须与 LocCC 做等预算
竞争实验，不能与 RGID 同时加入。每个新版本都比较 `新增模块 - U3`，而不是只与 P3 比。
