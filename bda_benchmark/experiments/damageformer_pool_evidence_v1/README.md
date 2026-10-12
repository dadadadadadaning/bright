# 固定 hard 路径：实际汇总证据检查

这不是新训练实验，不调整温度、空间权重、迭代数、stride，不加入 Memory。
新代码全部隔离在本目录；旧模型、损失、数据加载、checkpoint 和源实验均不改动。
GPU/真实数据尚需在服务器 smoke 与完整 Val 检查，本地 CPU 合同测试不能代替它们。

## 每一步回答什么

1. `report` 读取已有 `audit_final.json`：建筑漏检/误检、B概率分箱、边界、对象代理、损毁类别转移已经发生在哪里。只读 JSON，不导入 PyTorch，不使用 GPU，也不重新推理。
2. `smoke` 只读推理前两张 Val：检查源文件、冻结状态、原读出重放和统计代码。两张图的分数没有全 Val 含义。
3. `audit` 完整 Val450：增加实际区域汇总中的 GT 混合、支持量、有效支持数与错误关联。需要运行模型，默认用 GPU，但无优化器和反传。
4. `report` 读新审计结果：据证据决定是否值得做尺度对照、路由修复或现有 Memory/无Memory对照。不自动选参数，不自动启动下一训练。

## 输入与输出不变

- 固定源是区域联合训练 3000 步的 V2 front_snapshot，不是历史 A1。
- 读出是完成的 ROI_EXCHANGE_V1 的 hard_last，使用同一步最终权重，不混用 best。
- hard 是 B>=源阈值 的建筑/背景分组，组内 Q 仍是软归属；不是 Q argmax。
- 使用原图 RGB–SAR 融合 + 已有 Soft ROI，主预测直接调用原 head.forward。
- 另行拆解 Embedding、classifier 和原 unpool，只用于验证重放一致，不替代正式输出。
- GT 仅在预测完成后参与离线统计，绝不输入 Q、B、路由或分类器。
- 原始文件哈希、参数及 buffers 前后检查；完整 Val 基线使用既有历史有界复现规则，规则在本次推理前固定，不根据当前分数放宽。
- 不访问 Test。源码哈希不匹配时停止，不能绕过检查；应同步和源权重对应的旧依赖版本。

## 统计口径

### 两条汇总路径分别检查

- `stats`：权重 Q，对应 mean/std 统计旁路。
- `attention`：权重 Q * sigmoid(score(feature))，对应主路径。
- mass：这些权重之和，单位是原生特征位置；不是原图建筑面积。
- ESS：(sum w)^2 / sum(w^2)，同一位置重复候选ID先合并，估计有效支持位置数。注意力质量小不等于位置少。
- GT 映射：原图每类 one-hot 和有效掩码按整数倍率平均到特征格，保留小建筑占比。不使用最近邻删掉小对象；非整数倍率明确拒绝，不猜测映射。
- 这种 GT 面积映射仍只是特征格标签代理，不能描述编码器真实感受野。
- GT 纯度用有效 GT 质量归一化；不使用模型的 1e-6 分母 guard 来计算纯度。
- 记录严格空槽、正质量分母guard槽、没有有效GT的槽，防止将空槽记成高纯度。
- 四类纯度、建筑/背景二类纯度、仅前景三种等级纯度分别报告。纯背景不算高前景等级纯度。
- `clean_supported` 是离线诊断组：四类纯度>=.95、mass>=1、ESS>=2，不是推理门控或最优阈值。
- `clean_supported_foreground` 另外要求GT前景占比>=.95。判断“纯建筑区域是否仍分类差”时优先看该组的 `roi_true_positive`，不能拿包含大量纯背景的 clean_supported/all 下结论。

### 与错误如何关联

先按区域的上述指标分箱，将分箱指示量按原回投 Q 与双线性插值映射回原图，得到各像素受到该类区域支持的权重。
以该权重统计 old_pixel正确→source_hard错误（harm），以及反向 rescue。
这里得到的是 **fractional/加权像素数**，不是独立对象数，也不是因果贡献。
主路径权重用于定义接收内容；映射错误位置时仍用实际 read Q，不会错误地把 attention 当回投权重。

在相同指标的完整互斥分箱内，加权统计可相加；不同指标、foreground 与 ROI 子组重叠，不能跨指标相加。
纯度分箱会排除没有有效 GT 的槽；前景纯度分箱排除没有前景 GT 的槽，曝光权重因此可能小于1，绝不重新归一化掩盖它。
具体分箱边界在 core.py 常量和 slot_masks 中固定。

`audit_final.json` 同时保留原有精确像素诊断（ROI/B分箱/类别转移/边界/尺寸/事件/每图混淆矩阵）和新增加权关联。
不将高全图纯度当作前景性能，不将像素当独立样本计算显著性。
主路径与旁路的混合率不是它们对最终预测的贡献比例，非线性分类器与可学习旁路系数也会影响结果。

## 上传与准备

将整个 damageformer_pool_evidence_v1 文件夹上传到服务器同名 experiments 目录。
依赖服务器现有 roi_followup_v1、roi_exchange_v1、partition_learning_v1、region_memory_v2 等原工程目录；不要只上传 run.sh。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth
export SOURCE_RUN=/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/formal5000_seed3407_20261005_133348

"$PYTHON" -c "import sys, torch; print(sys.executable); print(torch.__version__)"
ls -lh "$FRONT_CKPT" "$SOURCE_RUN/hard_last.pth" "$SOURCE_RUN/summary.json"
bash bda_benchmark/experiments/damageformer_pool_evidence_v1/run.sh preflight
```

## 先读已有统计，不用GPU

```bash
"$PYTHON" -B -m bda_benchmark.experiments.damageformer_pool_evidence_v1.report --root "$SOURCE_RUN"
```

读取不到 audit_final.json 就明确报错，不会替你重新推理。

## 两张Val冒烟检查

```bash
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/POOL_EVIDENCE_V1/smoke_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_pool_evidence_v1/run.sh smoke
```

不要提前 mkdir OUTPUT_ROOT；同路径重复执行会被保护机制拒绝，换新路径即可，不删除旧结果。

## 完整Val审计（冒烟成功、GPU资源可用后）

```bash
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/POOL_EVIDENCE_V1/full_$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/POOL_EVIDENCE_V1
export AUDIT_LOG="/root/shared-nvme/BRIGHT-train-logs/POOL_EVIDENCE_V1/$(basename "$OUTPUT_ROOT").log"
nohup bash bda_benchmark/experiments/damageformer_pool_evidence_v1/run.sh audit > "$AUDIT_LOG" 2>&1 &
echo "审计PID=$!"
echo "结果目录=$OUTPUT_ROOT"
tail -n 80 -F "$AUDIT_LOG"
```

tail 的 Ctrl+C 只停止查看。正式审计结束应看到 COMPLETE，随后：

```bash
bash bda_benchmark/experiments/damageformer_pool_evidence_v1/run.sh report
```

新终端须重新设置 OUTPUT_ROOT 为上述实际路径。

## 结果怎么决定下一步

| 证据 | 可以考虑的下一项实验 | 本程序是否执行 |
|---|---|---|
| B正确的前景位置，高混合区域伤害集中 | stride8/4整体尺度对照；其他参数、读出预算固定 | 否 |
| B漏检/误检位置伤害集中 | 检查硬分组切断的信息；如继续交换需关注接收端实际预算 | 否 |
| B正确且纯度/支持充分，仍分类较差 | 固定表示的Memory/无Memory增量对照 | 否 |
| 关联不清晰或样本很少 | 保留现有hard，不作因果结论或追加模块 | 否 |

上述是竞争解释，不会自动宣布根因。温度、空间权重、迭代数、区域数保持原值。

## 现有Memory实验如何衔接

如果已有 Memory/无Memory 训练正在跑，不重启、不改Q，等待原任务结束即可。
若还未启动，上面显式 export 的 FRONT_CKPT 与 SOURCE_RUN 同样供现有 Memory V2 使用。
先检查其既有 preflight/smoke；新代码不修改 Memory 算法，也不自动训练。

```bash
# 仅确认未有同一任务运行、资源足够后，手动执行。与审计无先后算法依赖。
export BATCH=2 ACCUM=8 SEED=3407 STEPS=5000 EVAL_EVERY=500
export MEMORY_ROOT="/root/shared-nvme/BRIGHT-checkpoints/REGION_MEMORY_V2/formal5000_seed3407_$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/REGION_MEMORY_V2
export TRAIN_LOG="/root/shared-nvme/BRIGHT-train-logs/REGION_MEMORY_V2/$(basename "$MEMORY_ROOT").log"
nohup bash bda_benchmark/experiments/damageformer_region_memory_v2/run.sh all > "$TRAIN_LOG" 2>&1 &
echo "Memory训练PID=$!"
tail -n 80 -F "$TRAIN_LOG"
```

`all` 是既有的 Train 建库 + 匹配 Memory/无Memory 新分类器训练，不是只训练Memory一组。
冻结源Embedding后建库；不要把改了Q、尺度、前端的权重与旧库混用。

## 输出与资源

只输出 manifest/summary/initial/reproduction JSON 和审计统计，不保存新的front、预测图或特征缓存。
GPU峰值统计覆盖本次审计（包括诊断），不能当正式训练显存；计时单列原前端+hard推理，另报总审计耗时。
失败也尽可能保留 audit_candidate.json 与错误原因，但 complete=false 不得当作成功审计。
不删除、不覆盖旧结果。尚未执行真实服务器GPU/Val检查。
