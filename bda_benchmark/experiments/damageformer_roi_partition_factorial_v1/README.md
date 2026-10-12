# ROI分区二因素实验：先判断区域形状是否值得保留

本目录是独立新增实验。不修改A1、V2前端、旧ROI实验或正在运行的任务。
输入必须是已有 `SRAC_FOURSTAGE_FRONT_V2` 固定快照，而不是A1权重或区域读出权重。
当前前端是A1初始化后联合训练得到的模型；`old_pixel`不是历史原始A1。

## 1. 本轮仅验证两个因素

| 组 | 分区 | 预测建筑/背景分组 | 本组用途 |
|---|---|---|---|
| g0 | 特征图上的固定方格，硬归属 | 无 | 简单区域基线，不是无区域像素基线 |
| g1 | 相同固定方格 | 有 | 简单方格是否也能受益于建筑隔离 |
| s0 | 快照原始学习型软Q | 无 | 与旧S0完全相同的路由和读出 |
| s2 | 快照原始学习型软Q | 有 | 与旧S2完全相同；隔离背景后是否还需要学习分区 |
| h0，可选 | 对同一个Q取argmax | 无 | 分离软归属与硬区域布局的效应 |
| h1，可选 | 对同一个Q取argmax | 有 | 与s2比较软归属；与g1比较硬区域布局 |

默认仅训练g0/g1/s0/s2，h0/h1不默认启动。没有实现soft18候选、Memory、结构重建损失、SLIC或新主干。

```text
同一张灾前RGB + 灾后SAR
         ↓
冻结的V2前端（eval，参数/BN统计均不更新）
  ├─ 原RGB结构特征 → 原始Q（冻结）
  ├─ 预测建筑概率B（冻结）
  └─ 原图融合特征 × (0.1 + 0.9B) → 同一内容F
         ↓
 g0 / g1 / s0 / s2 四种路由
         ↓
相同初始权重、独立训练的完整区域读出
注意力Pooling + mean/std旁路 + 四类分类
         ↓
使用同一条路由回投 → 原图概率 → NLL + Lovasz
```

代码仍调用旧前端完整forward，以核验old_pixel/old_region与旧头复现；其中旧图像调制分支只用于对照输出。新四组内容全部来自原像素头输入的Soft ROI特征，不使用调制内容。这是为了保证一致性，尚未做绕过旧分支的速度优化。

## 2. 哪些会训练，哪些不会

- 每组只训练区域特征投影、位置注意力、统计旁路、归一化和四类分类器；dim128时每组117382参数。
- Swin、跨模态融合、定位头、Q生成器、建筑概率都冻结；本轮不声称优化了超像素生成器。
- 四组每个微批共享同一次前端输出、同一数据增强、相同读出初值；每组有独立optimizer。
- 分组作用于Pooling、mean/std、回投三个环节。背景组也可以输出四类，不强行输出Background。
- GT仅进入损失与离线诊断。不使用GT建筑轮廓、类别或尺寸生成分区，不删除小建筑，不裁掉SAR输入。
- 分组隔离在原生特征网格上成立；最后双线性上采样仍可能在邻接位置混合概率，不能宣称全分辨率绝对隔离。

## 3. 保持原来的训练协议

默认每组5000次optimizer更新，batch2 × accum8 = 有效batch16，每组80000训练裁剪。
四组同一步一起更新，不是四组分摊5000步。每个微批按组依次反向，不长期保留四张读出计算图。
AdamW：lr1e-4、weight_decay0.005；warmup最多100步，然后cosine；梯度裁剪1。
每500步验证完整Val450。初次先复现固定快照的old_region记录，否则停止，不绕过源代码/权重哈希检查。
Train2918/Val450名单严格验证。旧loader内部 `type='test'` 仅表示无训练增强，提供的是Val名单，不读取Test划分。

主结果是预先规定的第5000步配对差值，best作为次要结果。20步smoke仅检查流程，不用于性能判断。
不自动比较旧独立任务的best，不自动继续训练旧头或旧optimizer。

## 4. 为什么默认四组一起重跑

旧S0/S2可作为历史参考，但不能仅凭模块相同就拼接成严格配对实验。
默认重跑四组，保证checkpoint、batch、数据顺序、裁剪增强、初始化、optimizer和验证点相同。
本地合成测试会验证s0/s2与旧实现的前向和梯度一致；这不是服务器全数据数值复现的替代。

当前框架控制相同seed stride和读出参数，**不保证实际有效区域数量相同**。
S2−G1包含软硬性、形状和区域占用差异，不直接叫作“纯形状收益”。有信号再补h0/h1。

## 5. 上传与启动

上传本目录全部 `.py`、`run.sh` 和README到服务器同名experiments目录。它复用项目现有V2和诊断模块，不能只上传run.sh，也不需要新CUDA扩展。

下面命令使用明确的环境Python，避开conda activate未初始化或系统Python缺包的问题。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth

test -x "$PYTHON" && test -s "$FRONT_CKPT"
"$PYTHON" -c 'import sys, torch, scipy, imageio; print(sys.executable); print(torch.__version__)'

bash bda_benchmark/experiments/damageformer_roi_partition_factorial_v1/run.sh preflight

export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/ROI_PARTITION_FACTORIAL_V1/smoke_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_roi_partition_factorial_v1/run.sh smoke
```

若快照不存在，先定位之前同一快照，不要用A1权重代替，也不要复制训练进程正在覆写的best作为稳定快照。
输出目录必须不存在。保留失败目录，重新运行换新目录，不关闭防覆盖保护。

smoke完整成功后再启动正式训练：

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/ROI_PARTITION_FACTORIAL_V1/formal5000_seed3407_$(date +%Y%m%d_%H%M%S)"
export BATCH=2 ACCUM=8 STEPS=5000 EVAL_EVERY=500 SEED=3407
export VARIANTS="g0 g1 s0 s2"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/ROI_PARTITION_FACTORIAL_V1
export TRAIN_LOG="/root/shared-nvme/BRIGHT-train-logs/ROI_PARTITION_FACTORIAL_V1/$(basename "$OUTPUT_ROOT").log"

nohup bash bda_benchmark/experiments/damageformer_roi_partition_factorial_v1/run.sh train > "$TRAIN_LOG" 2>&1 &
echo "训练PID=$!"
echo "结果目录=$OUTPUT_ROOT"
echo "日志=$TRAIN_LOG"
```

GPU不足时等旧实验完成或使用其他GPU；本脚本不会自动停止任何旧任务。
重新登录需重新设置PYTHON/FRONT_CKPT/OUTPUT_ROOT，不依赖上一终端环境。

查看进度与报告：

```bash
tail -f "$TRAIN_LOG"
"$PYTHON" -m bda_benchmark.experiments.damageformer_roi_partition_factorial_v1.report --root "$OUTPUT_ROOT"
```

后续确需软硬控制时，用新输出目录并设置 `VARIANTS="g0 g1 s0 s2 h0 h1"`，每组协议保持一致。不要只对一组延长训练，再与其他组5000步结果比。

## 6. 结果怎么看

| 差值 | 要回答的问题 |
|---|---|
| g1−g0 | 普通网格是否因建筑隔离受益 |
| s2−s0 | 原学习Q是否因建筑隔离受益 |
| s2−g1 | 建筑隔离后，自适应软区域方案是否优于硬网格方案 |
| (s2−s0)−(g1−g0) | 两种分区下建筑隔离收益的差异，不是像素mIoU可加分解 |
| s2−h1，可选 | 相同Q来源下软回投是否有价值 |
| h1−g1，可选 | 硬归属条件下自适应布局是否有价值，仍须考虑实际区域数 |

若g1和s2接近，可优先简单网格；但“没有显著差异”不是等效证明。
若s2对Damaged、定位漏检区域明显有害，不只看平均mIoU。
有明确正向信号再重复至少三个配对种子，不能把同一次训练的多个验证点当独立实验。

## 7. 最后一步的诊断

末次验证额外统计：

- 四类混淆矩阵、P/R/F1、背景误报；
- g0→g1、s0→s2、g1→s2，以及各新组相对同快照old_pixel的rescue/harm；
- Damaged→Intact伤害、Intact→Damaged救回；
- GT类别×边界/内部、预测ROI漏检/误检、B位于[.4,.6]的位置；
- GT前景8连通域尺寸代理（<256、256..1023、>=1024原图像素），不是真实建筑实例；
- 每图、每事件混淆矩阵，便于后续图像/事件级成对重采样。

定位错误分组把原生特征网格B用nearest扩展到输出网格，以对应真实路由决定；不是另用bilinear制造新的阈值分组。
GT仅用于最终统计，条件mIoU不是全图mIoU。缺失类比例为null；固定三前景类mIoU缺失类计0，同时提供present类口径。
当前保存原始统计，不自动计算bootstrap置信区间或多种子显著性；也未新增GT实例约束或连通性损失。

## 8. 保存与磁盘

只保存每组best和last读出，不保存前端副本、特征缓存、optimizer或每个迭代的checkpoint。
默认四组共8个小权重文件；dim128原始FP32参数约3.6MiB合计，另有序列化元数据、训练日志、末次审计JSON。
这不是整个目录大小承诺；诊断JSON随图像数量增长。原有约761MiB前端快照仍必须保留。
没有恢复optimizer的resume功能，last供评估而非精确断点续训；不要误认为存有训练状态。

## 9. 实现文件

- core.py：仅新增六种路由，复用经过对照的原完整读出。
- train.py：同批配对训练、冻结检查、原头复现、末次诊断。
- diagnostics.py：只读GT的离线分组与转移统计。
- report.py：主末步/次最佳分开汇报，同步差值，不冒充历史A1公平比较。
- test_core.py / test_diagnostics.py / test_report.py：旧逻辑一致性、分组隔离、梯度、统计与报告检查。
- test_training.py：临时目录内20步CPU合成数据训练入口集成测试，验证四组独立更新与保存；不是真实数据CUDA冒烟。

本实验是机制验证，不保证涨分，不是严格复现SRAC，也不以历史75.65%作为通过门槛。

## 10. 本地验证记录（2026-10-05）

Windows / Python bright环境 / PyTorch 2.12.1+cpu：新目录18项测试、旧ROI实验8项回归测试合计26项通过。
包含真实模型64×64随机输入的前向、读出反向、冻结buffer检查，以及20步合成数据的完整trainer入口集成检查。
train/report入口帮助检查通过；run.sh为LF换行。
未运行服务器真实Train/Val，也未验证服务器PyTorch 2.2.1 CUDA算子。当地没有bash，因此不将本地Python测试当作Linux启动脚本运行验证。
服务器仍需先preflight，再20步真实数据smoke。该smoke只评价2张Val图像，不代表完整Val性能。

### CUDA统计初始化修复（2026-10-05）

服务器PyTorch 2.2.1在模型加载前执行reset_peak_memory_stats曾报
`Invalid device argument 0: did you call init?`。现已在train.py中通过prepare_device显式检查CUDA可用、初始化CUDA、选择设备，再重置峰值统计。
这只修复启动与资源记录顺序，不改变模型、Q、Soft ROI、损失或优化器。已补4项mock调用顺序回归测试；CPU测试不能替代真实GPU验证。
上传更新的train.py和test_training.py，然后重跑preflight与20步真实smoke；不需更新V2前端代码或改动输入checkpoint。
