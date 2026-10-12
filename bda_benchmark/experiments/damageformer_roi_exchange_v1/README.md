# ROI有限跨组交换验证：先检查代价，再训练最小对照

本目录与旧实验隔离。不要替换旧 factorial/control 的文件，不改变旧权重、数据和结果。
本轮不是重新训练 DamageFormer，也不是完整 SRAC 复现；只研究预测建筑分组是否过于严格。

## 1. 三项任务分别在验证什么

| 入口 | 是否训练 | 目的 |
|---|---|---|
| `inspect` | 否；仅读旧 JSON | Intact、Destroyed 召回损失流向哪里？是否真的被改成 Damaged？ |
| `smoke_audit` / `audit` | 否；冻结同一步旧 S0/S2 | 按 B 十分箱 × 定位正确/漏检/误检，定位伤害；包含高置信度错误，不预设只在 B≈0.5 |
| `smoke` / `train` | 只训练两套独立读出 | 同一二值建筑判断下，少量跨组权重能否减轻错误分组的代价？ |

`smoke` 使用真实 Train 裁剪、20次更新和Val前2图，只检查流程；不能用20步/2图的mIoU判断性能。
正式训练为每组5000次更新、每次有效batch16，共80000个裁剪/组；两组看到相同裁剪。
同一个进程对同一批前端特征分别更新两套读出，不是把5000步分给两组。
所有评估仅 Train/Val，拒绝Test列表。GT只进入训练损失与评估，不进入分区。

## 2. 模型中保留什么、只改什么

保留当前V2第3000步不可变快照、原图RGB–SAR融合内容、原Soft ROI、学习得到的Q、建筑概率B。
Swin、融合层、Q生成器、B以及全部前端运行统计冻结。最终输出仍来自区域分类与回投，
不是像素残差；没有Memory，没有新损失，也不压暗原图或绘制边界作为新输入。
捕获器为了与旧实现一致仍执行既有前端路径；本轮新读出只读取原像素头输入的Soft ROI特征。

| 名称 | ε | 相对旧S2的变化 |
|---|---:|---|
| `hard` | 0 | 数学上等价旧S2，扩展接口中另一组权重为0 |
| `exchange` | 0.05 | 每个位置95%权重到原组，5%到另一组 |

这不是连续概率分组 `q×B`：仍使用完全相同的 `g=(B>=0.5)`，只放松隔离强度。
默认先固定ε=0.05，不建议立即扫多个ε用Val挑最好结果。
两组完整读出结构相同，dim128时各117382参数，初始化完全相同但参数互不共享。
训练损失保持区域NLL+Lovasz，AdamW lr1e-4、wd0.005、100步warmup、余弦衰减、梯度裁剪1。

### 核心张量和归属

```python
# Q: [batch, 9, h*w], B: [batch, 1, h, w]
# feature: [batch, 256, h, w]; ids: [batch, 9, h*w]
g = (B.flatten(2) >= 0.5).float()
u = epsilon + (1 - 2 * epsilon) * g
weights = torch.cat([Q * (1-u), Q * u], dim=1)  # [batch,18,h*w]
support_ids = torch.cat([2*ids, 2*ids+1], dim=1)
# 偶数槽背景，奇数槽建筑；总槽位2K。两部分相加仍为Q，总权重1。
```

Pooling的均值为 `sum_i r_ik*f_i / max(sum_i r_ik,1e-6)`。
注意力主路再乘位置注意力；mean/std统计旁路仍使用r；回投使用同一份r，不在回投端改成硬分组。
新 pool/unpool 遍历实际候选数18，不能沿用固定 `range(9)`。
空槽的均值为0；它对任意位置的权重也为0，因此即使分类器有bias也不能影响回投。
保持旧实现的FP32累积和1e-6分母保护，不额外剪掉小质量组。

当特征图256×256、stride8时，K=32×32=1024，分组后2048槽；
这只是典型Val尺寸，代码按实际h/w计算，并验证种子网格与快照一致。
stride8是特征网格间距，不是原图8像素。Q、B、内容特征必须同分辨率，不允许静默缩放。
类别概率回投到特征图后使用既有双线性插值、align_corners=False回到原图。
诊断用最近邻将native B对应到原图，保留模型实际分组判断；不是用平滑后的B重新分组。
最后的双线性插值可能融合邻近位置，所以native硬分组不等于输出边界绝对隔离。

两组梯度仅流向注意力、特征投影、mean/std旁路和区域分类器。
GT不作为Q、B、分组的输入，也没有GT建筑实例裁切。
建筑/背景子组仍可能包含多栋建筑，不保证实例一致或连通性。

### 特别注意

5%是单个位置跨组的归属权重，不是“区域内最多5%的背景”。
10个建筑位置、90个背景位置时，建筑组可能收到9.5份建筑和4.5份背景，后者约占32%。
因此输出还记录子组异组内容占比；这是风险诊断，不是GT纯度。
ε>0通常激活更多软槽，不能声称两组有效区域数相同，也不能声称18候选意味着整网耗时必然翻倍。
资源日志是同一进程前端加两头的总耗时/峰值显存，不能拿它当单方法速度对照。

ε=0.5只用于单元测试：两子组相同，在正常质量范围内应退化为未分组S0。
极小质量受原有1e-6分母保护影响，不承诺任意极端输入的代数精确相等。

## 3. 先上传并检查

将整个 `damageformer_roi_exchange_v1` 文件夹上传到服务器：

```text
/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_roi_exchange_v1
```

不要只传run.sh；它还依赖服务器已有的fourstage_v2、region_evidence/content、modulation_controls、
roi_partition_control和roi_partition_factorial目录。不要为解决哈希不匹配而覆盖这些旧目录。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth
export RUN_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_PARTITION_FACTORIAL_V1/formal5000_seed3407_20261005_024506

test -s "$FRONT_CKPT"
test -s "$RUN_ROOT/s0_last.pth"
test -s "$RUN_ROOT/s2_last.pth"
"$PYTHON" -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
bash bda_benchmark/experiments/damageformer_roi_exchange_v1/run.sh preflight
```

显式指定解释器，不依赖当前shell是否成功conda activate。
前端应与旧factorial manifest一致；已知SHA256是
`0503203589f8282173c49b8e38aaf6992d5bf3f913d1b0448306b3c17362b394`。
本代码读取并校验哈希，不新复制761MiB前端，也不把原始A1任务权重当此快照使用。

## 4. 不训练的两项检查

### 4.1 读取现成转移矩阵：不用GPU

```bash
bash bda_benchmark/experiments/damageformer_roi_exchange_v1/run.sh inspect
export OUTPUT_ROOT="$RUN_ROOT"
bash bda_benchmark/experiments/damageformer_roi_exchange_v1/run.sh report
```

`inspect`给出完整4×4 harm/rescue矩阵；`report`给出易读摘要、
每类TP/FP/FN差值、Damaged误报来自哪些GT类，以及Intact/Destroyed被改成什么。
旧JSON没有B十分箱时明确提示“没有”，不从总体混淆矩阵伪造分箱。

### 4.2 冻结旧S0/S2补B置信度诊断

```bash
export NAMES="s0 s2"
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/audit_smoke_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_roi_exchange_v1/run.sh smoke_audit
```

通过后才运行完整Val450：

```bash
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/audit_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/ROI_EXCHANGE_V1
export AUDIT_LOG="/root/shared-nvme/BRIGHT-train-logs/ROI_EXCHANGE_V1/$(basename "$OUTPUT_ROOT").log"
nohup bash bda_benchmark/experiments/damageformer_roi_exchange_v1/run.sh audit > "$AUDIT_LOG" 2>&1 &
echo "审计PID=$!"
echo "结果=$OUTPUT_ROOT"
echo "日志=$AUDIT_LOG"
tail -f "$AUDIT_LOG"
```

这里只读取同一步 `_last.pth`，不混用各组不同步的best；要求来源manifest、源码、权重哈希一致，
完整遍历450图并复现旧最终指标。不满足就报错，不关闭检查继续跑。
`Ctrl+C`退出tail只停止看日志，不自动终止后台任务；不要在GPU任务仍运行时无意启动下一项。

输出B区间[0,.1)、…、[.9,1]分别与ROI TP/FN/FP/TN相交的混淆、伤害、救回及分母。
既看绝对数量也看伤害率，不用少量高置信错误的比率替代总影响。

## 5. 训练两组对照

前一项完整审计结束、GPU资源足够后运行。此步骤不会接着更新旧S2；两组新读出从同一随机初始化独立训练。

```bash
export VARIANTS="hard exchange"
export EPSILON=0.05
export ROI_THRESHOLD=0.5
export BATCH=2
export ACCUM=8
export SEED=3407
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/smoke_seed3407_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_roi_exchange_v1/run.sh smoke
```

smoke通过，再启动新目录正式训练：

```bash
export STEPS=5000
export EVAL_EVERY=500
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/formal5000_seed3407_$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/ROI_EXCHANGE_V1
export TRAIN_LOG="/root/shared-nvme/BRIGHT-train-logs/ROI_EXCHANGE_V1/$(basename "$OUTPUT_ROOT").log"
nohup bash bda_benchmark/experiments/damageformer_roi_exchange_v1/run.sh train > "$TRAIN_LOG" 2>&1 &
echo "训练PID=$!"
echo "结果=$OUTPUT_ROOT"
echo "日志=$TRAIN_LOG"
tail -f "$TRAIN_LOG"
```

训练前先完整Val复现前端并检查新ε=0对旧S2的输出/全部读出梯度一致性。
前端仅冻结评估，梯度核对在临时读出副本上执行，不影响训练初始化或随机数状态。
正式训练结束自动生成完整诊断、打印报告，不必额外再跑450图。
查看进行中状态或最后结果：

```bash
"$PYTHON" -m bda_benchmark.experiments.damageformer_roi_exchange_v1.report --root "$OUTPUT_ROOT"
```

如果只想重新审计已完成的新实验，把RUN_ROOT设为新训练目录，NAMES设为`hard exchange`，
OUTPUT_ROOT换全新审计目录。不要让输出目录等于已有训练目录。

## 6. 输出、磁盘与验证边界

训练输出：`manifest.json`、`initial.json`、`training.jsonl`、`validation.jsonl`、`summary.json`、
`audit_final.json`，以及 `hard_best.pth`、`hard_last.pth`、`exchange_best.pth`、`exchange_last.pth`。
只保存读出，不保存优化器、不复制前端、不保存特征缓存或逐像素预测。
dim128时四个头的FP32参数载荷约1.8MiB，实际还含序列化/manifest和JSON诊断开销；远小于复制四套前端。
审计只保存JSON，不保存新权重。输出目录存在时拒绝启动，不覆盖或删除结果。
last仅在训练结束保存；不提供中断后的精确优化器恢复，也不把best冒充可恢复检查点。

所有训练与审计都校验冻结前端参数和buffers前后不变，初始head独立且相同。
每图/事件的混淆矩阵保留在audit_final.json，可以做后续配对重采样；本版未自动生成置信区间。
若工程源码与历史manifest不一致，应先核实旧版本来源，不能直接改JSON哈希绕过检查。

## 7. 如何判定下一步

主要比较同一5000步 `exchange - hard`，不同步best仅作补充。
需要同时看：全图前景mIoU、各类P/R/F1，Damaged→Intact伤害、Intact/Destroyed流向，
ROI漏检/误检的伤害、前景侧边界、小对象代理、事件差异、背景误报反弹以及实际区域占用。

- 若定位错误组改善且总体与类别指标不明显恶化：存在继续验证的信号，再做配对种子重复。
- 若漏检改善但背景污染明显反弹：证实交换存在代价，不把Damaged单项上涨称为全面提升。
- 若对定位错误组无帮助或整体更差：停止本版固定交换假设，不自动扫更多ε、不继续加Memory。
- 若hard无法复现旧S2逻辑：先排接口或数据问题，不解释exchange性能。

单次随机种子的多个验证时间点不是独立重复；多种子波动和Val图像/事件统计不确定性要分别报告。
条件子集指标不是全图mIoU，GT连通域只是对象代理，不代表真实建筑实例；建筑等级GT也不代表楼内破损位置。
本轮实验可以检验“完全隔离是否有代价”，不能单凭ε改动就宣称方法原创或保证超过A1。
