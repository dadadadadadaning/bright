# DamageFormer ROI Selective Memory V1

这是一个与旧实验完全隔离的两阶段验证，不修改前端、超像素 `Q`、建筑分组、区域
Embedding、原区域分类器或已有 Memory Bank。它使用已经完成的
`ROI_EXCHANGE_V1/hard` 路径和 `ROI_MEMORY_QUALITY_BANK_V2` Train-only hard bank。

它不是官方 SRAC 复现，也不预设 Memory 一定有效。两个阶段分别回答两个问题：

1. **方案一：检索对象和名额是否设计错了？**
2. **方案二：即使邻居有用，原来的 Cross-Attention 是否没有真正利用支持标签？**

Test 不被读取；所有选择仅使用 Train/Val。

### 基线复现闸门

实际训练起点`source_hard`和`old_pixel`继续使用原严格复现策略。旧前端中的
`old_region`仅作为诊断，不参与本实验的Pooling、检索、分类或损失；其CUDA软区域
散点归约可能让少量临界像素跨进程翻转。因此，只有当该项同时满足以下条件时才记为
“完整保留但非阻断”：失败原因仅为绝对差异、没有历史异常、且差异不超过复现模块
早已预设的`0.005 pp`硬上限。超过上限、格式异常、来源哈希异常，或
`old_pixel/source_hard`失败仍会在第0步阻断训练。阈值不根据本轮结果拟合。

## 1. 固定的数据流

```text
冻结 DamageFormer 前端与预测建筑概率
            ↓
冻结 hard 建筑/背景分组 Q
            ↓
冻结区域 Embedding 与 source_hard logits
            ↓
     本实验的新读出头（唯一可训练部分）
            ↓
使用原 hard Q 回投到像素
            ↓
     原像素 NLL + Lovasz 监督
```

Memory 中只保存 Train 区域 Token、Train 区域软标签和质量元数据。查询图像的 GT
不会进入检索、门控或分类器。Val GT 只在邻居完全确定后用于评价。

## 2. 阶段一：质量自适应检索

四个独立读出头具有相同参数量、相同初始状态和相同训练批次：

| 名称 | 检索方式 | 验证目的 |
|---|---|---|
| `nomemory` | 不读取 Bank，使用同容量 learned slots | 排除新增分类容量本身 |
| `global_hard` | hard bank 全局余弦 Top-K | 原不平衡检索基准 |
| `fixed_balanced` | 每类固定 `K/4` 个支持 | 检查强制四类配额是否有害 |
| `adaptive_cap` | 质量过滤后全局排序，每类只有上限 | 方案一 |

### 2.1 质量过滤

阈值只由 Train hard bank、按其存储的主类别分别计算。默认
`QUALITY_QUANTILE=0.25`：

```text
purity     >= 本类第25百分位
entropy    <= 本类第75百分位
hard_count >= 本类第25百分位
```

比较使用闭区间。`labelled_mass`仍被记录，但本版不把它作为第四个隐藏过滤条件。

### 2.2 自适应上限不是类别配额

默认先取质量合格候选中的全局余弦 Top-32，再按原顺序扫描：

```text
最多保留8个邻居
每个存储类别最多4个
某类可以是0个
不为缺失类别强行补足
不从Top-32之外回填
不使用低质量支持回填
```

因此它保留的是“类别上限”，不是之前的“每类必须2个”。同图像始终排除；默认还
排除同事件，`ALLOW_SAME_EVENT=1`只能作为明确消融。

## 3. 阶段二：显式标签后验与可靠性门控

四个头同样具有完全相同的可训练张量和初始状态：

| 名称 | 使用方式 | 验证目的 |
|---|---|---|
| `query_only` | 只使用查询 Token 的零初始化残差 | 同容量、无检索对照 |
| `posterior_nogate` | 显式后验始终参与（有支持时 gate=1） | 支持标签本身是否有用 |
| `posterior_query_gate` | 门控只看查询特征与原预测不确定性 | 普通查询门控是否足够 |
| `posterior_quality_gate` | 再加入相似度、后验和支持质量 | 方案二 |

支持标签后验为：

```text
w_i = softmax(cos(query, support_i) / temperature)
p_mem = sum_i w_i * support_soft_label_i
```

最终在 logit 空间做几何后验修正：

```text
final = base + query_residual
        + strength * gate * center(clamp(log(p_mem) - log(p_base), -5, 5))
```

截断用于避免单个近似one-hot支持把缺失类别推到极端负logit；类维中心化只移除对
softmax无影响的共同偏移。

`query_residual`最后一层和`strength`均从0开始，所以 **step 0逐元素等于
source_hard**。每次优化后将`strength`投影到 `[0,1]`。

质量门控额外使用：Top-1相似度、Top-1/2差值、Memory后验置信度、后验熵、有效
邻居比例、平均支持纯度和平均支持熵可靠度。没有有效邻居时，Memory项严格为0，
模型回退到同一个`query_only`路径；训练后不要求回退到裸`base`，但step 0必须是
裸`base`。

## 4. 自动契约测试

测试覆盖：

- 自适应类别上限、无最低配额、Top-L外不回填；
- 保持全局相似度顺序；
- 同图像/同事件排除；
- 查询GT不在检索和门控接口；
- 所有头参数量和初始状态一致、存储独立；
- step 0严格恒等；
- 无支持回退`query_only`；
- 支持标签改变会改变显式后验；
- 无效padding含NaN及全无邻居时仍为有限值；
- 标签打乱不改变邻居ID/相似度，并且不依赖query chunk大小；
- 两个阶段各20步的合成CPU端到端流程。

服务器先运行：

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python

bash bda_benchmark/experiments/damageformer_roi_memory_selective_v1/run.sh preflight
```

只有全部测试显示 `OK` 后才进行真实数据 smoke。

## 5. 准备三个只读输入

```bash
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth

# 已完成、包含 hard_last.pth/manifest.json/summary.json 的 ROI_EXCHANGE_V1 目录
export SOURCE_RUN=/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/<completed_run>

# 两个已完成的 balanced-v2 Train bank，均须包含
# bank_all.pt、bank_hard.pt、manifest.json。
# smoke Bank只能用于smoke；formal Bank只能用于正式审计/训练。
export SMOKE_BANK_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_BALANCED_V2/<completed_smoke_bank>
export FORMAL_BANK_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_BALANCED_V2/<completed_formal_bank>
```

不能用A1任务权重代替`FRONT_CKPT`，不能用旧Memory head代替Bank，也不能使用Val/Test
建立Bank。程序会校验来源、哈希、split和smoke/formal标记。不要为了复用同一Bank而
放松这项校验；若尚无smoke Bank，先用`damageformer_roi_memory_balanced_v2/run.sh
smoke_bank`建立一个，正式Bank则使用其`bank`命令建立。

默认关键参数：

```bash
export TOPK=8
export PRESELECT=32
export PER_CLASS_CAP=4
export QUALITY_QUANTILE=0.25
export MIN_COSINE=-1.0
export TEMPERATURE=0.07
export BATCH=2
export ACCUM=8
export SEED=3407
```

本版为了保证`random`只改变排序而不被语义相似度门槛筛选，正式主实验固定
`MIN_COSINE=-1.0`。非默认阈值应另建实验，不要混入本轮。

## 6. 先做三个真实数据 smoke

每次必须使用不存在的新目录：

```bash
export BANK_ROOT="$SMOKE_BANK_ROOT"

export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_SELECTIVE_V1/smoke_audit_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_roi_memory_selective_v1/run.sh smoke_audit

export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_SELECTIVE_V1/smoke_retrieval_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_roi_memory_selective_v1/run.sh smoke_retrieval

export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_SELECTIVE_V1/smoke_posterior_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_roi_memory_selective_v1/run.sh smoke_posterior
```

Smoke仅验证真实CUDA/数据/权重流程，不用于比较精度。

## 7. 正式执行顺序

先建立日志目录：

```bash
export BANK_ROOT="$FORMAL_BANK_ROOT"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/ROI_MEMORY_SELECTIVE_V1
```

### 7.1 先做只读检索审计

```bash
export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_SELECTIVE_V1/audit_seed3407_$(date +%Y%m%d_%H%M%S)

nohup bash bda_benchmark/experiments/damageformer_roi_memory_selective_v1/run.sh audit \
  > /root/shared-nvme/BRIGHT-train-logs/ROI_MEMORY_SELECTIVE_V1/audit_seed3407.log 2>&1 &
echo "审计PID=$!"
```

审计固定邻居后才读取Val GT，报告正常、随机、标签打乱和关闭支持时的覆盖率、类别
构成、相似度、纯度、后验置信度、kNN准确率与软交叉熵。它不是最终像素mIoU。

### 7.2 方案一训练

```bash
export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_SELECTIVE_V1/retrieval5000_seed3407_$(date +%Y%m%d_%H%M%S)
export STEPS=5000
export EVAL_EVERY=500

nohup bash bda_benchmark/experiments/damageformer_roi_memory_selective_v1/run.sh retrieval \
  > /root/shared-nvme/BRIGHT-train-logs/ROI_MEMORY_SELECTIVE_V1/retrieval5000_seed3407.log 2>&1 &
echo "方案一PID=$!"
```

主要比较同一步：

```text
adaptive_cap - nomemory
adaptive_cap - global_hard
adaptive_cap - fixed_balanced
```

### 7.3 方案二训练

方案二是独立实验，不加载方案一训练出的head。默认只复用方案一的
`adaptive_cap`检索规则：

```bash
export POSTERIOR_RETRIEVAL=adaptive_cap
export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_SELECTIVE_V1/posterior5000_seed3407_$(date +%Y%m%d_%H%M%S)

nohup bash bda_benchmark/experiments/damageformer_roi_memory_selective_v1/run.sh posterior \
  > /root/shared-nvme/BRIGHT-train-logs/ROI_MEMORY_SELECTIVE_V1/posterior5000_seed3407.log 2>&1 &
echo "方案二PID=$!"
```

主要比较：

```text
posterior_nogate       - query_only
posterior_query_gate   - query_only
posterior_quality_gate - query_only
posterior_quality_gate - posterior_query_gate
```

## 8. 查看过程与结果

```bash
tail -F /root/shared-nvme/BRIGHT-train-logs/ROI_MEMORY_SELECTIVE_V1/<log_name>.log

export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_SELECTIVE_V1/<completed_run>
bash bda_benchmark/experiments/damageformer_roi_memory_selective_v1/run.sh report
```

每个训练目录只保存每个新head的`best.pth`与`last.pth`，不复制761 MiB前端、不保存
优化器、不复制Bank。输出目录不能复用；已有结果不会被覆盖。

## 9. 判断和停止条件

### 在训练方案一之前停止

满足任一项时，不推进自适应Memory训练：

- 语义检索不优于随机检索；
- 打乱支持标签基本不影响kNN证据；
- Damaged/Destroyed覆盖仍接近0；
- 质量过滤使大量查询无可用邻居。

这表示当前Embedding或支持标签不具备可靠检索价值，继续调门控没有依据。

### 方案一停止

以统一5000步最终结果为主。若`adaptive_cap`未同时优于`nomemory`和至少一个原检索
策略，或提升只来自背景误报下降却持续伤害Damaged召回/Destroyed，则停止继续调
配额。若简单`global_hard`同样好，保留简单方案。

### 方案二停止

若`posterior_quality_gate`不优于`query_only`，或随机/标签打乱/关闭Memory后结果几乎
不变，则不能声称Memory提供了信息。若`posterior_nogate`更好，说明复杂门控没有必要。
若`posterior_query_gate`与质量门控相同，不能把收益归因于检索质量特征。

只有单种子出现清晰正向信号后，才用相同种子集合重复。报告种子均值/标准差，同时
按图像或事件做bootstrap区间；不能把像素当独立样本。

## 10. 必须同时查看的指标

- 全图前景mIoU；
- Damaged Precision / Recall / F1；
- Damaged→Intact伤害及反向救回；
- Destroyed的代价；
- ROI定位正确/错误位置；
- 前景侧和背景侧边界、前景内部、小对象代理；
- 每事件变化；
- 每类邻居覆盖、平均有效邻居数、相似度、纯度与熵；
- posterior置信度/熵、gate均值、strength、实际改变区域比例；
- 显存、耗时。

`random`、`shuffle_labels`和`off`是固定已训练头上的推理干预，不是重新训练的公平
基线。最终结论必须优先使用相同训练步的结果，而不是从不同方法各自挑最好的检查点。

