# DamageFormer SRAC-style Transformer V1

## 先说明复现边界

这是一个**隔离的、可审计的 SRAC-style Swin 实现**，不是官方 SRAC 代码。

截至 2026-10-08，可核实的公开材料只有 IGARSS 2026 题目、作者、宣传说明和用户提供的流程图；会议页面明确显示 `No resources available`。因此，超像素数量、SSN 距离、Memory 配额、Top-R、元信息编码、训练阶段等无法严格照抄。所有无法核实的量都在命令行和 manifest 中显式记录，不能将本目录称为“官方复现”。

公开依据：

- SRAC 会议页：<https://2026.ieeeigarss.org/view_paper.php?PaperNum=2851&SessionID=1356>
- 国科大介绍：<https://emse.ucas.ac.cn/index.php/zh/xwzxe/535-igarss-2026>
- SSN（ECCV 2018）：<https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html>
- SSN 官方代码：<https://github.com/NVlabs/ssn_superpixels>
- Superpixel Transformers：<https://arxiv.org/abs/2309.16889>
- SPFormer：<https://arxiv.org/abs/2401.02931>
- Retrieval Augmented Classification（CVPR 2022）：<https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html>
- BRIGHT 数据论文：<https://essd.copernicus.org/articles/17/6217/2025/>

这些文献分别支持“可微局部聚类”“像素到区域的局部 Cross-Attention”“Transformer 中的自适应区域表示”和“外部检索分支”。它们不证明这里选择的具体超参数，也不保证在 RGB–SAR BDA 上提升。

## 这版与旧代码的关键区别

```text
原始A1 DamageFormer（严格加载、参数和buffer冻结）
        │
        ├─ A1像素logits ─────────────── 只做基线审计，不进入最终输出
        │
        ├─ 灾前RGB Swin多尺度特征
        │       ↓
        │  轻量嵌套解码器
        │       ↓
        │  Swin网格局部Cross-Attention SSN → Q
        │
        └─ 原A1 fused_roi特征
                + 预测建筑概率B
                + Q软边界
                + Q归属熵
                      ↓
              小幅LayerScale结构适配
                      ↓
       空间注意力Pool + 原特征mean/std旁路
                      ↓
                  区域Token
                      ↓
            ┌─────────┴─────────┐
        learned slots       Train Memory Top-R
            └─────────┬─────────┘
              Pre-LN Cross-Attention
                      ↓
               区域四类logits
                      ↓
               用同一个Q概率回投
                      ↓
                  最终预测
```

特别注意：

1. 不压暗原始 RGB、不画边界后重新编码；此前审计显示该路径有明显损失。
2. 不执行 `A1 logits + Memory correction`。最终类别由当前区域 Token 与 Memory 共同决定。
3. A1 只提供冻结特征、Soft ROI、建筑概率以及外部基线。
4. Train Memory 的标签可作为支持值；Val 查询的 GT 绝不参与 Q、检索或分类。
5. Q 是软超像素归属，不保证连通，也不能称为真实建筑实例。

## Transformer 适配到底是什么

### Stage 1：Swin 网格局部 Cross-Attention

设 A1 第一层融合网格为 `Hf×Wf`。1024 输入通常对应 `256×256`，`stride=8` 时种子为 `32×32=1024`。每个位置只看相邻 `3×3=9` 个种子：

```text
score(i,k) = <Wq LN(x_i), Wk LN(c_k)> / temperature
             - lambda_xy * ||pos_i-pos_k||^2
             + lambda_b * log(building_compatibility)
Q(i,k) = softmax over the 9 valid candidates
```

然后按 Q 更新中心，共三轮。相比旧卷积直接输出9个归属分数，这更符合 Transformer 的 Query–Key 关系；相比硬建筑二分，它只把预测建筑概率作为软先验，避免将定位错误完全固化。

### Stage 2：非破坏式结构注入

输入仍是 A1 的原 `fused_roi`。结构分支只读取：

- 预测建筑概率；
- 相邻位置属于不同种子的软概率边界；
- Q 的归属熵。

它通过初始幅度 `1e-3` 的 LayerScale 进入区域内容特征。该残差发生在**新区域特征内部**，不是对 A1 输出类别做残差修正。

### Stage 3：图示对应的区域表示

主路使用空间注意力加权 Pooling；旁路从未被结构适配覆盖的 A1 `fused_roi` 计算区域 mean/std；再加入预测几何位置编码。默认不增加全局区域自注意力，避免第一轮多一个无法归因的因素；`REGION_CONTEXT=1` 仅作为后续单独消融。

### Stage 4：直接 Memory 分类

no-Memory 与 Memory 使用完全相同的 `JointMemoryReader` 参数结构：

- no-Memory：与Top-R等长的可学习支持槽；
- Memory：逐槽使用真实Train区域，无效位置才退回同一个可学习槽；
- 支持值包含区域 Token、软标签分布、灾害类型、事件代理ID、预测几何；
- Pre-LN Cross-Attention + FFN 后直接输出四类区域 logits；
- 不再把 query 自己伪装成一个支持邻居，也不读取 A1 logits。

## 每个阶段验证什么

| 阶段 | 只回答的问题 |
|---|---|
| `front` | Transformer-aware Q + 区域表示在不破坏A1的条件下，能否接近或超过A1？ |
| `build_memory` | 仅用Train2918能否形成固定、可追溯的历史区域库？ |
| `build_neighbors` | 固定Token空间中实际检索到了谁？不训练模型。 |
| `audit_retrieval` | Damaged/Destroyed是否有同类支持；GT只在邻居ID固定后做审计。 |
| `nomemory` | 同容量Transformer分类器自身能达到多少？ |
| `memory` | 真实历史支持是否比learned slots提供额外信息？ |
| `random/shuffle/off/no_metadata` | 收益来自相似特征、支持标签，还是元信息/额外计算？ |

只有 `memory_normal > nomemory`，并且随机邻居、打乱标签或关闭 Memory 后优势减弱，才支持“Memory 提供了有用历史证据”。

## 检索策略不是一回事

- `global`：纯余弦 Top-K，最接近普通检索，但 Intact 可能挤占稀有类别。
- `per_class`：每个**支持类别**取相同数量；不使用查询 GT，专门检验稀有支持覆盖不足。
- `cross_event`：排除同事件和同图像，证据更保守但覆盖更低。
- `same_image`：只排除同图像，允许同事件，通常更容易但可能利用事件风格。

这些都是需要分别报告的实验因素，不能混称为 SRAC 的官方配置。

## 服务器运行

### 1. 预检

```bash
cd /root/shared-nvme/BRIGHT-master

export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth

bash bda_benchmark/experiments/damageformer_srac_transformer_v1/run.sh preflight
```

### 2. 完整 CUDA/数据冒烟

必须使用全新目录：

```bash
export RUN_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_TRANSFORMER_V1/smoke_$(date +%Y%m%d_%H%M%S)"

bash bda_benchmark/experiments/damageformer_srac_transformer_v1/run.sh smoke_all
```

冒烟只验证接口，不判断性能。

### 3. 正式串行流程

默认是 `global + cross_event`：

```bash
export RUN_TAG="$(date +%Y%m%d_%H%M%S)"
export RUN_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_TRANSFORMER_V1/formal_seed3407_${RUN_TAG}"
export RETRIEVAL_POLICY=global
export EXCLUSION=cross_event
export QUERY_SAMPLING=uniform
export SEED=3407
export BATCH=2
export ACCUM=8
export AMP=bf16

mkdir -p /root/shared-nvme/BRIGHT-train-logs/SRAC_TRANSFORMER_V1
export TRAIN_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_TRANSFORMER_V1/formal_seed3407_${RUN_TAG}.log"

nohup env \
  PYTHON="$PYTHON" A1_CKPT="$A1_CKPT" RUN_ROOT="$RUN_ROOT" \
  RETRIEVAL_POLICY="$RETRIEVAL_POLICY" EXCLUSION="$EXCLUSION" \
  QUERY_SAMPLING="$QUERY_SAMPLING" SEED="$SEED" BATCH="$BATCH" ACCUM="$ACCUM" AMP="$AMP" \
  bash bda_benchmark/experiments/damageformer_srac_transformer_v1/run.sh pipeline \
  > "$TRAIN_LOG" 2>&1 &

echo "PID=$!"
echo "结果=$RUN_ROOT"
echo "日志=$TRAIN_LOG"
tail -F "$TRAIN_LOG"
```

正式预算：

- Stage A：50000 optimizer updates，effective batch 16；
- 两个 Reader：各5000 updates；
- Val450；不访问Test；
- 前端保存 `best.pth` 与 `last.pth`，但只存新模块，不重复保存A1的761MiB权重。

Train全图Token缓存预计约1GB量级，取决于实际有效区域数和序列化开销；运行前仍应检查磁盘。

### 4. 在同一个已完成前端上补稀有类别均衡检索

不重训前端、不重建Memory：

```bash
export RETRIEVAL_POLICY=per_class
export EXCLUSION=cross_event

bash bda_benchmark/experiments/damageformer_srac_transformer_v1/run.sh build_neighbors
bash bda_benchmark/experiments/damageformer_srac_transformer_v1/run.sh audit_retrieval
bash bda_benchmark/experiments/damageformer_srac_transformer_v1/run.sh train_readers
bash bda_benchmark/experiments/damageformer_srac_transformer_v1/run.sh evaluate
```

### 5. 汇总

```bash
$PYTHON -m bda_benchmark.experiments.damageformer_srac_transformer_v1.report \
  --root "$RUN_ROOT" --policy global --exclusion cross_event
```

## 结果判定

第一关：`front_region` 必须先和同一权重下的 `a1_pixel` 比较。若区域路径明显低于A1，不应靠Memory掩盖该缺口。

第二关：`memory_normal > nomemory`。同时检查 Damaged Precision/Recall/F1、Damaged→Intact、Destroyed混淆和每事件指标，不能只看mIoU。

第三关：`shuffle_labels`、`random`、`off` 至少一项应显著破坏 Memory 优势，否则模型可能只是在利用额外参数或元信息。

若三关不成立，应停止声称“Memory有效”。下一步可以单独验证像素细节联合解码，但不能同时解冻A1、更换损失和修改Memory。

