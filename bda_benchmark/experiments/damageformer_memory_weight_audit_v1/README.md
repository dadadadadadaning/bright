# 同一批 Memory 邻居，怎样分配读取权重？

这是独立的、只读的推理验证。它不训练、不重建 Memory、不修改超像素、不修改旧代码或旧结果、不访问 Test。它回答：**已有参考案例相同，改变案例之间的相对权重，是否能减少前景伤害？**

## 一、每项是在验证什么

| 输出 | 用途 |
|---|---|
| `normal` | 已训练的原 Memory 头，仍走原 `need_weights=False` 生产路径。 |
| `identity` | 手工重放原注意力，检查实现没有改变答案。不是新模型。 |
| `uniform` | 同样的邻居，外部邻居之间均匀分配权重。 |
| `cosine` | 同样的邻居，外部邻居之间按余弦相似度分配，温度预先固定为 0.07。 |
| `nomemory` | 同一正式训练中独立训练的无 Memory 对照，不能用推理关闭 Memory 替代。 |
| `source_hard` | 当前冻结的区域分类器，尚未加这些修正头。不是历史原始 A1。 |
| `old_pixel/old_region` | 3000 步前端快照的两个原输出，仅作来源与复现参照。 |

主机制比较是 `cosine-normal`，还必须同时看 `cosine-uniform` 和 `cosine-nomemory`。均匀读取是为了区分“相似度真的有用”和“只要换掉原权重就会有变化”。不能只选最高的一行。

不修改：RGB/SAR 前端、Soft ROI、建筑概率 B、Q、建筑/背景分组、区域 Token、原区域 logits、Memory 内容、同事件/同图排除规则、支持标签、检索 Top-K、检索到的邻居，以及所有已训练参数。每个正质量槽都保留，包含 soft-tail。

## 二、究竟改变哪一个计算

假设一个 attention head 的权重为：8 个邻居权重 + 1 个自身 query 权重。对每个查询、每个 head 分别保留自身权重 `a_self` 以及原外部总质量 `m=sum(a_neighbors)`，只重新分配这份 `m`：

```text
normal:  原来的 a_j
uniform: m / 合法邻居数
cosine:  m * softmax(cosine_j / 0.07)
```

只对合法邻居归一化，无效邻居保持 0。没有合法支持时只用原 query fallback，不产生 NaN。**保留的是每个 head 各自的自身权重，不是平均后的一个数。**

重分配之后仍使用同一个 MHA 的 V 投影、bias、各 head 拼接、输出投影，以及同一个 `query + context -> LayerNorm -> FFN -> 加基础 logits`。不是另写一个邻居标签投票器，也不是去掉残差。

支持标签仍只进入 V，不进入 K。查询 GT 不参与检索、权重或预测，仅在预测完成后用于统计损毁类别和错误位置。这里“区域多数 GT”是诊断标签，不意味着每个超像素就是一栋真实建筑。

## 三、为什么先检查 identity

现有收益量级极小，手工 MHA 重放如果漏掉 bias 或打乱多头通道，也可能产生类似大小的变化。

本包记录 context/logits 数值误差，同时要求：

- 原生产路径与 identity 的区域类别完全相同；
- 按同一 Q 回投后的像素类别完全相同；
- 每个 head 的自身权重、外部总质量保持不变；
- 非法邻居权重为零，Q 与区域 ID 未变。

identity 不通过时保存诊断证据，但标记失败；不把该结果解释成改进，不自动放宽阈值。仍需原前端和正式 Memory/无 Memory 的完整 Val 复现通过。

## 四、怎么运行

上传整个新目录到服务器项目的同名位置：

```text
/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_memory_weight_audit_v1
```

只上传这个新目录；旧实验目录不要覆盖。本包依赖服务器已经跑通的 `damageformer_memory_readout_audit_v1` 及其原工程依赖。

输入直接从你已完成的审计 `manifest.json` 恢复，避免重复填写 FRONT_CKPT、MEMORY_RUN、BANK_ROOT 导致串实验：

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export PARENT_AUDIT=/root/shared-nvme/BRIGHT-checkpoints/MEMORY_READOUT_AUDIT_V1/run_20261008_155643/full

export WEIGHT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/MEMORY_WEIGHT_AUDIT_V1/run_$(date +%Y%m%d_%H%M%S)"

bash bda_benchmark/experiments/damageformer_memory_weight_audit_v1/run.sh preflight
```

预检只运行合成 CPU 测试；不能证明真实 CUDA/权重已通过。

### 先跑两张图的 smoke

```bash
export OUTPUT_ROOT="$WEIGHT_ROOT/smoke"
bash bda_benchmark/experiments/damageformer_memory_weight_audit_v1/run.sh smoke
```

smoke 仍使用正式权重和正式 Train 库，只检查前两张 Val 的流程。看 `MEMORY WEIGHT AUDIT COMPLETE`，不能用 smoke 分数选择方案。失败后不要跳过检查直接跑 full。

### smoke 成功后，完整 Val450

```bash
export OUTPUT_ROOT="$WEIGHT_ROOT/full"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/MEMORY_WEIGHT_AUDIT_V1
export AUDIT_LOG="/root/shared-nvme/BRIGHT-train-logs/MEMORY_WEIGHT_AUDIT_V1/$(basename "$WEIGHT_ROOT").log"

nohup bash bda_benchmark/experiments/damageformer_memory_weight_audit_v1/run.sh audit > "$AUDIT_LOG" 2>&1 &
echo "审计PID=$!"
echo "结果目录=$OUTPUT_ROOT"
echo "日志=$AUDIT_LOG"
tail -n 80 -F "$AUDIT_LOG"
```

不要预先创建 `OUTPUT_ROOT`。已有输出或 `.lock` 会拒绝覆盖；换新的 WEIGHT_ROOT 即可，不删除旧结果。训练仍占 GPU 时不要盲目并行，确认显存/资源足够。

### 完成后只读报告

```bash
export REPORT_ROOT="$WEIGHT_ROOT/full"
export BOOTSTRAP=2000
bash bda_benchmark/experiments/damageformer_memory_weight_audit_v1/run.sh report
```

重新登录后，请把 REPORT_ROOT 明确设成启动时打印的实际完整目录，不要重新生成时间戳。报告不使用 GPU、不推理、不训练，也不依赖 OUTPUT_ROOT 或 PARENT_AUDIT。

旧 shell 的 FRONT_CKPT、SOURCE_RUN、MEMORY_RUN、BANK_ROOT、SEED、TOPK、QUERY_CHUNK 不参与本轮输入选择。可设置 DEVICE/WORKERS 仅改变运行设备与加载进程数。温度与检索条件没有调参入口，不在 Val 上扫描挑最高。

## 五、先看哪些结果

1. `identity_passed=True`，来源核验、正式复现通过。
2. `cosine-normal`：是否比原注意力更好？
3. `cosine-uniform`：是否确实需要相似度，而不只是换一种权重？
4. `cosine-nomemory`：是否最终超过无历史案例的匹配对照？
5. 全体与 GT 前景分别救回多少、改错多少；Damaged P/R/F1、其他类别 FP/FN、Damaged->Intact 转移。
6. B 漏检/误检位置、建筑侧边界、对象尺寸代理和每事件表现是否恶化。

读出证据含实际 attention 加权标签质量与余弦标签质量，按 all/hard/tail 和类别分开。它们不是分类准确率，也不是注意力的因果贡献。槽位数、质量加权区域变化和像素变化不能互相替代。

置信区间按已观察事件内的配对图像重采样，不把像素当独立样本；不覆盖训练种子或未见事件，邻近切片的空间相关性仍是限制。smoke/失败/缺失数据不产生有效性结论。

如果 cosine 同时超过原注意力、均匀读取和无 Memory，并减少前景伤害，才值得另开训练对照。若推理替换失败，也不能证明重新训练不可能有效，因为下游已经适配原注意力；工程上可以保留无 Memory，暂停这版读取扩展。不要据此改变 Q/B 或删区域。

## 六、输出与磁盘

只新建 `manifest.json`、`baseline_policy.json`、`initial.json`、`reproduction.json`、`reader_evidence.json`、`audit_final.json`、`summary.json` 等汇总记录。没有 `.pth`，没有复制 761MiB 前端、Memory 库、稠密预测或注意力图。正式审计仍需读取原始权重和库；运行时间、峰值显存记录在 summary 中。

本地 CPU 合成测试与真实服务器 CUDA 验证是两回事。正式效果必须以新审计结果为准，本方案不承诺涨分。

本地验证记录：新包 48 项测试通过；原 `damageformer_memory_readout_audit_v1` 的 52 项测试回归通过；两个 Python 命令行入口通过帮助检查。环境为 PyTorch 2.12.1+cpu。未运行服务器 PyTorch 2.2.1/CUDA 的真实数据 smoke 或 Val450，也未在本地执行 Bash 启动器。服务器仍须依次运行 preflight、smoke、audit。
