# Q 明确化 / 跨事件检索距离：两个独立、可否定的实验

## 先明确边界

本目录是新增实验，不改 `damageformer_srac_transformer_v1`、`damageformer_srac_joint_context_v1` 或历史结果。
当前在跑的父流程继续作为对照。本目录不杀进程、不删除结果、不访问 Test，也不自动并行抢占同一张 GPU。

现有证据只说明 Q 偏软是风险，不证明是主要掉分原因；4 张图 smoke 中的稀有类近邻结果也不是正式检索结论。
这里不换 Swin，不恢复图像压暗/画边界，不添加建筑实例限制，不加新损毁类别权重，不同时训练 Memory 和 Q。

这不是 SRAC 官方复现。新代码是对本项目父实现的单因素实验，论文支持的是相关机制，而不是本实验会涨分的保证。

## 实验 A：只改变 Q 的明确程度

| 组 | 初始化及训练 | 唯一区别 | 回答的问题 |
| --- | --- | --- | --- |
| 当前父流程 / `control` | 同一原始 A1、相同新模块初始化和 50,000 步协议 | beta 恒为 1 | 当前区域路径能力 |
| `q_anneal` | 从同一 A1 重新开始；不是接着父流程 best 训练 | beta 从 1 缓慢增到 2 | 更明确的归属是否减少不必要混合 |

原分区的每次局部迭代先得到 9 个候选的完整分数 s，包括特征相似度、空间距离、预测建筑相容性。
新公式仅为：

```text
Q(i,k) = exp(beta * s(i,k)) / sum_j exp(beta * s(i,j))
无效候选保持 -inf；空中心不参与；每位置候选权重总和为 1。

step <= 1000: beta = 1
1000 < step < 10000: beta = 1 + (step - 1000) / 9000
step >= 10000: beta = 2
```

乘的是**完整候选分数**，不是只把特征相似度权重调大。每轮都计算归属并更新中心，共三轮。
最终同一份 Q 同时用于边界提示、注意力 Pooling、mean/std、几何统计以及概率回投，不是只在输出端硬化。
没有 argmax 代替训练、没有熵损失，也没有保证一个区域就是一栋建筑。

Q、区域 Embedding 和区域分类头仍然可训练；原 A1 全部参数和缓冲区冻结。GT 只进入损毁损失，不进入推理分区。
新代码 beta=1 的输出、候选 ID、梯度和初始化随机数消耗与父分区做精确契约测试。
beta 是持久化 buffer：加载某个 best 时恢复该时刻 beta，不能误用最终 beta=2。

正式训练仍是 batch=2、accum=8，有效 batch=16，50,000 次优化器更新；AdamW 1e-4 / 5e-3，warmup1000 + cosine，原 NLL+Lovasz。
正式 Val450 每1000步，smoke 只看两张图，不能用于判断效果。

观察：同一步数前景 mIoU、各类 IoU/P/R/F1、实际 Q 的 top1/熵/有效区域数、adapter 修正量相对内容的范数。
温度日程本身不保证训练后 Q 更硬，因为学习参数可能补偿它。

**停止条件：**如果 Q 更明确却持续损伤 Damaged 或整体指标，拒绝该版本；不能以“超像素更漂亮”接受它。
如果实际 Q 没变化，只能说本日程没有产生足够有效干预。单次最佳 checkpoint 或单种子的小优势不算可靠胜出。
若 best 恰好选在 beta=1 的早期阶段，也不能把其成绩解释为 Q 明确化的增益；优先看同一步数与最终结果。

## 实验 B：Q 不动，只改变 Memory 的找法，并保留 A1 像素证据

这一实验用**已完成的当前父流程**的 `front/best.pth` 和它自己生成的 Train Memory。
不能用实验 A 的新 Embedding 查询旧库，也不能用未完成父流程的中间文件建正式结论。

先只训练一个 128×128、无 bias、单位矩阵初始化的检索投影 W：

```text
旧检索：相似度 = cosine(query token, support token)
新检索：相似度 = cosine(W * query token, W * support token)

查询区域 -> 得到 Top-8 行号
          -> 读取原始 support token / 原始软标签 / 原始元信息
          -> 原来的 Reader
```

新投影只改变邻居排序，不改变库成员数量、不改邻居输入特征、不换 Reader、不按查询 GT 指定类别。
训练投影时只用 Train 库里纯度 >=0.8 的区域多数类别，四类均衡采样 anchor；同类、不同事件和不同图像作正对，异类且不同事件/图像作负对。
这个纯度门槛**只用于拟合检索距离**；实际检索仍用完整原库，Val 查询也不按 GT 纯度过滤。
缺少跨事件正对会被报告；正式训练要求四类都至少有两个事件支持，不能偷偷放宽为同事件配对。
只训练 W 共2000步，固定最后一份，不用 Val 选择 W。记录额外预算，不能宣传为零成本改进。

随后冻结 W，用同一前端、同一 Q、同一库、同一批训练图像训练四组独立联合分类器：

| 组 | 输入 | 比较目的 |
| --- | --- | --- |
| J0 | A1 像素特征，无区域上下文 | 仅训练像素读出有什么变化 |
| J1 | A1 像素特征 + 当前区域（可学习槽位，无历史） | 区域是否有增量 |
| J2 | A1 像素特征 + 区域 + 原余弦 Memory | 真实历史是否胜过同结构无历史 |
| J3 | A1 像素特征 + 区域 + 新距离 Memory | 只改邻居排序是否有价值 |

J1/J2/J3 的 Reader 与联合分类器同结构、相同初值、各自独立优化；J0 绕过 Reader，不能声称有效容量相同。
每个头5000次更新，完整 Train 图像 batch=1、不增广，沿用已有 joint_context 对照协议。
这个预算**不是**前端的 50,000步/有效batch16协议；四头结果只能与同轮匹配对照解释，不能与旧区域训练横向冒充公平对照。
本版训练时在线检索冻结的重算 Token，旧 joint_context 使用缓存 Token 预计算邻居；半精度缓存可能使极近邻排序略变。
因此请用本轮 J2 作 J3 的直接检索对照，不把旧联合实验的历史 J2 当作逐位相同的复现。

联合分类器沿用现有 `JointContextDecoder`，没有悄悄更换最终判别逻辑：

```text
A1 fused_roi: [1,256,h,w] ─────────────┐
区域 Token: [1,K,128] -> Reader -> 按Q回投 [1,128,h,w]
                                     ↓
             同一个线性分类器 -> 四类 logits -> 原图预测
```

分类器像素权重复制 A1，区域权重从零开始；第0步必须复现 A1 概率/混淆矩阵。
数学上分块线性分类仍可写成 `W_pixel F + W_context C`；这不是新的“非残差融合”。
训练后复制出来的像素头也会更新，因此保留像素证据不等于保证永远不低于 A1。
W 的学习不会反向更新 A1、Q 或 Token；Reader 训练也不会反向更新 W 或 Memory 库。

观察重点：

- J3−J2：检索排序改善是否转化为最终分割改善？
- J2−J1：原检索的历史信息是否有用？
- J1−J0：区域上下文是否有增量？
- 每类查询的同类 Recall@8、同类邻居比例、邻居类别混淆、支持覆盖；这些不是全图 mIoU。
- 全图前景 mIoU、四类 P/R/F1/IoU、各事件结果、Damaged→Intact 伤害/救回。

**停止条件：**检索稀有类没有改善，不扩大 Reader；检索改善但 J3 不胜 J2，下一次再单独检查 Reader，不把 W 的训练损失下降当成成功。
即使 J3 高于 J2，若仍低于 J1，也不能宣布 Memory 已有用。

## 服务器运行

上传整个新目录到服务器同名 `bda_benchmark/experiments/` 下。不要覆盖父实验目录。
先确认当前 GPU 任务；以下命令不自动停止或排队其他任务。

### 1. 契约检查 + Q 冒烟

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export QMETRIC_EXP=bda_benchmark/experiments/damageformer_srac_qmetric_v1
bash "$QMETRIC_EXP/run.sh" preflight

export EXPERIMENT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_QMETRIC_V1/smoke_q_$(date +%Y%m%d_%H%M%S)"
bash "$QMETRIC_EXP/run.sh" smoke_q
```

20步 smoke 使用缩短的 beta 日程，以真正检查 beta变化和保存加载。它不是正式性能实验。

### 2. 正式 Q 实验（GPU 空闲后运行）

```bash
export CONTROL_FRONT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_TRANSFORMER_V1/formal50000_seed3407_20261008_173616/front
export EXPERIMENT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_QMETRIC_V1/q50000_seed3407_$(date +%Y%m%d_%H%M%S)"
export BATCH=2 ACCUM=8 SEED=3407 AMP=bf16
mkdir -p /root/shared-nvme/BRIGHT-train-logs/SRAC_QMETRIC_V1
export QMETRIC_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_QMETRIC_V1/$(basename "$EXPERIMENT_ROOT").log"
nohup bash "$QMETRIC_EXP/run.sh" q > "$QMETRIC_LOG" 2>&1 &
echo "新实验PID=$! 日志=$QMETRIC_LOG"
```

CONTROL_FRONT 必须是你当前实际运行的父流程；若路径不同请改成实际路径。代码核对初始化/数据/训练协议，遇到不匹配停止，不跳过。
若无法提供父流程记录，可用新输出目录运行 `control`，代价是额外一轮完整训练。

```bash
"$PYTHON" -m bda_benchmark.experiments.damageformer_srac_qmetric_v1.report \
  --root "$EXPERIMENT_ROOT/front" --reference "$CONTROL_FRONT"
```

### 3. Memory/联合实验：父流程完成后

先用**已经通过的父流程 smoke 目录**运行 `smoke_metric_joint`，不能拿正式库冒充 smoke；程序会检查来源。

```bash
# 设置 PARENT_RUN 为你实际已完成的父流程 smoke 根目录，不是 front 子目录。
# export PARENT_RUN=你的真实smoke目录
export EXPERIMENT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_QMETRIC_V1/smoke_joint_$(date +%Y%m%d_%H%M%S)"
bash "$QMETRIC_EXP/run.sh" smoke_metric_joint
```

通过后使用正式库：

```bash
export PARENT_RUN=/root/shared-nvme/BRIGHT-checkpoints/SRAC_TRANSFORMER_V1/formal50000_seed3407_20261008_173616
export EXPERIMENT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_QMETRIC_V1/metric_joint_seed3407_$(date +%Y%m%d_%H%M%S)"
export QMETRIC_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_QMETRIC_V1/$(basename "$EXPERIMENT_ROOT").log"
nohup bash "$QMETRIC_EXP/run.sh" metric_joint > "$QMETRIC_LOG" 2>&1 &
echo "检索及联合实验PID=$! 日志=$QMETRIC_LOG"
```

顺序：投影2000步 → 四组联合头各5000步；不启动新的前端训练。不是把两种改进同时放入一个模型。
Q 与检索分别得到可信正结果之后，才值得做结合实验；届时必须用新 Q 前端重新建库，本版没有自动合并此阶段。

## 存储及验证边界

- A 阶段只存小型 adapter best/last（含优化器），引用 A1 文件及 SHA256，不重复761MiB A1。
- B 阶段复用只读原 Memory 缓存，不复制缓存；增加一个约128×128的投影和四组小型联合头 best/last。
- 新结果目录已存在会停止；不使用覆盖、删除、放宽复现阈值来“修复”错误。
- 本地测试只证明模块契约、梯度和接口；不能替代服务器 CUDA / 真实数据 smoke。
- 本次本地 Python/PyTorch CPU 下26项测试、入口检查和启动脚本语法检查通过；尚未在服务器运行本版真实数据训练。
- 首轮用于筛选；正式方法结论需重复种子。配对事件/图像 bootstrap 与种子波动分别报告，不能把像素当独立样本。
- 边界/小对象/定位错误分箱、显存耗时和 Memory 随机/打乱标签干预仍应使用专项审计；本目录未冒称已完成所有诊断。
- GT 为建筑损毁等级；GT 连通域不是建筑实例；前RGB/后SAR特征差不能称为真实楼内破损变化。

文献依据与迁移边界见 `notes/insights/20261008-qmetric.md`。
