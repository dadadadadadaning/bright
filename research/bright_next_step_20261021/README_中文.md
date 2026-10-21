# BRIGHT：先理清已有证据，再开展一个新对照

本包提供只读结果清单、原始槽特征余弦先验的模型组件和兼容性检查。
**不包含正式训练/续训启动器，不会把未完成任务改成完成。**

读取修订：兼容审计报告的 `pixels.all_labeled.metrics` 及其逐事件结构。
旧版显示的 `result.json has no supported metrics object` 对这类报告表示汇总器未适配，不能解释成审计失败。
修订版将这些指标标记为 `audit_replay`，不作为额外的训练消融，也不把GT筛选子集混入总体指标。
已经生成的普通训练/验证指标无需因这项修订重跑实验；未知split和未完成状态仍原样保留。

## 1. 当前可以保留的结论

- Extended V2 Test878：Grid 四类 mIoU 68.5259%，Control 1000 步 69.0753%，增加约 0.5494 个百分点。
- 这是完整方案相对冻结基线的改善，不能将全部收益归因于分槽或 Memory。
- Control 1000 与 2000 步只差约 0.0029 个百分点。不要据此确立“1000 步最优”的训练策略。
- 相应 Anchor 结果没有显示额外收益；这不是“所有 KL 都无效”的证明。
- Fresh V3 的训练和 audit_1500 已有完成证据。它与 Extended V2 是不同运行，不能混为同一条轨迹，也不应重复恢复已完成审计。
- 六分支消融的最后一次续训曾成功启动，但当前对话没有给出它的最终完成结果。状态是“待查”，不能写成“未做”或“已经完成”。

## 2. 缺口清单与优先级

| 优先级 | 缺什么 | 怎么补 | 是否需要重新训练 |
|---|---|---|---|
| P0 | 统一的运行、代码、源模型、数据划分、步数、指标台账 | 读取已有 summary/manifest/result，保留原文件路径和哈希 | 否 |
| P0 | E/F/B/D/R/L 最终状态、同一步数和同一 Val 的结果 | 先查已有 resume 输出；只在确实缺失时恢复或补做 | 先不训练 |
| P0 | Damaged 的错误去向及 Precision/Recall | 对同一报告的混淆矩阵计算；逐事件检查 | 通常否 |
| P1 | 原 Control 与新接口的等价性 | beta=0 比较输出、梯度、优化更新；真实数据与保存恢复检查 | 仅短验证 |
| P1 | C0/C1 在统一配方下的 Val 比较 | 同种子、共同前 500 步、相同数据顺序和预算，继续到 2500 | 是 |
| P2 | 相似度对应关系的因果证据 | C1 有收益后做 C2：合法候选内置乱先验数值 | 是 |
| P2 | 稳定性与最终泛化 | 有希望后补至 3 个种子；配置冻结后做 Test；报告逐事件差异 | 部分需要 |

先验的数值统计“变好”不能替代最终指标提高。一个种子不能证明稳定性。

结构主张与必要对照：

| 论文主张 | 对照 | 可解释范围 |
|---|---|---|
| 双槽有益 | B−E、D−F | 在图内/Memory 两种支持下检验软双槽；不要只挑有利环境 |
| 使用图外带标签支持的整体收益 | D−B | 支持来源和标签来源一起变化，不能单独称为“检索算法收益” |
| 相似检索优于随机合法支持 | D−R | 候选数量、排除规则和其余实现需匹配 |
| 正确标签对应关系有用 | D−L | 真实标签优于同分布置乱标签；不是“有标签 vs 完全无标签” |
| 标签信息本身不可替代 | D−N（仅在坚持此主张时补） | N 要真正去掉标签嵌入，不是随意换成某个错误标签 |
| 新读取方式有益 | C1−C0、C1−C2 | 保持候选、标签、槽和原损失不变 |

现有 N 在 supports.py 中已有定义；先查是否有有效结果，不自动启动新训练。
若 E/B 已经获得接近完整 Control 的收益，应收缩 Memory 主张，或再做分类头/区域适配对照。
“只用 pixel+base 的适配头”可以量化适配本身，但其有效参数量与完整头不同，不能伪称严格等容量对照。

## 3. 基线怎么整理

把三个对象分开：

1. **参考结果**：Extended V2 已给出的 Test 数值，标注探索性、seed=3407、具体 checkpoint。
2. **开发基线 C0**：本轮固定的 Control 配方，用于和 C1 同环境、同初始化、同顺序直接比较。
3. **冻结 Grid**：每次同输入前向得到的 grid_probq；用来核对前端和评价链是否一致。

每个正式表格行必须能追溯到：run/result 路径、split 与图像名单哈希、checkpoint 步数与哈希、seed、源模型/Q/bank/value 身份、代码版本、学习率与训练预算、指标定义、完成状态。
外层包和内层依赖的 release 哈希可能不同，不能只因两串哈希不同就判结果无效；比较相同身份字段及其范围。
本工具只保存/汇总这些可读取信息，不会自动认证整个证据链，也不根据 Test 选最佳点。

主指标统一四类 mIoU；同时保留前景三类 mIoU、各类 IoU、Damaged Precision/Recall。
指标从全数据混淆矩阵求和后计算，不能用逐图/逐事件 mIoU 的简单平均冒充全局 mIoU。
这里固定类别平均，缺失类别贡献 0，保持已检查原评价代码的风格；逐事件报告要说明类别缺失。
OA 和明确命名的像素级建筑二分类 F1 可由四类混淆矩阵计算。F_ch/F_cls 需先确定目标论文定义。
单独两份 4×4 混淆矩阵不能恢复“同一像素被救回/误伤”的配对数，后者需要保存的 paired_counts。

## 4. 先执行已有结果汇总

下载单文件 pyz 后，在服务器运行下面一行。无需设置 BRIGHT_RUN，不用 heredoc，不启动后台进程：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_experiment_review_v1.pyz inventory --root /root/shared-nvme/BRIGHT-checkpoints
```

结束打印 `OUTPUT=/tmp/bright_experiment_review_...`，目录包含：

- REPORT.md：运行状态和全集指标；
- metrics.csv：全集和逐事件指标，以及 Damaged→Background/Intact/Destroyed 等方向；
- inventory.json：路径、原结果哈希、配置线索、未知项和读取异常。

默认不写原实验目录、不加载 .pt、不使用 GPU、不扫描 predictions/checkpoints 大文件。
若只希望检查一个任务，可将 --root 换成该完整运行目录；多任务可重复 --root。
输出中的 `complete` 是原报告声明，不是本工具新认证；`unknown` 不等于失败。读取错误必须单独核对。
工具不读取 NLL 逐像素日志、不重新改 NLL 容差、不重写 loss。

## 5. 先验代码实际改动

已检查的原模型 `JointRegionMemoryClassifier._read` 使用可学习多头 Query/Key 点积；
原始 256 维槽特征还经过 T128、role、fraction 编码，因此显式原特征余弦项不是原分数的简单改名。
新模块使用：

```text
prior = cosine(detach(query_slot_f), detach(support_slot_f))
scores = learned_attention_scores + beta * prior
attention = original_masked_softmax(scores)
```

beta=0 直接调用原 forward / _read；没有复制一套新 baseline，也没有修改原 model.py。
beta>0 仅新增打分项；GT 不参与先验。候选 IDs、合法性掩码、标签、路由、损失保持原接口。
C2 仅在每个查询槽的有效候选之间打乱先验值，保持值的集合、数量和掩码；不打乱候选标签，不消耗训练全局 RNG。
置乱映射依赖 seed/image/region/query-role，并与 chunk_size 无关。

核心接入示例（用于正式 runner 的模型构建处，**不是直接运行的训练脚本**）：

```python
import copy
from reader_prior import classifier_with_prior

# OriginalClass 来自已核对的原包；prefix_head/prefix_optimizer 来自同一共同前缀。
PriorClass = classifier_with_prior(OriginalClass)
c0 = copy.deepcopy(prefix_head)
c1 = PriorClass(chunk_size=32, prior_beta=2.0).to(device)
c1.load_state_dict(prefix_head.state_dict(), strict=True)

# 独立优化器，并从同一 500 步 AdamW 状态恢复；不能只复制模型权重。
o0 = torch.optim.AdamW(c0.parameters(), lr=1e-4, weight_decay=.005)
o1 = torch.optim.AdamW(c1.parameters(), lr=1e-4, weight_decay=.005)
o0.load_state_dict(copy.deepcopy(prefix_optimizer.state_dict()))
o1.load_state_dict(copy.deepcopy(prefix_optimizer.state_dict()))

output = c1(**example.common, **example.supports['d'], prior_key=example.image)
loss = runtime.loss(output['probabilities'], example.target)
```

构建临时模型会消耗 RNG：正式 runner 要在共同前缀后捕获状态，建完两个分支和优化器后恢复；
每个分支每步使用配对随机状态，评估后恢复训练随机状态。以上示例没有替代这部分 runner。
原分类头残差最后一层初始化为零，因此只检查“新建模型预测一样”可能掩盖 Reader 错误；本包测试专门激活该层。

注意：state_dict 参数名与原模型一致，beta/mode 是配置而非参数。
**checkpoint 必须另存 `model.prior_config()`，加载时先恢复配置，再严格加载权重；否则可能悄悄加载成 beta=0。**

## 6. 正式实验的固定规则

建议首轮 C0/C1：seed3407，共同前 500 步，同一模型/优化器/RNG 状态分叉，继续至2500。
学习率调度 horizon=5000；1e-4 最大值，500步后乘0.2；AdamW weight_decay=.005，clip=1；原NLL+Lovasz，额外KL=0。
每500步完整Val450。beta=2 只是预先指定的试验值，不是已证明最优参数。
先检查实际 beta×prior 的候选间变化尺度，避免把近常数的偏置误认为有效干预；不同时引入新归一化、损失或采样。
使用共同500步前缀时，C1在训练前就可能改变输出，这是方法的推理效应，应保留500步初始评价。
此实验检验500步后适配阶段的改进；不是新方法从第一步训练的结论。

统一以完整 Val 四类 mIoU 选择检查点，同分选较早者；同时报告固定2500终点，避免只挑各自最有利节点。
2500步只是沿用基线的阶段性预算，不是收敛证明；若每步一张图且无覆盖补充，它也不足以覆盖Train2918全部样本。记录unique_train_queries和实际epoch，不将有限预算结果夸成充分训练结果。
若C0在同协议Val上明显偏离旧结果，先定位源模型、数据、调度和实现差异；不要用本轮C1直接与另一运行的旧Test做唯一比较。
首轮筛查关注四类 mIoU、Damaged IoU 和所有类别代价；例如 +0.1 个百分点可作为实践上的继续研究阈值，但不是统计显著性标准。
若收益极小或不稳定，优先看重复种子/事件差异，不能宣布成功。

C1有希望后，再补同beta的C2和另外两个种子2026、2027。各seed内部共享前缀，seed之间前缀独立。
逐图/逐事件使用配对分析；bootstrap 要重采样图像/事件并重新合并混淆矩阵算指标，不能把相关像素当独立样本。
事件数少时，区间也不能替代独立数据；说明数据的代表性限制。

最终Test不需要所有调参节点都跑：固定Grid、C0、C1及论文关键消融，使用Val决定的检查点。
Test878已被用于探索，后续重复运行不能让它重新变成未触碰测试集。可报告统一协议的比较，并披露这一点；有条件增加新的事件/独立数据。
不为了“重新获得干净Test”随意重划原benchmark划分。

## 7. 怎样避免重复报错

| 旧问题 | 本次策略 |
|---|---|
| SSH重连后变量空，产生 /logs/... | 用户命令给完整路径；runner配置文件记录路径，不依赖会话变量 |
| heredoc被粘成一行 | 下载脚本后用单行命令执行 |
| 下载超时/半文件 | 限定连接/总超时、临时文件下载、SHA256核对后再执行 |
| nohup有PID但程序已退出 | 单独记录 queued/preflight/training/failed/complete；只认可子进程就绪和有效更新证据 |
| tail先于日志创建 | 启动器前台先创建日志；看日志用 tail -F；不得以 tail失败判训练失败 |
| audit目录存在后反复恢复 | 先读 summary；完成任务归档；失败目录保留，新任务独立输出 |
| 六分支部分更新误记完整步 | 全部分支optimizer.step和日志/检查点提交后才记录 last_complete_step |
| 仅加载权重却称精确续训 | 保存/恢复全部Adam、RNG、数据游标、全分支权重、先验配置和文件身份 |
| NLL不同求和顺序被误判 | 复用已验证评价器；比较同一对象/聚合口径，保留误差记录，不为“通过”修改训练损失 |
| CPU测试过但CUDA长程仍失败 | 服务器先真实数据短跑、保存恢复对照、完整Val，再正式跑；单图通过不能证明历史状态故障已修复 |

发生异常后优先保留失败阶段、图像ID、分支、最后完整步和首个traceback。
不得使用 `|| true` 忽略失败，不自动从resume退回fresh，也不把异常后的CUDA进程继续用于训练。

## 8. 检查与交付边界

已检查原 model.py SHA256：`11f033a83dc3844f3ab1f9cf447b243f27278d984a3fbb726a0a7e12f86150f1`。
测试环境为 PyTorch 2.1.2 CPU 和 2.14.1 CPU；两者均通过 synthetic Reader 检查。
结果汇总另有8项标准库测试。没有运行服务器完整数据训练，也没有证明历史CUDA故障消失。

服务器可运行合成兼容性检查（不加载checkpoint，不读取训练/测试图像）：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_experiment_review_v1.pyz check-reader --run /root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_ANCHOR_FRESH_V3/run_20261020_225009_163709 --device cpu
```

确认服务器GPU空闲后，可将device换为cuda:0做同样的合成检查；这仍不是完整数据smoke。
实际模型文件哈希和Reader实现会打印出来。组件只接受已检查的 `_read` 实现；如它发生改变会明确停止，需先核对差异，不能自动绕过。

本包刻意不提供未经真实数据验证的 `nohup ... train` 长命令。下一步首先拿到已有六分支状态和混淆结果，再把此组件接入固定runner并验证真实数据。
文件 `experiment_plan.json` 是拟定协议，不是可直接交给旧训练器的配置，也不能充当训练完成证据。
