# 固定 E0 的 Q × Memory 分层诊断

本目录只新增独立审计代码；不改变原 SRAC、E0/E1、Memory、权重或旧失败报告。它回答：当前 Q 覆盖混合标签的区域，是否更容易出现 J1→J2/J3 的 Damaged 救回或 Intact 误改？这是关联诊断，不将相关性写成 Q 的因果作用，也不提供建筑实例计数。

## 输入和严格校验

入口只接受已通过的完整 Val450 `legacy/native/all` E0。使用 E0 保存的参数、同一数据根、同一设备、同一 Python/数值配置和同一 `seed+2` 加载顺序；调用现有 `load_verified_bundle` 检查权重、旧源码哈希、分割列表、运行环境和 E0 的 `1e-5` 复现门槛。若 E0 之后同步的 `mmap_fix` 等改动使源码哈希不符，**立即失败，不绕过**。可选 E1 也需通过完整证据、复现和六项不变性检查。

E0 `predictions/*.npz` 每图保存 GT、A1、direct_region、J1/J2/J3 预测及 J2/J3 的 Top-8 行号。此审计逐图验证 SHA256、GT、原图顺序、各路径混淆矩阵；冻结前端重新计算 Q、A1 与 direct_region，并严格比较 Top-8 行号。J1/J2/J3 的分类器**不重跑**：使用 E0 原逐像素预测。因此没有训练，也没有 Val GT 进入查询、选邻居或分类器。

重放顺序与 E0 一致：前端计算后立即选 J2/J3 邻居，再读取保存的预测文件。若首张图的 Top-8 仍不同，输出目录中会保存 `mismatch.json`，包含至多五个不一致查询的 E0/当前行号、在**当前查询**下的相似度、Top-8/Top-9 分数间隔，以及同一进程连续两次前端计算的 Q/token 最大差值。E0 没有保存原相似度，故这些数值不能当作 E0 相似度。出现不一致仍立即失败，不放宽门槛；查看失败证据后请用**全新** `Q_DIAG_OUTPUT` 重跑，不要覆盖旧目录。

## 统计定义

对每个 Q 区域，按现有 `region_targets` 计算 Train/Val 同定义的四类像素组成 `y`，但 Val 的 `y` **只用于预测完成后的统计**。前景占比为 `f=y_Intact+y_Damaged+y_Destroyed`；在 `f>0` 时，前景类别纯度为 `max(y_Intact,y_Damaged,y_Destroyed)/f`。这样 `[0.8 BG, 0.2 Damaged]` 与 `[0 BG, 0.5 Intact, 0.5 Damaged]` 不会被混为一种“不纯”。

硬分层将每个特征格的最大 Q 路由作为其区域，再以最近邻映射到 GT 像素网格：

- 前景占比 `<0.2`、`[0.2,0.8)`、`>=0.8`；无有效标签单列。
- 前景类别不存在、纯度 `<0.8`、`[0.8,0.95)`、`>=0.95`；无有效标签单列。
- 同时给出两轴交叉表，以及逐事件表。

由于实际模型用**所有软 Q 路由**，上述硬分层只是清晰可复核的像素归组，不等于真实回投范围。另报告 Q 对两种混合区域的软权重暴露度：建筑/背景混合 (`0.2<=f<0.8`) 和高前景占比但前景三类混合 (`f>=0.8` 且条件纯度 `<0.8`)。软权重先在特征网格求和，再双线性映射到像素网格；报告各固定错误集合的平均暴露度。

每个分层都有 `GT × J1 × J2/J3` 整数转移矩阵。核心率的分母固定为 `GT=Damaged 且 J1!=Damaged` 的救回机会、`GT=Damaged 且 J1=Damaged` 的原本正确者，以及 `GT=Intact 且 J1=Intact` 的原本正确者；分别报告救回、伤害与 `Intact→Damaged`。还有 Damaged 假阳性净变化、E0 四类整体指标和事件统计。单位是像素，**不是建筑**。

## 服务器执行

先同步整个新目录，保持原旧实验包与其源哈希不变：

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export EXP=bda_benchmark/experiments/damageformer_srac_q_memory_diagnostic_v1
bash "$EXP/run.sh" preflight

export RUN_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130
export E0_PROTOCOL="$RUN_ROOT/protocol"
export E1_EVIDENCE="$RUN_ROOT/evidence"
export Q_DIAG_OUTPUT="$RUN_ROOT/q_memory_diag_$(date +%Y%m%d_%H%M%S)"
unset Q_DIAG_LIMIT
bash "$EXP/run.sh" diagnose
```

上述 `RUN_ROOT` 是用户此前给出的已通过运行；运行前确认其真实存在，并确认服务器硬查询包仍与该 E0 源哈希一致。输出 `diagnostic.json`、`summary.json`、`images.jsonl`。若只做前两图接口检查，请在**另一全新输出目录**设置 `Q_DIAG_LIMIT=2`；该结果会标记 `full_validation_evidence=false`，不能用于正式结论。程序拒绝覆盖已有目录，也不会访问 Test。

若哈希校验因 E0 后代码变更失败，需在保持原文件不改的情况下使用 E0 创建时的独立代码副本重跑诊断，或新代码下重新建立通过门槛的 E0；不能改写 E0 JSON、忽略校验或放宽 `1e-5`。
