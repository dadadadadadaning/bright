# Idea Evaluator：P3光学锚定的层级结构检索分类

## 1. First impression

- Paper type: Novel Method（跨领域Technique迁移）
- One-sentence story: 保留P3可靠的光学建筑定位，将最终损毁决策由像素头改为层级结构区域分类，并用严格训练域、类别均衡的区域记忆补充稀有损毁证据。

## 2. Fatal-flaws audit (early gate)

| # | Flaw | Severity | Defense |
|---|---|---|---|
| 1 | 与SRAC高度接近；在SRAC全文未公开时无法证明“超像素+检索”的方法新颖性 | MAJOR | 不把SR1/SR2直接作为最终创新；在全文公开后逐项对照。最终差异必须落在层级Occurrence/Severity、事件留出memory或可拒绝检索至少一项，并做对应消融。 |
| 2 | 当前最强直接工作声称75.65 mIoU，但指标口径、基线和消融不可核验，存在基线不真实风险 | MAJOR | 把SRAC列为不可复现的reported result；实际实验必须比较P3、官方BRIGHT基线、DFC公开模型，并统一clean协议。 |

## 3. Lifecycle and capability match

| Aspect | User's input | Assessment |
|---|---|---|
| Idea category | Innovative technique / cross-domain transfer | 需要先做3个月受控验证，再决定是否扩展 |
| Lifecycle | 预计6–9个月；SR1/SR2约4–6周 | 与遥感方法论文节奏基本匹配 |
| Weekly effective hours | 未提供 | 无法完整判断，需用户自查 |
| Compute | 已有RTX 5090服务器训练记录 | SR1/SR2可行；完整VMambaSSN多种子风险较高 |
| Fit | 当前工程能力已能维护多条严格实验链 | Yellow：技术可执行，但新颖性取决于SRAC全文对照 |

## 4. Five-dimension radar

| Dimension | Score 1-10 | Evidence | Lift suggestion |
|---|---:|---|---|
| Higher | 7 | 机制型、尚无数据：区域主分类直接针对P3碎片化和稀有类别；但H13/Q2残差已失败，不能预先归因提升 | SR1先隔离决策单位，SR2再隔离检索 |
| Faster | 4 | 超像素减少分类token，但P3双编码与Top-k检索增加开销，尚无效率目标 | 预计算训练memory并报告检索延迟 |
| Stronger | 8 | 机制型、尚无数据：训练域只读memory、类别均衡和事件留出可直接针对跨事件偏移 | leave-one-event-out为决定性实验 |
| Cheaper | 5 | 不增加新标注，但需要保存训练区域特征；没有明确成本下降 | memory压缩/每类容量消融 |
| Broader | 7 | 将区域记忆分割迁移到异构RGB–SAR三级BDA，并结合光学锚定 | 在BRIGHT外增加一个公开异构变化或实例BDA设置才可进一步提高 |

## 5. Paradigm-shift probe

| Probe | Yes or No | Rationale |
|---|---|---|
| First Principles | Yes | 挑战“BDA最终必须是像素分类”的默认假设，将结构区域设为主决策单位。 |
| Elephant in the Room | Yes | 建筑内标签不一致、稀有严重度和跨事件崩溃是公开且长期存在的问题。 |
| Technology Cycle | No | 超像素和memory均非近两年才可实现。 |
| Hamming's Rule | Partial | 若跨事件严重度显著改善会有实用价值，但单数据集增益不足以改变领域优先级。 |

Disruptive potential: possible（5/8）；更适合作为机制清晰的Technique论文，不宜过度包装成范式革命。

## 6. Feasibility

| Risk | Level | Mitigation |
|---|---|---|
| Compute | 中 | 先冻结P3，只训练分区和区域头；SR1通过后才启用SR2。 |
| Data | 低到中 | BRIGHT标签可用；训练memory合法，但不能使用validation/test写入。 |
| Engineering | 中高 | 可微scatter、memory checkpoint和多GPU一致性容易出错；用单元测试锁定。 |
| Timeline | 中 | 不实现未公开MMA-P/VMambaSSN；只做SR0–SR2。 |

## 7. Verdict

**Accept with Revisions — worth pursuing, pending the SR1 validation experiment.**

Top three actions to take first:

1. 运行SR0严格复现P3并确认输出逐元素相等。
2. 只运行SR1，验证“区域主分类”是否独立通过门槛；失败则停止检索路线。
3. SR1通过后运行SR2，并以随机邻居、非均衡memory和无memory做三项检索归因消融。

