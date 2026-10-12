# CMR Memory 下一阶段：隔离代码与实验协议

## 1. 为什么不继续直接融合

已有实验已经证明：Raw Mean 的 Top-20 中通常包含正确 Damaged 案例，但当前全局
Embedding、Pairwise Scorer 和 Margin 无法稳定区分“救回 Damaged”与“误伤
Intact”。因此，新代码不再扩大 Memory 的推理权限，而是把尚未回答的三个问题
拆开验证。

```text
M0：Damaged 是否存在稳定、跨事件的多个原型？
V0：Mean 负责召回后，局部 Token 能否验证建议是否可信？
DQ：如果 Memory 只用于训练，动态 Teacher 队列能否稳定产生跨事件困难配对？
```

三条实验都满足：不修改 P3 文件、不覆盖旧 CMR 结果、不访问 Test、未通过准入前
不改变最终预测。

## 2. 文件边界

### M0：无训练多原型审计

- `model/cmr/multiprototype.py`
- `script/standard_ML/audit_CMR_M0_multiprototype.py`
- `run/run_CMR_M0_multiprototype.sh`
- `tests/test_cmr_m0_multiprototype.py`

M0 只读取 Train 的冻结 Mean Key。它比较 K=2/3/4/6，并检查多种子稳定性、每个
原型的事件覆盖、单事件占比、聚类与事件标签的 NMI，以及同原型跨事件相似度增益。
如果聚类主要等于灾害事件划分，就停止“多原型 Damaged”路线。

### V0：Mean 召回 + 局部 Token 验证

- `model/cmr/local_verifier.py`
- `script/standard_ML/build_CMR_V0_token_cache.py`
- `script/standard_ML/audit_CMR_V0_local_verifier.py`
- `run/run_CMR_V0_local_verifier.sh`
- `tests/test_cmr_v0_local_verifier.py`

V0 保留 Raw Mean Top-20 候选不变，只为候选建议提取冻结 P3/MICB 建筑内部 Token。
训练事件采用 LOEO Logistic 校准并锁定阈值，Validation 只运行一次。A/B/C 对照为：

```text
A：P3 + Mean 置信度
B：局部 Token 双向相关统计
C：A + B
```

只有 C 的 AUROC、相对 A 的增量和锁定阈值正收益同时通过，才允许进入后续的
P3 锚定小范围救援 Screen；V0 本身仍不改 P3 答案。

### DQ：Train-only 动态队列合同

- `model/cmr/dynamic_queue.py`
- `script/standard_ML/audit_CMR_DQ_contract.py`
- `run/run_CMR_DQ_audit.sh`
- `tests/test_cmr_dynamic_queue.py`

DQ 按“类别 × 事件”建立独立 FIFO 单元，写入 detached Teacher 特征。查询时强制
排除同事件，并限制单一事件在邻居中的数量。Destroyed 暂不作为 Damaged 强负样本。
因此准入只要求 Intact 与 Damaged 的有效事件覆盖，不要求本来就被 Ignore 的
Destroyed 在每个事件中都存在；所有空单元仍会写入 JSON 供数据诊断。
当前只实现队列核心和无训练合同审计；只有合同通过并解决现有 H1 精确区域回放问题，
才允许把 DQ 接到 Student 的 Occurrence 训练。

## 3. 推荐执行顺序

```bash
bash run/run_CMR_M0_multiprototype.sh all
bash run/run_CMR_V0_local_verifier.sh all
bash run/run_CMR_DQ_audit.sh all
```

优先看 V0，因为它直接回答当前最关键的“局部证据能否减少 Intact 误伤”。M0 和 DQ
是独立诊断，不应因为其中一个通过就直接进入 Test。

## 4. 决策规则

```text
V0 通过
→ 只做 P3 锚定、区域级 accept/reject Screen
→ Validation 多事件稳定后再讨论 Test

V0 失败，M0 通过
→ 用原型仅对 Damaged 支持集分层，再重跑同一 V0 协议

V0、M0 都失败
→ 停止推理 Memory；保留 DQ 作为 Train-only 困难样本工具

DQ 失败
→ 不接 H1；先修队列质量或区域回放合同
```

任何阶段都不能用 Validation 扫阈值、挑 K 或选择局部描述子后再报告同一次
Validation 结果。所有选择必须来自 Train-LOEO，Test 只在完整方法冻结后访问一次。

## 5. 区域回放与层级审计补充

旧P0只保存BBox和面积，不能完整证明在线区域身份。`run_CMR_V0S0_replay.sh`
会报告精确匹配率以及BBox IoU、面积误差、Mean Key余弦和概率差，但不会据此自动
模糊匹配。若仅“受保护身份”通过，应在同一运行环境重建P0/M3/V0缓存。

`run_CMR_hierarchy_audits.sh`包含三个无训练阶段：E0拆分Occurrence与Severity；
M0-B在每事件最多200个Damaged的子集上拟合和检查事件泄漏，并在全部自然样本上
检查表示收益；M0-C只有在M0-B通过后才执行14折LOEO迁移审计。
