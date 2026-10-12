# BRIGHT 实验档案总审计（2026-09-14）

## 1. 审计范围

- 训练日志：`D:\software\dowmload\BRIGHT-train-logs`
- 评估结果：`D:\software\dowmload\BRIGHT-eval-results`
- 实验代码：`D:\software\dowmload\BRIGHT-master\bda_benchmark\experiments`
- 共清点 1,829 个文件，解析 283 条训练日志结果、227 个主要 JSON 文件和 18,875 条嵌套指标记录。
- 完整逐文件清单见 `all_files.csv`，训练结果见 `training_log_results.csv`，JSON 指标见 `json_metric_records.csv`，Test 清单见 `test_results.csv`。

## 2. 必须统一的指标口径

本项目日志中存在两个容易混淆的指标：

- `valid_mIoU`：只平均 Intact、Damaged、Destroyed 三个建筑类别，是当前主要选择指标。
- `mIoU4`：包含 Background 的四类平均。它通常比 `valid_mIoU` 高约 9～12 个百分点。

因此 P3 的 Validation 是 `valid_mIoU=60.5019%`，不是 69.5772%；后者是四类 mIoU。

## 3. 权威基线

| 模型/划分 | valid mIoU | mIoU4 | Damaged IoU/F1 | Destroyed IoU | 说明 |
|---|---:|---:|---:|---:|---|
| P3 / Val450 | 60.5019 | 69.5772 | 43.0309 / 60.1701 | 62.8957 | 当前主网络基线 |
| P3 / Test878 | 57.4086 | 67.2725 | 40.0857 / 57.2303 | 56.9040 | 已经访问的官方 Test |
| IA-MLA / Test878 | 45.3505 | 57.8863 | 25.7877 / 41.0019 | 42.8934 | 明显弱于 P3 |

权威文件：

- `D:\software\dowmload\BRIGHT-eval-results\S8_pair\val\S8_MICB_DSMIL_P3\summary_official.json`
- `D:\software\dowmload\BRIGHT-eval-results\S8_pair\test\S8_MICB_DSMIL_P3\summary_official.json`
- `D:\software\dowmload\BRIGHT-eval-results\S8_pair\test\S8_IA_MLA\summary_official.json`

## 4. 各路线结论

### 4.1 Selective Context / 伪实例后处理

Val450 上最强的完整结果：

| 变体 | valid mIoU | 相对 P3 | Damaged F1 | 结论 |
|---|---:|---:|---:|---|
| P3 | 60.5020 | 0 | 60.1701 | 基线 |
| S1 LocCC hard | 61.5399 | +1.0379pp | 61.2905 | 有效强基线 |
| S2 LocCC soft context | **61.5716** | **+1.0696pp** | **61.2925** | 当前 Val450 最值得保留 |
| S3 uncertain interior | 61.1444 | +0.6424pp | 60.7637 | 次优 |
| S4 local uncertain control | 60.5080 | +0.0060pp | 60.1768 | 基本无效 |

注意：旧脚本的预设总准入候选是 S3 `uncertain_interior`，不是 fixed S2。S3 的 Damaged F1 只提高约 0.59pp，因此其 `pass=false` 正常；fixed S2 的 Damaged F1 提高约 1.12pp，但旧脚本没有单独给它生成总准入决定。该路线尚无 fixed S2 Test 结果，不能把 61.5716 写成 Test 成绩。

### 4.2 P5：SAR 输入与 Cross-Attention

| 变体 | 结构 | Val valid mIoU | 判断 |
|---|---|---:|---|
| E1 | 原生 SAR + 拼接 | 56.1679 | 低于 P3 |
| E2 | 固定滤波 SAR + 拼接 | 56.4699 | 比 E1 +0.3020pp，但 Damaged F1 下降，14 个事件仅 5 个不降，未通过 |
| E3A | 原生 SAR + 局部 Cross-Attention | 55.2175 | 最差，不支持“单加注意力” |
| E3 | 滤波 SAR + 局部 Cross-Attention | 57.8419 | 比 E2 +1.3720pp，12/14 事件提升，CI95%=[+0.652,+2.047]pp；但错位稳定性检查失败，且仍低于 P3 2.66pp |

结论：E3 只证明 Cross-Attention 在 P5 弱基线内部有效，不能证明它优于 P3。它适合保留为机制消融，不适合直接替换 P3。

### 4.3 P3 + P5 安全残差

这一想法已经以 `S8_P5_C2_R1` 跑过，不需要重复训练：

- P3 初始：60.5019%
- 最佳：60.5108%，仅 +0.0089pp
- 训练到 6000 步：60.2669%，开始退化
- Damaged IoU 没有改善，增益主要是极小的 Destroyed 波动

结论：全局零门控残差几乎没有产生有效增益。新建的 `p3_p5_safe_residual_v1` 只能作为独立复现/审计入口，不能当作未验证的新创新。

### 4.4 SALT / CDE

代表性 Val 结果：D0 58.2719、D1 58.1147、D2 58.4368、D2R 58.4155、D3 57.7481。均未超过 P3。继续简单增加训练步数不合理；应停止旧 D2/D3，除非改变监督目标或结构假设。

### 4.5 SEC / 建筑约束对齐

- S1/S2 的初始最好值约 58.1147%，来自 D1 初始化。
- S2 训练结束 clean 约 57.8978%，最佳仍是 step 0/4000 附近，未形成可靠提升。
- S2R 最新审计中 clean 约 56.0550%，位移响应只达到目标的一小部分。

结论：当前 offset/equivariance 监督没有学会可用的几何对齐；S2R、S2RCF 不应连续盲跑。先用可视化与位移 MAE 证明对齐头真的会移动，再训练损毁头。

### 4.6 H6 / OA-HMD 等较高 Validation 结果

| 实验 | Val valid mIoU | 相对 P3 | 风险判断 |
|---|---:|---:|---|
| H6 B1 | 60.7586 | +0.2567pp | 开发过程中反复监看 Test；相应 Test 约 57.28，低于 P3 |
| H6 A2 F1 | 60.7324 | +0.2305pp | 单种子、小幅提升；尚无对应独立 Test |
| H6 A2 F0 | 60.7202 | +0.2183pp | 与 F1 差异极小，模块归因弱 |
| H6 A2 DG0 | 60.7196 | +0.2177pp | 控制组几乎同样高，说明提升未必来自目标模块 |
| H6 A2 F2 | 60.6801 | +0.1782pp | 小幅提升 |
| OA-HMD C1 | 60.6008 | +0.0989pp | 很小，Damaged IoU几乎不变，最差事件下降 |

这些结果说明训练配方/冻结方式可带来约 0.1～0.25pp 的 Val 波动，但控制组也同样提高，暂时不能作为强结构创新。

## 5. Test 结果与协议风险

已经找到至少 13 份 Test 结果。主要结果按 valid mIoU 排序：

| 模型 | Test valid mIoU | 与 P3 Test 的关系 |
|---|---:|---:|
| H6-B1R seed7777 | 57.4100 | +0.0014pp，数值上等同 P3 |
| P3 | 57.4086 | 主基线 |
| H6-A3 | 56.1141 | -1.2945pp |
| H6-A2 / H8-C1 | 56.1105 | -1.2981pp |
| CCU-E1 | 56.1074 | -1.3012pp |
| CCU-E3 | 56.0924 | -1.3162pp |
| H10-H3 | 56.0103 | -1.3983pp |
| H11-T1 | 55.9975 | -1.4111pp |
| H11-T1E | 55.9850 | -1.4236pp |
| A2-M1 | 54.5258 | -2.8828pp |
| CCU-C0 | 48.7753 | -8.6333pp |
| IA-MLA | 45.3505 | -12.0581pp |

重要协议结论：Test 已经被访问，而且部分 H6 训练日志在开发过程中反复打印 `Test monitor`。从现在开始不能再把这个 Test 当成完全未见集合反复选模块、阈值或训练步数。论文中应如实说明历史访问；后续开发用 Val450/meta-confirm/LOEO，最终结论需要新的锁定集合或严格一次性评估。

## 6. 目前真正可以保留的内容

1. **P3 是最可靠主网络。** 它同时拥有完整 Val/Test 基线，Test 仍为当前最佳水平。
2. **fixed S2 是当前最有效的 Val450 增益。** +1.0696pp，且 Damaged F1 提高；需要独立确认，而不是继续在 Val450 调阈值。
3. **E3 Cross-Attention 是有效的内部机制证据。** 相对 E2 的提升稳定，但尚不能接替 P3，也没有证明错位鲁棒性。
4. **H6 冻结训练策略值得做多种子复核。** 但模块组与控制组差距太小，创新归因尚不成立。
5. **Metric-aligned cluster verifier 已经做过，不能再列为未运行方向。** M0 Oracle 证明存在理论空间；线性 E1 在 meta-confirm 相对 fixed S2 提高 +0.3524pp、回收 27.13% Oracle，但 Damaged F1 下降、仅 9/14 事件不降且 bootstrap CI 下界为 -0.0652pp，因此正式决定为 `STOP_MACU_E1_AFTER_META_CONFIRM`。

## 7. 建议的下一步（按优先级）

1. 将 P3 + fixed S2 作为当前候选系统冻结；不要再在 Val450 或已访问 Test 上选阈值。
2. 不重复 M2/M3 Metric-aligned 旧实验。若继续 Verifier，必须增加新的独立证据或改成损失约束明确保护 Damaged F1，并建立新的 confirm 划分；仅延长现有线性模型没有意义。
3. 对 H6-F1、DG0 控制各补 3 个种子，只比较 Val/新 meta-confirm。若 F1 不能稳定超过 DG0 至少 0.2～0.3pp，就不把模块写成贡献。
4. Cross-Attention 仅做对齐机制分支：补完整 x/y 方向和 0/4/8/16 像素下降曲线；错位稳定性仍失败就停止该论点。
5. 不重复跑 P3+E3 全局安全残差，不继续加长 SALT/SEC 的旧失败配置，也不继续已经被 STOP 的 memory/retrieval、SAR adapter、旧 utility 路线。

## 8. 本地代码检查结果

- 档案汇总脚本通过 Python 语法检查，并已实际生成全部 CSV/JSON 清单。
- `p3_p5_safe_residual_v1/audit.py` 通过语法检查。
- P5 指标计算的 3 个纯本地单元测试全部通过。
- 依赖模型前向的 P3/P5 recovery、OA-HMD、Selective Context 测试在本机未运行，原因是本地 Python 没有安装 PyTorch；这不是代码断言失败。GPU 前向/训练必须在服务器 `bright4090` 环境完成。
