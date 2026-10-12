# 跨事件、第二父模型与伪实例机制证据：2024–2026 插件筛选

- 用户研究方向：灾前 RGB + 灾后 SAR 的建筑损毁评估；在冻结 P3 与 fixed S2 的基础上进行支持域守恒的选择性伪实例细化。
- 用户研究问题：如何补齐跨事件稳定性、第二模型/第二设置、SRAC/BRIGHT Challenge 对比、伪实例几何指标与计算开销，并从本地 2024–2026 即插即用代码中筛选可迁移模块。
- 分析日期：2026-09-07
- 当前父系统：P3 + LocCC + A6 + fixed S2；Selective Strip-RGID E0 正在审计。

## 1. 先给结论

不建议继续把通用注意力模块接入 P3。当前优先级为：

1. **完成 Selective Strip-RGID R0–R4**，先确定候选分裂和选择策略是否成立；
2. **采用 BRIGHT CVPRW 2026 实例设置作为第二数据设置**，用官方实例标签只做冻结评估，补充粘连率、过分裂率、Boundary F1、PQ/mAP 与两个未见事件；
3. 若 R4 高而 R2 失败，增加一个 **Cut-band Edge Evidence E0**，首选 Scharr/LoG/Gaussian residual 的无参数边界证据；
4. 若边界 E0 通过，再把本地 EGA/HFP 思想改写成零初始化、GroupNorm、低容量的 RGID boundary branch；
5. 第二父模型首选 **SiamAttnUNet**，因为它与 Swin/DamageFormer 的归纳偏置明显不同，又能输出四类 logits；只复用 LocCC/fixed S2/Selective RGID 外挂流程，不共享 P3 特征；
6. 跨事件稳定性主要由 **严格事件划分、事件均衡训练和结构证据**提供，不应宣称某个注意力模块天然带来跨事件泛化。

## 2. 可迁移模块筛选

| 候选 | 本地来源 | 任务匹配 | 建议迁移 | 难度 | 判定 |
|---|---|---:|---|---:|---|
| Large Strip Convolution | 2026 `StripModule.py`；2025 同名版本 | 高 | 继续放在 RGID center/boundary head，不进入 P3 | 低 | **已采用，等待 R0–R4** |
| Edge-Gaussian / Scharr prior | 2025 `EdgeGaussianAggregation.py` | 高 | 先做固定 Scharr/LoG/Gaussian residual 的切割带 E0；通过后才训练轻量边界分支 | 中 | **首选后备模块** |
| High Frequency Perception | 2025 `HFP.py` | 中高 | 只增强 Stage3/RGID 边界证据；禁止复用已失败的 post-fusion FSA 接口 | 中 | **第二后备模块** |
| Cross-scale feature refinement | 2026 `CAFR.py` | 中 | 使用项目中已有零初始化 `CrossScaleBoundaryFusion` 替代原始 CAFR；只在边界门内注入 Stage3 | 中 | 条件采用 |
| TFFA 三域分支 | 2026 `TFFA.py` | 低中 | 不直接采用。当前代码的“wavelet”实际是可学习卷积，三分支共用一个门且无恒等初始化 | 高 | 停止 |
| Geometry Self-Attention | 2025 `GeometrySelf-Attention.py` | 低 | 原方法依赖真实深度几何；不能把 SAR 或 LocCC 任意伪装成 depth prior | 高 | 停止 |
| FFM / Cross Attention | 2025 `FFM.py` | 低中 | 可能重新引入同位 RGB–SAR 交互和错配；已有匹配/关系证据未通过 | 高 | 当前停止 |
| ARConv/Deformable Attention | 2025 ARConv/DSA | 中 | 只有边界候选上限高、Strip 表达不足时才作为等容量竞争头 | 高 | 暂缓 |
| RCM/FSA | 2024 | 低 | 当前接口已完成因果试验：RCM 负增益；FSA 基本未打开 | - | 正式停止 |

### 本地代码风险

- 本地插件多为单文件复现/整理版，不能默认等价于官方仓库。
- `EdgeGaussianAggregation.py` 片段只包含 Gaussian depthwise convolution，没有论文摘要所述的完整 Scharr 与不确定性建模，并依赖 MMCV。
- `HFP.py` 依赖 `torch_dct`，分组卷积隐含通道数可被 32 整除；原实现没有零初始化回退。
- `CAFR.py` 没有残差恒等路径，且全局池化会丢失候选切割带的位置证据。
- `TFFA.py` 的 DoG/Mexican 分支不是固定解析小波，不能据此声称采用了真实小波先验。

因此迁移时只借用论文机制，不原样复制单文件代码。

## 3. 推荐的模块插入方案

### 3.1 Cut-band Edge Evidence E0（无参数，优先）

```text
LocCC + Strip-RGID候选切割线
              ↓
在切割线两侧取宽度 d 的 band
              ↓
Pre-RGB: Scharr/LoG/Gaussian residual
Post-SAR: robust gradient/Gaussian residual
              ↓
统计边界连续性、两侧差异、噪声稳定性
              ↓
预测“候选分裂是否可信”的审计分数
```

第一轮不得训练，不改变 R2。报告候选 split utility AUROC、event-macro AUROC、10/14 事件、位置打乱对照和边界偏移稳定性。只有通过后才进入可学习分支。

### 3.2 Edge-Gaussian RGID（E1，E0 通过后）

```text
fused_roi(detach)
  ├─ StripSpatialBlock ──────────────┐
  └─ fixed Scharr/LoG edge prior ────┤
                                     ↓
                       zero-init edge residual
                                     ↓
                         center/boundary heads
```

约束：GroupNorm 替代 BatchNorm；输出增益参数零初始化；建筑支持必须完全不变；与等参数 `3x3` 和 Strip-only 比较。

### 3.3 CrossScaleBoundaryFusion（只有小建筑边界失败时）

```text
Stage4 semantic ───────────────────────┐
Stage3 detail ─ projection ─ boundary gate ─ zero-init residual
```

只供 RGID 边界头使用，不反馈 P3 damage logits。该分支必须证明小建筑 Boundary F1/merge rate 改善，而不仅是 mIoU。

## 4. 跨事件稳定性方案

### Level A：当前父模型上的事件稳定性（最低要求）

- 报告 14 个事件的 `delta mIoU`、Damaged F1、merge rate、over-split rate、Boundary F1；
- 报告 event macro、worst event、标准差、正增益事件数量；
- 使用图像级 paired bootstrap，并补充事件级 bootstrap；
- 至少两个种子方向一致；不允许只报告总体像素汇总。

### Level B：插件级 Leave-One-Event-Out（推荐）

对每个目标事件 `E`：

```text
其他13事件：训练Strip-RGID/选择risk策略
目标事件E：只评估，不参与阈值和超参数选择
```

如果 P3 仍由全事件训练，只能表述为“refinement cross-event transfer”，不能声称整个模型 zero-shot。建议先做 4 个代表事件筛查，再做完整 14 折。

### Level C：完整模型 Zero-shot（最强、最贵）

严格复用 BRIGHT 官方 cross-event protocol：一个事件完全留出，父模型、RGID 和阈值均不得看到该事件。此设置才支持“整模型未见灾害泛化”的强主张。

### 可迁移的训练策略

项目内 `train_DamageFormer_BGMP_K3EventBalanced_accum_ddp.py` 已实现逆事件频率权重、event-macro/worst 汇总与事件均衡思想。可迁移到 RGID：

- 每个实例损失先按图像平均，再按事件平均；
- 活跃事件等权，而不是让大事件/多建筑图像支配梯度；
- checkpoint 以 calibration event-macro 为主、worst-event 为安全条件；
- 不迁移其中的 prototype memory。

## 5. 第二父模型和第二数据设置

### 首选第二父模型：SiamAttnUNet

原因：CNN U-Net 双编码器与当前 Swin/DamageFormer 明显不同；仓库已有四类输出与训练/推理代码。适配流程：

```text
SiamAttnUNet四类logits
   ↓ softmax
建筑支持 = argmax(logits)>0
   ↓ LocCC
fixed S2
   ↓
同一个支持域守恒Selective RGID包装器
```

只比较“外挂前后增量”，不要要求 SiamAttnUNet 绝对精度超过 P3。先做 50 张 compatibility smoke，再做完整 calibration。

### 首选第二设置：BRIGHT CVPRW 2026 instance-level

它比简单更换父模型更有论文价值，因为官方实例标签可以直接计算：

- merge rate：一个伪实例覆盖多个 GT 建筑；
- fragmentation/over-split rate：一个 GT 建筑对应多个伪实例；
- boundary precision/recall/F1；
- instance covering/ARI/PQ；
- 三等级 instance mAP/AP50/AP75；
- 两个未见灾害事件上的泛化。

实例标签在本研究中优先只作为冻结评估真值，不能回流训练 Strip-RGID，否则“pseudo-only”主张失效。

## 6. 与 SRAC / Challenge 的正式比较

必须拆成两张表，不能混用指标：

1. **BRIGHT semantic setting**：四类 mIoU、Damaged F1、event macro。比较 DamageFormer、P3、P3+fixed S2、你的方法、SRAC（只有在相同 split/metric/输入下才列直接差值）。
2. **CVPRW 2026 instance setting**：mAP/AP50/AP75 和每类 AP。比较官方 Mask R-CNN baseline、两支公开 winning solution 与你的实例化输出。

对于无法获得相同 checkpoint/split 的 SRAC，只列“reported result”并注明不可直接横比；最好运行其官方代码或向作者获取评估配置。

## 7. 完整实验矩阵

### 主消融

| ID | 分区 | 选择 | 边界证据 | 回答问题 |
|---|---|---|---|---|
| R0 | LocCC | 无 | 无 | fixed S2 基线 |
| R1 | Strip-RGID 全部有效分裂 | 无 | 无 | 无条件分裂是否有效 |
| R2 | Strip-RGID | split-risk | 无 | 当前主方法 |
| R3 | Strip-RGID | 随机同数量 | 无 | 排除随机/数量效应 |
| R4 | LocCC/RGID 二选一 Oracle | GT，仅上限 | 无 | 候选分裂理论空间 |
| B0 | Strip-RGID | split-risk | cut-band fixed edge | 无参数机制审计 |
| B1 | Strip-RGID+edge branch | split-risk | 可学习零残差 | 边界机制增益 |

### 几何证据

按面积分层（small/medium/large）和灾害事件报告：merge rate、over-split rate、Boundary F1、中心召回、有效 child 数、support violation（必须为 0）。

### 复杂度

统一 1×RTX 4090、batch=1、1024×1024、warmup 20、计时 100 张，报告：

- parent-only、+fixed S2、+RGID-all、+Selective RGID；
- trainable/total parameters；
- FLOPs（可计算部分）；
- GPU forward、CPU partition/S2、端到端 latency；
- peak CUDA memory 与主机 RSS；
- images/s 和每 1000 张总耗时。

## 8. 准入规则

Strip-RGID 主线保留条件：

- R4−R0 ≥ +0.20 pp；
- R2−R0 ≥ +0.10 pp；
- R2 > R1 且 R2 > R3；
- paired bootstrap 95% CI 下界 > 0；
- 至少 10/14 事件不下降；
- Damaged F1 不下降；
- merge rate 或 Boundary F1 至少一项有显著改善，且 over-split 不恶化超过预注册容差；
- support violation = 0。

第二父模型要求方向一致，不强制达到与 P3 相同的绝对增益。实例设置要求至少在 merge rate、Boundary F1、instance AP 中两项改善。

## 9. 建议题目方案

1. **Support-Conserving Selective Pseudo-Instance Refinement for Multimodal Building Damage Assessment**
   - 主线：Utility 诊断 + Strip-RGID + risk-selective fixed S2。
   - 最小实验：R0–R4、14 事件、SiamAttnUNet、实例几何指标。

2. **When Building Context Hurts: Utility-Aware Pseudo-Instance Refinement for Optical–SAR Damage Mapping**
   - 主线：突出 corrected/harmed 与 Building Oracle，再给出选择性分裂方法。
   - 最小实验：harm audit、Oracle decomposition、Selective RGID、跨事件。

3. **Boundary-Reliable Pseudo-Buildings for Cross-Event Optical–SAR Damage Assessment**
   - 主线：仅在 R4 高而当前 R2 失败、cut-band edge evidence 通过后采用。
   - 最小实验：Strip-only vs fixed-edge vs learnable-edge、Boundary F1、未见事件。

4. **From Semantic Masks to Reliable Building Instances in All-Weather Damage Mapping**
   - 主线：把 2025 semantic setting 与 2026 instance setting连接起来。
   - 最小实验：mIoU + instance mAP/PQ + merge/fragmentation。

## 10. 证据来源

- BRIGHT 官方仓库与 cross-event protocol：一个事件留出为未见测试事件；官方也公开了 2026 instance challenge 设置与 winning solutions。
- BRIGHT Challenge 2026：任务采用 RGB+SAR、三类实例损毁和 segm_AP；最终两个事件在训练中不可见。
- Strip R-CNN：使用正交大条带卷积建模方向空间结构，原任务是遥感目标检测，迁移到 RGID 必须重新验证。
- LEGNet/EGA：正式方法包含 Scharr 边缘先验和 Gaussian refinement；本地单文件不是完整官方实现。
- HS-FPN/HFP：高频响应用于突出小目标；迁移目标应限定为小建筑/边界证据。
- DFormerv2：其几何先验来自 depth，不与 SAR 或伪实例标签天然等价，因此不直接迁移。

## 11. A/B 级模块的机制与 RGB–SAR 交互细化

### 11.1 StripSpatialBlock

作用对象不是损毁分类，而是伪实例中心/边界预测。顺序正交的 `1×k` 与 `k×1` 深度卷积沿建筑长轴聚合上下文，补足普通 `3×3` 对长屋顶、窄缝和粘连边界感受野不足的问题。它必须读取冻结并 detach 的 `fused_roi`，输出仅进入 center/boundary head。风险是核过大导致小建筑过平滑，因此需要按建筑面积报告几何指标，并与等参数 `3×3` 头竞争。

### 11.2 Edge-Gaussian / Scharr

Gaussian 卷积本身是低通，不是边缘算子。适配版应显式构造：Scharr 方向梯度、LoG 或 `x-Gaussian(x)` 高频残差、局部噪声尺度。第一阶段只在 RGID 候选切割带统计，不训练；第二阶段才通过零初始化残差注入 boundary head。SAR 分支必须先做稳健归一化/平滑，避免把 speckle 当作建筑边界。

### 11.3 HFP

HFP通过高通或 DCT 高频掩码增强小目标。对本任务，它只适合补充小建筑的 Stage3/RGID 边界细节。灾前 RGB 高频通常能表达屋顶轮廓；灾后 SAR 高频混合了结构与 speckle，不能对称使用。应先比较 `RGB-HFP`、`SAR-HFP` 和 `RGB+SAR-HFP`，预期优先保留 RGB-HFP；任何版本必须设置可回退增益并与固定梯度控制组比较。

### 11.4 CrossScaleBoundaryFusion

这是同一模态内的跨尺度融合，不是 RGB–SAR 融合。Stage4提供语义，Stage3提供细节，边界门限制细节注入位置，零初始化增益保证初始严格等价原特征。只在统计证明小建筑/窄缝 Boundary F1 明显差时开启；否则没有必要暴露新的 Stage3 接口。

### 11.5 推荐的非对称局部 RGB–SAR 交互

```text
Pre-RGB Stage3/4(detach) ─ query / geometry anchor ─────┐
                                                       ├─ 候选切割带内5×5局部匹配
Post-SAR Stage3/4(detach) ─ local keys / values ────────┘
                                  ↓
                confidence + residual + entropy + offset
                                  ↓
                     Split Verifier / risk score
                                  ↓
       接受候选分裂，或回退原LocCC；不修改P3 logits
```

原因：RGB是灾前清晰几何源，SAR是灾后变化源，二者不应完全对称。全图或同位 cross-attention 容易把配准残差、SAR speckle 和真实损毁混在一起。局部匹配允许 SAR token 在邻域内移动，并把置信度/位移显式暴露给选择器。

无参数 E0 特征仅在候选切割带及其两侧统计：

- `rgb_edge_support`：RGB边缘对候选切割的支持；
- `rgb_center_valley`：切割处建筑置信度是否形成稳定低谷；
- `sar_robust_edge`：去噪后SAR结构边缘；
- `match_confidence`、`match_entropy`、`offset_magnitude`；
- `matched_residual`：局部匹配后而非同位直接差；
- `child_purity_delta`：分裂前后P3类别纯度变化（不使用GT）。

目标标签只在 meta_train 由 R0/R1 的 GT utility 生成；目标事件、meta_confirm、Val450 和 test 不参与特征/阈值选择。

## 12. 下一步条件分支

1. 当前 R0–R4 未完成前，不启动 EGA、HFP、CrossScale 或跨模态训练。
2. 若 `R4-R0 < 0.20 pp`：Strip候选空间不足，停止选择器；先用实例GT诊断 center recall、merge/over-split 和 Boundary F1，再决定替换候选生成器。
3. 若 R4 高但 R2 失败：运行 `CutBand-RGB E0 → CutBand-RGB+SAR LocalMatch E0`；后者必须显著超过前者，才能声称 SAR 交互有贡献。
4. 若 R2 已通过：冻结主方法，优先补实例几何、跨事件和第二父模型；EGA/HFP只作为机制消融或后续工作，不堆进主模型。
5. 只有局部匹配 E0 达到 `AUROC>=0.60`、event-macro `>=0.58`、至少 `10/14` 事件方向正确且位置打乱后显著下降，才训练两层 MLP Split Verifier。

RGB–SAR 因果对照固定为：

| ID | 证据 | 目的 |
|---|---|---|
| X0 | 原 split-risk | 当前选择基线 |
| X1 | RGB cut-band edge | 几何证据是否足够 |
| X2 | SAR robust edge | SAR单独是否可靠 |
| X3 | RGB+SAR 同位差 | 错配敏感负控制 |
| X4 | RGB→SAR局部匹配 | 局部匹配是否修复同位差 |
| X5 | X4空间打乱 | 排除全局场景/强度偏差 |

只有 `X4 > X1、X2、X3、X5` 且最终 R2 改善，RGB–SAR 交互才能成为论文方法的一部分。

## 13. 已落地代码（2026-09-07）

新实验位于 `bda_benchmark/experiments/spi_cut_band_evidence_v1/`，没有修改 DamageFormer、P3 checkpoint、fixed S2 或 Strip-RGID checkpoint。

- `core.py`：内部候选切割带、RGB/SAR robust Scharr、局部证据汇总和 split utility；
- `build_cache.py`：冻结 P3 与 Strip-RGID，仅生成 meta_train/meta_calibration 缓存，支持断点；
- `audit_e0.py`：X0–X5 标量通道选择、事件宏平均、空间打乱与 permutation 对照；
- `train_verifier.py`：E0 通过后才能启动的 32-hidden 两层 verifier，使用事件均衡与效用加权；
- `run.sh`：preflight、smoke、audit、train 四个隔离入口；
- `test_core.py`：检查切割带不包含外轮廓、支持守恒、边缘响应和 utility 统计。

E1 不向 P3 注入新特征或残差，只在 `原 LocCC` 与 `RGID children` 之间二选一。因此若实验失败，可以完全回退当前 `P3 + LocCC + A6 + fixed S2`，不会改变已有基线归因。
