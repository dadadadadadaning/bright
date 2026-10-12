# PIUM：文献、既有代码失败与新实验实现审计

- 用户研究方向：灾前RGB/灾后SAR建筑损毁评估
- 研究问题：如何用检索Memory改善P3+fixed S2，同时避免像素Gate、事件泄漏和直接类别覆盖造成的失败
- 日期：2026-09-06
- 新代码：`bda_benchmark/experiments/spi_pseudo_instance_memory_v1/`

## 1. 文献与官方代码结论

| 工作 | 可核验来源 | 可迁移点 | 不直接照搬的部分 |
|---|---|---|---|
| SRAC, IGARSS 2026 | IGARSS录用列表、用户提供架构图、实验室介绍；未检索到官方代码/全文 | 超像素token、外部标注库、Top-R邻居、结构元数据、Cross-Attention | 未公开的Memory容量、损失、检索策略不能臆测；当前不直接训练其完整框架 |
| RAC, CVPR 2022 | CVF论文 | 非参数Memory作为并行分支服务长尾类别 | 其目标是类别分类，本项目先改为S2效用预测 |
| Iscen et al., CVPR 2023 | CVF论文 | 检索邻居有好有坏，需要学习/估计邻居重要性 | Attention必须等无参数检索通过后再训练 |
| RIO, ICML 2021 | NVIDIA官方代码 | 长尾检测需要object-level而非image-level memory sampling | 本项目没有真值实例，因此使用detached LocCC伪实例并显式审计噪声 |
| SSN, ECCV 2018 | NVIDIA官方代码 | 可微超像素可以构造区域token | fixed SLIC尚未通过，不允许当前引入可微SSN |
| SelectiveNet, ICML 2019 | PMLR论文 | 拒绝选项必须报告risk-coverage权衡 | PI-M0只做可行性；PI-M1才产生选择器 |
| FAISS | Meta官方仓库/文档 | Memory规模扩大后可做批量内积检索 | 当前规模先用PyTorch精确检索，避免ANN误差成为混杂变量 |

## 2. 既有代码中应继承的安全设计

仓库以下代码具有可复用价值：

- `spi_s2_evidence_ladder_v1/memory_bank.py`：P3与split指纹、禁用Validation/Test；
- `rm_region_retrieval_runtime.py`：类别/事件平衡、排除同事件、邻居事件多样性；
- `model/cmr/multiprototype.py`：事件主导诊断、跨事件相似性审计；
- `spi_building_utility_v1`：固定LocCC定义、建筑级utility标签、三分数据隔离。

## 3. 既有失败带来的约束

1. Pixel Utility Gate AUROC为0.409且接受率为0：不能继续逐像素监督。
2. fixed S2有正净收益但约40.1%修改有害：不能让Memory无条件替代S2。
3. Direction Policy只有2/14事件优于fixed S2：不能只按修改方向检索。
4. Relation/Matching/SLIC单通道均未准入：不能把这些弱信号直接宣称为有效模块。
5. FSA近似零作用、RCM显著负作用：不重新训练SAR适配器作为Memory前置条件。
6. 旧类别Memory可能成为第二分类器：新Memory只能预测`S2 utility`，输出严格限制为接受S2或回退P3。

## 4. 新方案PI-M0

### 决策单位

一个P3 LocCC对应一个查询，标签为：

`utility_B = (corrected_B - harmed_B) / changed_B`。

正负阈值沿用BU-E0的`+/-0.10`，中间样本不进入监督Memory。

### 数据路径

```text
BU-E0 meta_train cache
  -> event × utility label平衡Memory
  -> meta_train leave-one-event-out选择representation/k/tau/signal
  -> 只冻结一个正式配置

BU-E0 meta_calibration cache
  -> 正式跨事件评估一次
  -> geometry/base预注册控制

meta_confirm / Val450 / Test
  -> 不访问
```

### 表征

第一版复用BU-E0缓存而不重新前向P3。所有输入均为部署可获得的P3/S2/图像证据；GT只生成训练效用标签。采用Memory-only median/IQR归一化、异常值裁剪与L2归一化。

### 检索

- 每个utility类别独立Top-k；
- 每个事件最多贡献固定邻居数；
- 正式结果排除同事件；
- 始终排除相同伪实例ID；
- 不足k个邻居则标记unavailable；
- PyTorch批量精确余弦检索。

### 控制

geometry-only、base、same-event泄漏诊断、shuffled Memory标签、random score、target permutation null。

### 决策

只有Building Oracle先通过，且PI-M0跨事件AUROC、事件宏平均、事件数量、可用率、几何增益、置换检验和标签打乱控制全部通过，才允许PI-M1两层MLP。PI-M0本身不修改任何预测。

## 5. 代码文件

```text
spi_pseudo_instance_memory_v1/
├── core.py          # 稳健键、事件平衡Memory、跨事件Top-k、指标和置换检验
├── audit.py         # BU结果/指纹门控、train-LOEO选择、calibration一次评估
├── test_core.py     # 常数通道、平衡抽样、同事件/自身排除测试
├── run.sh           # 固定bright4090 Python与服务器入口
├── README_ZH.md     # 协议、阈值和运行说明
└── SERVER_SYNC_FILES.txt
```

## 6. 后续实验

1. BU-E0 Oracle不过：停止PIUM。
2. BU-E0过、PI-M0不过：停止Memory，保留fixed S2。
3. PI-M0通过：实现PI-M1低容量建筑Verifier并报告risk-coverage。
4. PI-M1通过：增加Stage3 core/boundary token，而非立刻可微超像素。
5. 固定SLIC层级检索优于等预算Grid后：才允许SSN和Cross-Attention。

## 7. 创新与难度

- 任务匹配：高；Memory与伪实例决策直接对应。
- 创新度：中高；创新来自“跨事件检索S2净效用 + 安全拒绝”，不是简单使用Memory。
- 迁移难度：PI-M0低，PI-M1中，层级超像素Memory高。
- 主要风险：伪实例合并、事件捷径、Memory标签噪声、几何捷径、校准样本不足。

## 8. 建议题目

1. **Retrieve to Accept or Reject: Cross-Event Utility Memory for Multimodal Building Damage Assessment**
2. **Safe Context Refinement with Pseudo-Instance Retrieval for RGB-SAR Damage Mapping**
3. **Utility-Aware Memory under Noisy Building Pseudo-Instances**
4. **Hierarchical Building-to-Superpixel Retrieval for Rare Disaster Damage Recognition**

## 9. 来源

- SRAC录用：<https://2026.ieeeigarss.org/papers/accepted_papers.php>
- SRAC实验室介绍：<https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026>
- RAC：<https://openaccess.thecvf.com/content/CVPR2022/papers/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.pdf>
- Retrieval-Augmented Image Recognition：<https://openaccess.thecvf.com/content/CVPR2023/papers/Iscen_Improving_Image_Recognition_by_Retrieving_From_Web-Scale_Image-Text_Data_CVPR_2023_paper.pdf>
- RIO官方代码：<https://github.com/NVlabs/RIO>
- SSN官方代码：<https://github.com/NVlabs/ssn_superpixels>
- SelectiveNet：<https://proceedings.mlr.press/v97/geifman19a.html>
- FAISS索引建议：<https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index>

