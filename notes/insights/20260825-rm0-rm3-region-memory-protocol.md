# RM0-RM3：BRIGHT 训练集区域记忆与无训练检索验证协议

- 用户研究方向：灾前 RGB、灾后原生 SAR 的建筑损毁评估
- 用户研究问题：在训练 Cross-Attention 和可学习超像素前，验证类别均衡区域检索是否真的携带稀有损毁证据，并排除同事件捷径与标签泄漏
- 分析日期：2026-08-25
- 论文来源：SRAC（IGARSS 2026 已录用，公开全文/代码尚未核实）、Tip-Adapter（ECCV 2022）

## 1. 总体原则

RM0-RM3 全部不训练新网络，也不修改 P3 输出。它们只回答三个问题：

1. 冻结 P3 的建筑区域特征是否能检索到同损毁等级训练区域；
2. 原始 SAR 均值/标准差是否提供额外信息，还是只编码事件/传感器风格；
3. 检索收益是否在标签打乱和排除同事件后仍成立。

只有上述证据成立，才允许训练 Cross-Attention。旧 SR2 不直接使用，因为它把在线 FIFO memory、区域分类器和 Cross-Attention 耦合在一起，无法提供上述因果证据。

## 2. 区域定义

第一阶段不使用尚未通过质量门槛的软超像素。训练与验证统一使用：

```text
冻结 P3 RGB 定位概率
→ 固定阈值二值化
→ 4 邻域连通区域
→ 面积和巨型合并区域过滤
```

训练集区域标签来自该区域与训练 GT 的重叠多数类；验证 GT 只用于事后评估，绝不参与区域生成或检索。建议过滤条件：最小面积 32 像素、GT 建筑覆盖率不低于 0.50、I/D/X 多数类纯度不低于 0.75。默认使用 P3 区域构建训练 memory，以保持训练 memory 与部署查询的几何分布一致；GT 区域仅作为 oracle 对照。

## 3. RM0：导出只含训练集的静态 Memory

冻结 P3 并遍历 `train_set.txt`。每个合格区域保存：

```text
key              P3 final_feat_fused_roi 的区域加权均值并 L2 归一化
label            0=Intact, 1=Damaged, 2=Destroyed
event_id         灾害事件名称，仅用于审计和排除，不进入 RM1 相似度
image_id         图像标识，用于排除自匹配/重复
bbox / area      区域几何
purity / coverage 训练 GT 标签质量
sar_mean/std     原始 8-bit SAR 的区域统计，RM0 只保存但 RM1 不使用
```

输出应包含 extractor checkpoint 路径或哈希、split 文件哈希、阈值和过滤配置，且声明 `source_split=train`。先收集所有区域，再按固定 seed 在每类中等量抽样；不能使用顺序相关的 FIFO 覆盖。推荐第一版每类最多 5,000 个区域。

建议文件：

```text
bda_benchmark/model/retrieval/offline_region_memory.py
bda_benchmark/script/standard_ML/build_RM0_train_region_memory.py
bda_benchmark/tests/test_rm_region_memory.py
```

输出：

```text
/root/shared-nvme/BRIGHT-checkpoints/RM_RETRIEVAL/
RM0_train_region_memory_seed3407.pt
RM0_train_region_memory_manifest.json
```

RM0 必须检查：三类容量相同、无 val/test ID、key 全部有限且单位化、每个 memory 条目能追溯到一个 train 图像、重复率和逐事件数量可见。

## 4. RM1：无训练类别均衡 kNN

验证集查询仍由冻结 P3 预测建筑区域产生，并使用与 RM0 完全相同的特征池化。对每个类别分别取 top-k，避免 Intact 数量支配检索：

\[
s_c(q)=\frac{1}{k}\sum_{i\in \operatorname{TopK}_c}
\exp[-\beta(1-\cos(q,k_i))].
\]

三类分数归一化后得到训练自由的 I/D/X 概率。第一版固定 `k=8, beta=5.5`，不扫大量超参数。RM1 只比较检索分类与 P3 区域预测，不回写像素 logits。

必须输出：Macro-F1、I/D/X Recall/F1、event-macro F1、近邻纯度与相似度，以及四格互补矩阵：P3 对/检索对、P3 对/检索错、P3 错/检索对、两者都错。关键量为：

\[
\text{NetComplement}=N(P3\ wrong,KNN\ right)-N(P3\ right,KNN\ wrong).
\]

重点单独报告 Damaged。若 `P3错、KNN对` 很少或净互补为负，禁止进入 Cross-Attention。

## 5. RM2：增加原始 SAR mean/std

使用原始 SAR 显示值、在任何 RGB 复制或网络归一化前统计区域均值与标准差。所有标准化参数只能由 RM0 训练 memory 计算：

```text
z_mean = (region_mean - train_mean_of_means) / train_std_of_means
z_std  = (log(region_std+eps) - train_mean_logstd) / train_std_logstd
```

组合 key：

\[
\tilde{k}=\operatorname{Norm}([k_{visual},\lambda z_{mean},\lambda z_{std}]).
\]

建议只比较预声明的 `lambda={0,0.25,0.5,1.0}`。`lambda=0` 必须精确复现 RM1。选择时优先 event-macro 和 event-worst，而非只看总体 Macro-F1。

RM2 的解释：若 SAR stats 提升 Damaged 且多数事件非负，它可能提供可靠散射强度线索；若总体提高但同事件排除后崩溃，则它主要是事件/传感器指纹。

## 6. RM3：反事实和跨事件控制

### 6.1 标签打乱

保持 keys、事件、区域数量和类别总数不变，只随机置换 key-label 对应关系，运行至少 20 个 seed。真实检索结果应显著高于打乱均值（建议高于均值 3 个标准差），且 Damaged 净互补应在打乱后消失。否则应怀疑评估泄漏、类别先验或实现错误。

### 6.2 排除同事件

对事件 `e` 的每个验证 query，在相似度 top-k 前把 memory 中 `event_id==e` 的条目全部 mask。不能先检索再删除，因为会改变有效 k。额外排除相同 image/AOI 或已知地理重复。

必须同时报告：普通 kNN、同事件排除 kNN、标签打乱 kNN、类别先验和随机 key 基线。

RM3 通过建议：跨事件版本的 Damaged 净互补为正；真实结果高于 shuffle mean+3std；至少 10/14 事件不劣；同事件排除后保留普通检索收益的 80% 左右；不存在 train/val/test ID 交集。若普通检索有效而跨事件失败，结论只能是事件记忆，不能声称跨事件损毁证据。

## 7. 通过后才训练 Cross-Attention

每个 query 区域从每类取 R 个跨事件邻居，support token 由 memory key、label embedding 和安全几何信息组成：

```text
query region token
→ Q

top-R train-only supports
+ class label embedding
+ area/aspect metadata
→ K,V

Cross-Attention → FFN → conditional I/D/X
```

训练 query 来自 train，但必须排除同图像，论文若主张跨事件稳健则还必须排除同事件。验证/测试 memory 永远只读。第一版保留 P3 的 Background/Building 概率，只学习条件 I/D/X，不允许 memory 修改建筑定位。

## 8. 超像素与元数据何时加入

超像素把数亿像素查询压缩为数万区域查询，使 kNN 复杂度从像素级的不可承受规模降到区域级可分块计算；但速度不是唯一理由，它还提供标签一致的 memory 单位。当前软超像素分区未通过，因此先用 P3 连通区域验证检索。只有 RM0-RM3 通过后，才把可学习超像素替换为新的 region generator，并要求检索指标优于 P3 连通区域。

事件名和地区名第一阶段只用于审计/排除，不能直接注入分类器。灾害类型若在真实推理时可获得，可以后置加入并做 leave-one-type-out；地理地区容易造成位置记忆，默认不进入模型。优先允许面积、长宽比、建筑置信度等部署时稳定可得的几何元数据。

## 9. 复用与新增代码边界

可复用：

- `DamageFormer_S7_SIRM_M2B.py` 的 `extract_instance_queries`、静态 key-value memory 和 Tip-Adapter affinity；
- `DamageFormer_P3_SRAC_Controlled.py` 的按类检索与 raw SAR mean/std 区域统计。

不能直接复用为最终 RM 协议：

- `ClassBalancedRegionMemory` 当前是在线 FIFO，存在样本顺序偏差且缺少 event/image metadata；
- `RegionRetrievalClassifier` 已经包含 Cross-Attention，会跳过 RM1-RM3 的无训练验证；
- 旧 SR2 依赖未通过质量门槛的软分区。

建议新增统一入口：

```text
run/run_RM_region_retrieval.sh preflight
run/run_RM_region_retrieval.sh rm0_build
run/run_RM_region_retrieval.sh rm1_val
run/run_RM_region_retrieval.sh rm2_val
run/run_RM_region_retrieval.sh rm3_controls
run/run_RM_region_retrieval.sh report
```

测试集在 RM3 和训练后 Cross-Attention 的 validation gate 通过前不得访问。

## 10. 证据与迁移判断

SRAC 已被 IGARSS 2026 官方录用列表确认，机构介绍公开描述了超像素、标注库检索和稀有类别增强，但截至当前未核实公开全文或官方代码，因此本协议只迁移公开可确认的模块思想，不声称复现其内部实现。Tip-Adapter（ECCV 2022）提供了无需训练的 key-value cache 与特征相似度投票依据。对当前项目而言，真正可能形成区别的不是普通 memory bank，而是 train-only、类别均衡、事件排除和 RGB 几何/native-SAR 严重度解耦。
