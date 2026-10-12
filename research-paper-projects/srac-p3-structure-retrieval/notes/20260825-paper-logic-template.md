# Technique Paper逻辑骨架：光学锚定的结构检索损毁分类

## 1. Paper-type positioning

- Type: Technique Paper（cross-domain technique）
- Rationale: BRIGHT BDA是既有任务，贡献是将任务特定结构分区与严格训练域区域记忆迁入光学–SAR层级损毁判断。

## 2. Thinking template

| Stage | Your content |
|---|---|
| Research background | BRIGHT表明灾前光学可稳定定位建筑，灾后SAR提供全天候结构证据；现有DamageFormer、DFC方法和BRIGHT Challenge结果说明跨事件严重度仍不稳定。 |
| Limitation 1 | 现有像素级融合最终独立预测每个像素，缺少建筑内部结构一致性；P3的MICB/DSMIL仅是训练辅助，推理仍由像素头决定。 |
| Limitation 2 | 少量类别原型或参数分类头会压缩灾害、地区和SAR成像条件下的多样化损毁模式，稀有Destroyed/Damaged容易被多数类淹没。 |
| Limitation 3 | 普通memory可能记住事件捷径，若验证/测试特征参与更新还会造成协议污染。 |
| Key Idea / Our Goal | 以光学建筑几何为锚，将损毁识别改为层级结构区域主分类，并从严格训练域、类别均衡的区域记忆中读取相似损毁证据。 |
| Challenge 1 | 区域必须由输入生成且与建筑/损毁边界一致；固定微小网格无法提供有效结构单位。 |
| Challenge 2 | 检索应补充稀有类别而不让Intact多数邻居支配，也不能把检索变成硬Router。 |
| Challenge 3 | 记忆需要跨事件稳定，同时保证validation/test只读和可审计。 |
| Methodology topic sentence | 提出光学锚定的层级结构检索框架，将P3稳定定位、可微区域和训练域记忆组合为区域主分类器。 |
| Module A (addresses Challenge 1) | RGB-guided Local Differentiable Partition：在P3 RGB特征上学习局部九候选超像素，并以区域主损失端到端优化。 |
| Module B (addresses Challenge 2) | Class-Balanced Region Retrieval：每类等容量memory、每类Top-k支持和软Cross-Attention读取。 |
| Module C (addresses Challenge 3) | Event-Safe Memory Protocol：memory只由训练标签更新、评估只读，并以leave-one-event-out验证事件捷径。 |
| Contribution 1 | 提出以结构区域替代像素损毁头的光学锚定RGB–SAR BDA框架。（Section 3） |
| Contribution 2 | 提出类别均衡、可审计的训练域区域检索机制，面向稀有损毁严重度。（Section 3.3） |
| Contribution 3 | 在BRIGHT标准与事件留出协议中进行决策单位、检索和memory污染控制消融。（Section 4） |

## 3. Self-consistency checks

- Check 1 Limitations -> Key Idea: pass
- Check 2 Key Idea -> Challenges: pass
- Check 3 Challenges -> Methodology: pass
- Check 4 Methodology -> Contributions: pass

## 4. Methodology outline

### 3.1 Optical-Anchored P3 Feature Extraction

- 保留双非共享Swin编码器。
- 保留RGB建筑定位与软建筑概率。
- 暂时冻结P3，隔离决策单位变化。

### 3.2 RGB-Guided Differentiable Structure Partition

- 从RGB融合特征学习任务特定像素Embedding。
- 每像素只关联周围九个种子。
- 区域统计包含关系mean/std/max与原始SAR mean/std。

### 3.3 Class-Balanced Training-Memory Retrieval

- 每个严重度类别使用相同容量。
- 每类独立Top-k，避免Intact数量优势。
- Cross-Attention只做软证据读取。

### 3.4 Event-Safe Training and Hierarchical Extension

- Train-only更新、validation/test只读。
- SR1通过后才实施事件留出memory。
- 平坦三级区域头通过后再拆Occurrence/Severity，避免过度设计。

## 5. Severity summary

- 0 CRITICAL, 2 MAJOR, 1 MINOR。
- Top three fixes first：获取SRAC全文并完成差异表；SR1先过受控门槛；用随机/非均衡邻居完成检索归因。

## 6. Next suggested skill

- 当前不进入Introduction写作；先运行SR0/SR1。SR1通过后再使用`intro-drafter`。
