# RGB–SAR 变化检测与建筑损毁评估

- 建立日期：2026-08-18
- 核心问题：灾前高分辨率 RGB 与灾后单通道高分辨率 SAR 的建筑损毁评估，重点关注跨灾害事件泛化。
- 当前基线：P3（稳定但使用伪 RGB SAR 接口）、H6-A2（原生单通道 SAR 专家）、CCU/CCRR-BDA（受控跨模态差异与局部对应实验）。

## 项目文件

- `searches/20260818-rgb-sar-coded-top-venue.md`：筛选后的论文与代码清单。
- `searches/20260818-rgb-sar-coded-top-venue.jsonl`：机器可读证据记录。
- `notes/20260818-project-insight.md`：面向 BRIGHT 项目的迁移分析、互补组合和实验设计。
- `notes/P001-hetecd.md`：HeteCD 出版社摘要 + 官方代码级精读笔记。
- `notes/P002-dstcd.md`：DSTCD 出版社摘要 + 官方代码级精读笔记。

## 当前研究决策

1. 不再从随机初始化重新训练整套 RGB–SAR 网络。
2. 先提升 A2 的 SAR 表征，再判断跨模态信号能否纠正专家错误。
3. E0-C 只做验证集诊断，不训练、不保存模型、不访问测试集。
4. 只有 E0-C 证明“可预测 A2 何时有益”后，才实现稀疏残差融合。
