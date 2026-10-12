# Memory Roles V1

该实验共享完整建筑卡，在一个连续队列中训练和筛选五种 Memory 用途：

1. R-M1 跨事件度量教师：B0 P3 全局、B1 固定完整卡、B2 学习完整卡、B3 无事件均衡、B4 无困难负样本、B5 硬多数标签，并附加 C1 打乱标签负对照。
2. R-M2 类别原型分类器：把外部病例压缩为可训练类别原型；这是建筑卡级模型接入筛选，不宣称已提高像素 mIoU。
3. R-M3 角色 Token 支持交互：检索病例只提供 core/ring/change/stable 特征，不输入其标签，不进行整栋类别广播。
4. R-M4 困难病例路由：Memory 一致性和分歧只预测 P3 是否可能错误，再调用参数化专家。
5. R-M5 稀有类训练课程：比较随机采样与事件—类别均衡、困难负样本训练；推理期不需要 Memory。

五阶段即使科学判定为 STOP 仍连续执行，最终 `memory_roles_summary.json` 只有在 R-M1 前置条件和对应阶段同时通过时才允许进入 P3 像素级模型集成。开发仅使用既有 `meta_train -> meta_calibration`，不重新拆分数据，不访问 meta_confirm、Val450 或 Test。

正式运行：

```bash
bash experiments/spi_memory_roles_v1/run.sh auto
```
