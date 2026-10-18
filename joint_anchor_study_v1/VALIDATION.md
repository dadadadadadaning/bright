# 验证记录

日期：2026-10-18。验证对象是代码的数学、状态和流程合约，不是新方法的BRIGHT实验成绩。

- Python3.10 / PyTorch2.1.2 CPU：76项测试通过。
- Python3.12 / PyTorch2.14.1 CPU：相同76项测试通过。
- 26项KL数学/梯度检查：方向、detach、ignore、背景、极端logits、精确定位/等级分解、分类贡献、gradcheck。
- 12项配对引擎检查：实际500次Adam更新后分叉；lambda0时含Dropout的两组参数/动量完全一致；正KL改变学生且不改冻结输入；诊断开关和分支顺序不改变结果。
- 12项评估检查：使用原指标函数；原目标一致；全局/事件/配对计数重读；无Test；模型模式、参数与RNG恢复。
- 16项来源检查：固定依赖哈希、完整缺失清单、fresh显式配置、无恢复伪造。
- 10项驱动检查：检查模式、失败停止、目录保护；合成运行时真实执行fresh2+20更新和500+20恢复；终检失败不产生正式完成标记。
- 另执行一次实际固定版本`JointRegionMemoryClassifier`的CPU集成：复用原fixture和真实双槽模型，两个共同前缀更新后分叉，新增KL前向/反向成功，两个分支Reader梯度均为正，输入保持冻结。此项使用缩小隐藏通道的合成输入，不是BRIGHT数据。
- `start_server.sh`通过bash语法检查。
- 共享依赖安装器已实际下载固定GitHub提交的ZIP、验证SHA并安装；已有正确版本复用逻辑保留。

尚未验证：用户服务器的实际数据/权重是否仍完整，真实BRIGHT CUDA smoke、1500步训练、完整Val450、新方案的准确率收益。

当前没有访问用户的研究服务器。不能由这些CPU结果宣称训练已启动、依赖已恢复或方法有效。

## 本地可重复检查

```bash
BRIGHT_ORIGINAL_TOOL=/workspace/bright-followup/lr_study_source/original_fix2 BRIGHT_LOSS_HELPERS=/workspace/bright-design-delivery/joint_loss_after500_v1 /workspace/bright-audit-legacy-env/bin/python -m unittest discover -s /workspace/bright-design-delivery/joint_anchor_study_v1 -p 'test_*.py' -q
```

legacy CPU环境为本次AdamW测试补齐`packaging==26.3`；未修改用户研究服务器环境。共享库和原fix2源码保持原始校验值。
