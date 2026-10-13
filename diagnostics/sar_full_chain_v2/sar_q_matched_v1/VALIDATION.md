# 发布前检查

- 新工具：42 项 CPU 合成测试通过（pipeline18、GPU排队12、预测存档与四组核查12）。排队测试模拟nvidia-smi，不使用真实GPU。
- 原 Q factorial：原 `run.sh preflight` 的31项CPU测试与全部CLI help通过。
- `start_server.sh`：bash语法检查通过。
- 独立审查核对了原训练器/评估器参数、报告字段、末步checkpoint、配对分析与Q诊断路径。
- 云端Python为3.12、Torch2.2.1+cpu；服务器计划使用既有Python3.10、Torch2.2.1+cu121环境。
- 未运行真实 BRIGHT Train/Val/Test；未验证 CUDA 训练速度、显存峰值、最终指标或SAR增益。
- 原仓库工作树保持不变。独立分支只用于发布工具包，不替换main上的项目。
