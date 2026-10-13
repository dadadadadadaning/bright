# 验证记录

2026-10-13，北京时间UTC+8。

本次新增串行启动器：24项CPU单元/集成测试通过。包括实际原Q流水线的21阶段调度与gate（用合成产物替代GPU子进程）、追加取证gate、失败阻断、真实子进程文件锁跨父进程返回后的存续、Memory PID/启动时间识别。测试中的2000步产物是合成记录，不是GPU训练。

单文件离线入口：7项CPU测试通过。包括完整清单校验、损坏及越界包拒绝、实际解包、无仓库/无CUDA的plan、只读status、extract-only、打包的原Q和Soft命令行计划。

冻结取证组件沿用已验证版本，SHA256为`0ba4321007543c36e93ac7f68ebb151f11e8b8b77a8c03c01a45a38eee79e2d0`，此前16项CPU测试及随机实际Q模型64×64 FP32/CPU-BF16各一次更新通过。本次串行入口不修改其训练数学。

尚未验证：服务器4090上的完整四组2000步、真实数据Val450、Soft八头训练及耗时；原冻结状态变化根因未定位。启动或CPU测试通过不能代替这些结果。

复现编排测试：进入本目录后运行`python -m unittest -q test_chain test_integration`。入口测试另存于仓库`diagnostics/sar_full_entry_tests/test_entry.py`；可通过`SAR_FULL_ENTRY_PATH`指定发布单文件路径。
