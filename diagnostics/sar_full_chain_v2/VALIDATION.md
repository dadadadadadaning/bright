# V2 验证记录

2026-10-13。V2仅调整完整SAR流程的资源调度，不改变模型、冻结检查、来源证书和原训练预算。

编排与取消队列：55项CPU单元/集成测试通过。其中25项覆盖原Q的21阶段调度、原gate与追加冻结取证gate、失败阻断、后台文件锁及环境绑定；10项覆盖Q/Soft各自的共享GPU策略、真实busy字段、来源文件绑定和默认等待行为；20项覆盖安全取消，包括Linux pidfd真实自建排队进程退出、PID复用拒绝、暂停后转入训练时恢复而不终止、活动子进程拒绝及有界超时。

单文件离线入口：10项CPU测试通过。覆盖嵌入清单校验、损坏/越界包拒绝、实际解包、plan与--run-now计划、只读status、空队列取消、参数互斥及原Q/Soft命令行计划。合计65项测试，真实进程测试在当前Linux环境执行，未跳过。

运行测试：进入本目录执行`python -m unittest -q test_chain test_integration test_concurrent_policy test_queue_control`。入口测试位于仓库`diagnostics/sar_full_entry_v2_tests/test_entry.py`；可用环境变量`SAR_FULL_ENTRY_PATH`指定单文件路径。

Q/Soft原工具和SAR_FREEZE_WITNESS_V1.py与V1逐文件一致。冻结取证脚本SHA256为`0ba4321007543c36e93ac7f68ebb151f11e8b8b77a8c03c01a45a38eee79e2d0`。

测试中的训练产物为合成fixture，没有在此云工作区进行BRIGHT训练。尚未验证4090真实并发显存、训练速度、四组2000步、Soft八头及Val450；原冻结状态变化根因仍未定位。CPU测试通过不能代替正式实验结果。
