# 独立 Val450 入口本地验证

执行环境：Python 3.10.22，PyTorch 2.1.2+cpu。未连接用户服务器 GPU，未进行真实 BRIGHT 评估。

25 项测试通过：新入口 16 项、已有 Grid 探针 9 项。

- 直接调用原 `_validate_probability`，合法张量及 RNG 不被观察器改变；合法路径不增加统计读取。
- 超出 1 一个 float32 epsilon 的概率仍被原校验器拒绝，捕获具体图像、分支、超界值和原 CPU 张量。
- NaN 统计能够 JSON 序列化，未修改原张量；模拟 CUDA RuntimeError 后观察器不额外读张量或保存张量。
- 原方法和校验器在成功、异常、已有实例覆盖三种情况下恢复；验证图顺序不符会失败。
- 450 行合成报告通过完整原报告/逐图日志校验和新完整性门槛后，只完成独立输出；保留待验收版本及源失败摘要。
- 部分图像数、缺失完整性门槛、缺失校验和、修改图像顺序或修改 NLL（即使重新计算文件 hash）均不能完成报告。
- CPU/smoke 或 Test 标志不能作为正式 Val 结果完成。
- 用真实原评估器执行六个微型 CPU 分类头 × 450 张微型合成图；共 3150 次原目标计算（含 Grid），模型/RNG 恢复由原评估器检查，优化器无变化。禁止 optimizer step 的测试钩子未触发；生成的 CPU 报告仍为 smoke，不能通过正式 Val 完成检查。

测试依赖既有固定 helper 包及其明确的合成 fixture，均未打入运行 pyz。

```bash
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -B -m unittest -v test_evaluate_checkpoint test_grid_probe
```

运行目录为本文件所在目录。`BRIGHT_RESUME_TEST_SOURCE` 下的 helper archive 必须匹配运行工具固定的 trainer SHA256；集成测试使用原评价源码和原目标函数 fixture。
