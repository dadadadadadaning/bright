# 验证记录（2026-10-21，CPU）

制品 SHA-256：`aec1a463e26ce8d9620863801ba44420fe951da91a6add9e222736d9a171197c`。

环境：Python 3.10.22、PyTorch 2.1.2+cpu。没有 CUDA、远端 BRIGHT 数据和实际 checkpoint2100；未声称复现、修复远端故障或完成远端训练。

## 实际执行

从 pyz 内导入被测模块，而非直接导入开发目录，以覆盖打包后的路径和来源记录。

```bash
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -B research/bright_qwatch_recover2100_20261021/test_recover.py -v
```

17 项全部通过，最终一次用时 6.606 秒：

- 原 JointRegionMemoryClassifier + SupportFactory + 原六分支 update_arms + AdamW 的真实 CPU 前向/反向：有/无观察的每臂日志、全部模型与优化器状态指纹、Python/NumPy/Torch RNG 完全相等。
- 注入 E 的 optimizer 后 Q 修改：在 F 前拦截，保留修改前后 CPU 张量，解绑观察钩子。
- 原路由归一化失败：保留原 allclose 操作数与结果，不修改 Q。
- 非有限 Q 元数据可保存；相同 NaN 位模式不误报成内容改变。
- 模拟 CUDA 异常：保留原异常对象；异常后不追加张量快照。
- 合成 checkpoint1000 的非零 AdamW 状态恢复，执行四个真实配对更新：结果、RNG、数据顺序与独立参考一致；原父目录字节及修改时间不变。
- B 反向异常和 E 前向期间 Q 修改：保留前一配对检查点与日志边界，不把部分更新升级为检查点，不进入评估。
- 只接受400个新增配对更新与完整 Q/路由观察；训练错误阻止评估，评估错误阻止 Test，且不重启训练。
- 固定 Train2000 用认证的2000状态，而非恢复起点2100。
- 预检接受给定失败上下文与固定2100摘要，拒绝不同摘要、部分2136状态和不同失败分支；这些上下文用合成元数据验证，远端必须再次校验实际文件及载入状态。
- QWATCH_BASE.json 内的每个继承模块摘要匹配；三个新增打包模块与源码字节一致。

合成恢复运行因 TinyHead 没有路由函数，单独替代了 PassiveRoutes 安装；原真实模型测试覆盖实际路由钩子、原检查代码和反向。该边界没有隐藏。

另已执行：源码 AST 检查、压缩包 CRC 检查、压缩包内全部31个 Python 模块编译、pyz --help。继承的评估/Test模块字节未改动；它们的既有完整验证记录见基础 finish2500 版本的 VALIDATION.md。

## 解释边界

CPU 等价不保证 CUDA 异步执行时序或非确定性轨迹等价。逐阶段 CPU 快照会同步 GPU；观察版成功通过不能证明原来异步运行的问题消失。需要以新远端输出判断是否完成，且保留观察设置声明。
