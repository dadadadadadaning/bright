# 验证记录

日期：2026-10-23。环境：本地 CPU，Python 3.10 / Torch 2.1.2。

执行：

```text
/workspace/bright-audit-legacy-env/bin/python -B -m unittest discover -v
Ran 42 tests
OK
```

覆盖：

- 真实发布的 JointRegionMemoryClassifier 和 slots，而非只测试伪造 Reader。
- 四版本共同参数初始化相同；新增参数不推进 RNG。
- 新路径零初始化逐位恢复基线，且系数有有效梯度。
- 正确 logits/probabilities 契约与前景质量约束；包含 argmax 定位可能变化的反例。
- 无有效支持时 Memory 证据为零。
- 新标签路径置乱不改变原视觉 Reader；只能使用当前图已选合法候选的标签供体。
- 禁止查询 GT 进入 forward；非法 route 和巨大索引继续报错。
- 原始损失、优化器检查点往返、首次异常持久化、评估 RNG 和模式恢复。
- 完整 Val450/训练轨迹/指标重算，拒绝不完整结果；量化 Grid 差异。
- 夜间串行顺序、GPU 忙拒绝、失败停止、版本改变停止。

独立代码复核发现并修复了置乱供体跨出合法候选集合的问题。
之后复核未发现阻断问题。

尚未执行：服务器真实 GPU smoke、正式训练、Val450、Test878、多种子性能评估。
测试通过不证明原服务器的偶发张量损坏已经解决，也不证明新设计提高指标。
