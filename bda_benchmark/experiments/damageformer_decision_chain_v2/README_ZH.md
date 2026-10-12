# DamageFormer Decision Chain V2

本包只回答三个连续问题，不访问 Test：

1. C0-C3：Region mean/std/critical Token 是否在 A1 状态之外提供增量价值；
2. Specialist + Prototype：跨事件历史原型是否提供当前证据之外的增量价值；
3. 5D Utility：是否能把当前/历史证据关系转成固定 2% 覆盖率下的真实像素收益。

关键修正：

- checkpoint 只由 Train-Cal 的 2% action utility 选择；
- Val 从不参与训练、标准化、checkpoint 或阈值选择；
- Prototype 使用 Train-Fit、事件均衡、查询同事件排除；
- Utility 只使用 `[zC,zH,min(zC,zH),abs(zC-zH),logArea]`；
- Utility 采用 leave-one-event-out OOF 选择模型；
- Global/Local/Weighted Memory 不进入本包。

连续运行：

```bash
bash experiments/damageformer_decision_chain_v2/run.sh all
```

最终结果：

```text
/root/data/BRIGHT-eval-results/DAMAGEFORMER_DECISION_CHAIN_V2/decision_chain_summary.json
```

