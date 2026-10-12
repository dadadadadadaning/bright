# T128 历史重放只读诊断

适用情况：已安装 `damageformer_srac_memory_key_compare_v1`，CPU 预检通过，四图 `build_smoke` 在 `Original T128 supports differ on replay` 处失败且数值有限。

关闭终端会丢失临时环境变量，不会删除实验文件。本工具默认从 `/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_KEY_COMPARE_V1/*/keys_smoke/summary.json` 找出修改时间最近的 T128 失败记录并打印实际路径；可用 `--failed-build` 指定具体的 `keys_smoke` 目录。核对失败记录的 E0 SHA、工具源代码 SHA 后才运行。默认 E0 是已经通过的 `SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json`。

## 运行

将本文件夹上传或解压到服务器 `/root/shared-nvme`；无需安装器、不改原实验目录、不依赖旧终端变量。不要与其他 GPU 实验同时运行。

```bash
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export DEBUG_TOOL=/root/shared-nvme/memory_replay_debug
"$PYTHON" -m unittest discover -s "$DEBUG_TOOL" -p 'test_*.py' -q
"$PYTHON" "$DEBUG_TOOL/run_debug.py" --plan-only
"$PYTHON" -u "$DEBUG_TOOL/run_debug.py"
```

工具路径不同只需改 `DEBUG_TOOL`。首次命令是 CPU 合成测试，不代表真实 CUDA、Train、Val 已通过。每次真实诊断创建独立新结果目录；旧案例库、旧失败记录、原模型和源码保持不变。

## 做什么

1. 原封不动调用已有建库程序 **一次**，仅四图 smoke。失败前不加追踪、模型钩子、GPU 分配或运行时覆盖；原程序仍自行执行原来的所有门槛。
2. 如果仍发生有限的 T128 失配，从异常 traceback 的局部变量取得原始 FP32 T128、旧 FP16 支持行和新程序已转换的值，保存 `failure_scene.pt`。原程序退出时已恢复它自己的池化和钩子。
3. 检查旧银行支持行与对应 SHA 绑定的原始 Train shard、region ID、标签、几何逐值一致。比较同一原始 FP32 值的 CPU 与 GPU FP16 转换，报告最多20个异常坐标、银行行、区域 ID、通道、相邻 FP16 值及舍入中点。
4. 对固定的失败输入，按固定顺序执行两次 `no_grad` 与一次 `inference_mode` 前向。报告 F、Q、建筑定位输出 `loc`、区域输入 `adapted_content`、mass、geometry、原始／存储 T128 和软标签的差异。它们均是当前前端的局部只读诊断；没有重跑旧2918图的历史生成前缀。
5. 核对权重、代码、数据文件、E0运行时及原失败文件不变。任何其他错误或 CUDA 错误立即终止，不在损坏的 CUDA 上下文中重试。

## 结果怎么理解

- `diagnostic_complete=true`：预定观察结束。**不表示历史重放或修复成功**。
- `original_attempt.passed=false`：本次独立 smoke 仍失败，原门槛不变。
- `phase=history_not_reproduced`：这一次 smoke 通过，但旧失败仍未解释，工具不会继续训练。
- `old_bank_matches_original_shard`：排查支持行映射或文件损坏。
- `baseline.cpu_vs_builder_gpu_cast_exact`：检查同一现场 CPU 与 GPU 转 FP16 是否不同。
- `baseline.outliers[].fp16`：只描述当前值所在的量化区间。旧 FP32 值没保存，不能由旧 FP16 反推出历史精确值。
- `builder_stored_outliers.outliers`：专门列出原程序 GPU 转 FP16 后的实际失配坐标，避免正常存储量化误差遮住这几个失败分量。
- `repeats[].raw_to_initial_failure` 与 `cast_to_initial_failure`：区分当前 FP32 重复计算的小差异和量化后的跳变；单次 `inference_mode` 结果不证明该模式稳定或已修复。
- `soft_label_replay`：补做先前因 T128 抛错而尚未执行的标签一致性观察。
- `historical_runtime_settings_recorded=false`：旧缓存缺少历史计算配置，不代表当前 E0 的配置没有通过。

所有张量检查仍用原 `atol=rtol=2e-5`。raw FP32 与旧 FP16 的对照还包含正常存储量化误差，不能作为新的正式门槛。没有扩大容差、替换旧银行、GT参与检索、训练、Test或完整Val成绩；`downstream_authorized` 与 `historical_failure_resolved` 始终为 false。

请返回实际输出目录内的 `summary.json`，不要将张量 `*.pt` 当成新的可训练银行。若原 smoke 在本次恰好通过，`exact_build` 内可能有新 smoke 产物；本工具不消费或批准它。
