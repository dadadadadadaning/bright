# SRAC Memory: A1 原生输出捕获修复

冻结的 `DamageFormer_S8_MICB_DSMIL.forward` 在当前 SRAC 前端的一次前向中只调用一次 `base.clf_2`。先前的评估补丁错误地要求两次，因此在第一张 Val 图进入新库对照前报出 `Expected exactly two native A1 classifier calls, got 1`。本补丁只更正三个新评估入口及其合成测试中的调用次数约束；它不修改原 E0 代码、模型权重、Q、库、检索或 Reader，也不放宽原五臂的 `1e-5` 正式复现门槛。

补丁 ZIP 的条目从 `bda_benchmark/experiments/` 开始，**从仓库根目录解压到 `.`**。如将 ZIP 放在服务器的 `bda_benchmark/experiments/`，运行：

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
"$PYTHON" -m zipfile -e \
  bda_benchmark/experiments/srac_memory_native_a1_onecall_fix_20261011.zip .
```

本补丁应在 v2 原包、v2 replay 修复包以及 no-gate v3 包之后安装。之后不要重新解压这些旧包，否则会覆盖本补丁。

本地相关 46 项 CPU 合成测试及六个修改文件的语法检查已通过；服务器仍需运行下面的预检和正式 Val450。旧质量库的建库源码哈希只覆盖 `bank.py` 与 `build.py`，因此更正评估代码后可以复用已建库。

先运行相关预检；它们仅使用 CPU 合成数据：

```bash
bash bda_benchmark/experiments/damageformer_srac_memory_bank_v2/run.sh preflight
bash bda_benchmark/experiments/damageformer_srac_memory_bank_diverse_v1/run.sh preflight
bash bda_benchmark/experiments/damageformer_srac_memory_metric_eval_v2/run.sh preflight
```

已完成的 3411 条质量库位于以下目录，可以直接复用，**无需重新运行 Train2918 建库**。上次失败的 `val_...` 目录要保留，重新评估必须使用新目录：

```bash
export E0_PROTOCOL=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/protocol
export BANK_V2_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_BANK_V2/bank_20261011_081312
export BANK_V2_VAL_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_BANK_V2/val_$(date +%Y%m%d_%H%M%S)
unset BANK_V2_LIMIT MATMUL_TF32 CUDNN_TF32 DETERMINISTIC
bash bda_benchmark/experiments/damageformer_srac_memory_bank_v2/run.sh evaluate
cat "$BANK_V2_VAL_OUTPUT/summary.json"
```

若仍失败，保留新目录的 `summary.json` 和 traceback。只有 `complete`、`full_validation_evidence`、`reproduction_passed` 全为 `true`，才解释 Memory 四格实验成绩。随后再运行 no-gate v3 的独立建库和检索投影实验。
