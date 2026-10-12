# 固定 F/Q 区域头适配：服务器命令

本包只新增 `damageformer_srac_region_head_adapt_v1`，不覆盖此前 bank_v2/v3。目标是验证区域分类路径；不训练 Memory 门控，也不改变现有 Q。服务器仍需保留原 SRAC 实验代码、通过的 E0、原权重和数据。

## 1. 解压、预检、设置本轮路径

将 ZIP 放到以下位置后，整段复制执行。**ZIP 带 `bda_benchmark/experiments/` 前缀，解压到仓库根目录 `.`。**

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
"$PYTHON" -m zipfile -e bda_benchmark/experiments/srac_region_head_adapt_v1_20261011.zip .

export EXP=bda_benchmark/experiments/damageformer_srac_region_head_adapt_v1
bash "$EXP/run.sh" preflight

export E0_PROTOCOL=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/protocol
export REGION_ADAPT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_HEAD_ADAPT_V1/run_$(date +%Y%m%d_%H%M%S)
export REGION_ADAPT_FIT="$REGION_ADAPT_ROOT/fit"
export REGION_ADAPT_SMOKE_OUTPUT="$REGION_ADAPT_ROOT/smoke"
export REGION_ANALYSIS_OUTPUT="$REGION_ADAPT_ROOT/old_region_analysis"
export REGION_ADAPT_PROBE="$REGION_ADAPT_ROOT/embedding_probe"
export REGION_ADAPT_EVAL="$REGION_ADAPT_FIT/final_val"
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC
mkdir -p "$REGION_ADAPT_ROOT"
printf 'export %s=%q\n' PYTHON "$PYTHON" EXP "$EXP" E0_PROTOCOL "$E0_PROTOCOL" REGION_ADAPT_ROOT "$REGION_ADAPT_ROOT" REGION_ADAPT_FIT "$REGION_ADAPT_FIT" REGION_ADAPT_SMOKE_OUTPUT "$REGION_ADAPT_SMOKE_OUTPUT" REGION_ANALYSIS_OUTPUT "$REGION_ANALYSIS_OUTPUT" REGION_ADAPT_PROBE "$REGION_ADAPT_PROBE" REGION_ADAPT_EVAL "$REGION_ADAPT_EVAL" > "$REGION_ADAPT_ROOT/session.env"
echo "本轮路径恢复文件：$REGION_ADAPT_ROOT/session.env"
```

预检只是 CPU 合成测试，不代表已处理 Train/Val。请保存上面打印的 `session.env` 路径；新终端里先 `cd /root/shared-nvme/BRIGHT-master`，再 `source` 这个文件，即可恢复本轮环境变量。已经有本轮目录时不要重新执行含时间戳的初始化去覆盖记忆中的路径。

## 2. 分析之前已经完成的 Val450

如果当前终端没有 `REGION_VAL_OUTPUT`，脚本会在 E0 所属运行目录寻找唯一匹配的完整旧结果。若找到多份，会列出目录；此时把希望分析的那份目录设为 `REGION_VAL_OUTPUT`，再运行。不会按任意最新目录偷偷选择，也不会重跑网络。

```bash
bash "$EXP/run.sh" locate
bash "$EXP/run.sh" analyze
```

读取 `"$REGION_ANALYSIS_OUTPUT/report.md"` 和 JSON，关注真 Damaged 误改的去向、三类非 Damaged 的假阳性净增、事件和像素边界。该分析没有 Q，不能据此生成区域混合率。若旧结果保存在 E0 以外，直接设置 `REGION_VAL_OUTPUT` 为那份已完成目录即可。

## 3. 先做20步 smoke

```bash
bash "$EXP/run.sh" smoke
cat "$REGION_ADAPT_SMOKE_OUTPUT/summary.json"
```

Smoke 检查前向、反向、三个头初始化与保存流程，只用少量Train图像和Val前两图。它没有正式的450张复现资格，其权重不进入正式实验。若任何阶段报错，保留目录和 `summary.json`，不要对同一路径重跑或删除后覆盖。

## 4. 正式 P/R/T 训练

```bash
bash "$EXP/run.sh" train
cat "$REGION_ADAPT_FIT/summary.json"
export REGION_ADAPT_EVAL="$REGION_ADAPT_FIT/final_val"
bash "$EXP/run.sh" report
```

这条 `train` 命令依次执行：初始Val450→三个头各5000次匹配更新→最终Val450。最终评估已经包含在训练中；首次看成绩直接 `report`，不需要再次计算450张。P/R头都从同一A1头初始化；T为现有128维区域token头的辅助诊断。

所有源权重、数据列表、设备、数值协议均从 E0 读取，因此不需要重新填写 `JOINT_RUN`、`A1_CKPT`、`BANK_V2_OUTPUT` 或 Train查询缓存。

## 5. 可选：组件敏感性检查

```bash
bash "$EXP/run.sh" probe
export REGION_ADAPT_EVAL="$REGION_ADAPT_PROBE"
bash "$EXP/run.sh" report
```

这里不训练；在同一前向得到的Q上分别检查均匀汇总、去结构残差、去几何、去统计。结果反映旧模型的依赖，不能直接当作重训消融成绩。完成后要读训练结果，重新设 `REGION_ADAPT_EVAL="$REGION_ADAPT_FIT/final_val"`。

## 6. 可选：在新进程独立复核训练后的头

仅在需要独立复核时执行，不是每次训练后的必需步骤：

```bash
export REGION_ADAPT_EVAL="$REGION_ADAPT_ROOT/replay_val_$(date +%Y%m%d_%H%M%S)"
bash "$EXP/run.sh" evaluate
bash "$EXP/run.sh" report
```

正式报告需完整450张、复现与完整性通过。重点比较P/R、R与原region_mean、T与原region，四类IoU与逐事件变化一起看。单看Damaged救回数、单看训练loss、或Test878上的既有表现，都不能替代这次Val结论。

本地合成测试不等于服务器真实数据训练；新方案收益需要以上Train/Val运行确认。
