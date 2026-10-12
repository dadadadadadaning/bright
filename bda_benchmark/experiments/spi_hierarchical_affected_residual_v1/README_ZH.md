# Hierarchical Affected-Occurrence Residual V1

该目录严格隔离五部分：

1. `core.py`：等量Affected残差与可达Oracle；
2. `model.py`：建筑内相对RGB/SAR证据、AO Head、B3/B4安全门；
3. `audit_oracle.py`：只验证fixed S2之后的理论空间，不训练模型；
4. `audit_detail.py`：拆解Oracle来源、位置、残差难度和集中度；
5. `build_evidence_cache.py` + `audit_evidence.py`：构建不依赖GT的候选证据，并进行跨事件无参数审计。

AO-E1只允许使用`meta_train`确定证据通道和方向，使用`meta_calibration`做14事件留一审计。它不会训练最终残差头，也不会读取`meta_confirm`、Val450或Test。

## 通俗流程

```text
AO-E0：假设有一个完美开关，最多能修多少？
  ↓ 已证明有约+6pp理论空间
AO-D0：这些机会来自建筑内部还是边界？是否被少数大区域垄断？需要多强的残差？
  ↓ 通过集中度与定位风险检查
AO-E1：不看GT，只看P3概率、RGB结构和SAR纹理，能否找出该打开的区域？
  ↓ 只有跨事件AUC通过
AO-E2：训练一个小型、零初始化的Occurrence Head
```

AO-E1的样本全集由预测产生：位于P3建筑掩码内、fixed S2仍判为Background或Intact的像素。GT只用于事后标记`Corrected/Harmed/Neutral`，绝不会进入输入特征或候选生成。

## 服务器运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_hierarchical_affected_residual_v1/run.sh preflight
bash experiments/spi_hierarchical_affected_residual_v1/run.sh smoke

nohup env NUM_WORKERS=1 BOOTSTRAP_ITERATIONS=10000 SEED=7341 \
bash experiments/spi_hierarchical_affected_residual_v1/run.sh oracle \
> /root/data/BRIGHT-train-logs/SPI_HIERARCHICAL_AO_E0_seed7341.log 2>&1 &
```

查看：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_HIERARCHICAL_AO_E0_seed7341.log
```

AO-E0使用P3建筑掩码和fixed S2概率，只修正满足以下条件的像素：fixed S2仍预测Background/Intact；等量提升Damaged/Destroyed后，原有严重度偏好恰好与GT一致。它不会利用GT改变Damaged/Destroyed边界。

AO-E0已经通过后，运行详细审计：

```bash
nohup env NUM_WORKERS=1 SEED=7341 \
bash experiments/spi_hierarchical_affected_residual_v1/run.sh detail \
> /root/data/BRIGHT-train-logs/SPI_HIERARCHICAL_AO_D0_seed7341.log 2>&1 &
```

查看D0决定：

```bash
grep -E 'AO-D0|Checks:|Decision:|Saved:' \
/root/data/BRIGHT-train-logs/SPI_HIERARCHICAL_AO_D0_seed7341.log
```

只有D0输出`ADMIT_AO_E1_EVIDENCE_AUDIT`，才运行证据缓存和审计：

```bash
nohup env \
NUM_WORKERS=1 \
FEATURE_CHANNELS=32 \
MAXIMUM_CANDIDATES_PER_IMAGE=2048 \
PERMUTATIONS=1000 \
SEED=7341 \
bash experiments/spi_hierarchical_affected_residual_v1/run.sh evidence \
> /root/data/BRIGHT-train-logs/SPI_HIERARCHICAL_AO_E1_seed7341.log 2>&1 &
```

AO-E1准入同时要求：像素AUC、方向平衡AUC、伪实例AUC、事件宏AUC和正确事件数通过，并且结构证据超过P3/fixed S2概率基线至少0.02。若结果为`STOP_AO_E1_NO_DEPLOYABLE_CROSS_EVENT_EVIDENCE`，停止训练AO Head；Oracle很高并不能替代可学习证据。

结果文件：

```text
/root/data/BRIGHT-eval-results/SPI_HIERARCHICAL_AFFECTED_RESIDUAL_V1/ao_d0_detail_seed7341.json
/root/data/BRIGHT-eval-results/SPI_HIERARCHICAL_AFFECTED_RESIDUAL_V1/ao_e1_evidence_seed7341.json
```
