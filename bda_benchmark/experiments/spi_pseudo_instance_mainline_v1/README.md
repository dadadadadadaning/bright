# SPI Pseudo-Instance Mainline V1

这个目录是当前“伪建筑结构提升语义分割”研究线的唯一总入口。它不复制
P3、LocCC、Selective Context 或 RGID 的实现，而是用清单、接口检查和统一
运行脚本把它们固定为一套可复现实验栈，避免服务器再次混用不同版本。

## 当前已知结论

| 变体 | Val450 mIoU | 相对P3 | 当前解释 |
|---|---:|---:|---|
| S0 Raw P3 | 60.5020 | 0.0000 pp | 锁定语义基线 |
| S1 LocCC + Hard A6 | 61.5399 | +1.0379 pp | 最强简单基线 |
| S2 LocCC + Soft Context | 61.5716 | +1.0696 pp | 最高点估计；尚未证明优于S1 |
| S3 Uncertain Interior | 61.1444 | +0.6424 pp | 11/14事件改善，但D-F1门槛失败 |
| S4 Local Control | 60.5080 | +0.0060 pp | 排除普通局部平滑解释 |

当前可以声称：预测得到的建筑区域上下文对冻结P3具有结构特异性的语义
纠错价值。当前不能声称：Soft显著优于Hard、RGID优于LocCC、跨事件泛化
已经解决。

## 目录内容

- `run.sh`：统一的服务器入口。
- `verify_stack.py`：检查导入路径、函数签名和P3关键默认值。
- `summarize_results.py`：汇总现有结果，并补算缺少的S2-S1配对Bootstrap。
- `component_manifest.json`：唯一源文件清单和每个组件的研究角色。
- `experiment_contract.json`：已成立、尚未成立的论文结论。
- `CODE_MAP_ZH.md`：主线、历史诊断、Oracle和候选模块的代码边界。
- `MODEL_FLOW_ZH.md`：从P3到底层伪实例反馈的完整流程和公式。
- `SERVER_SYNC_FILES.txt`：同步到服务器时必须保持同版本的文件集合。

## 服务器使用

环境已经激活时，不要再执行一个不存在的`bin/activate`。确认解释器即可：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
which python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

同步`SERVER_SYNC_FILES.txt`中的全部内容后，首先运行：

```bash
bash experiments/spi_pseudo_instance_mainline_v1/run.sh preflight
```

预检将明确拒绝旧版：

```text
connected_components_4(mask)
```

正确版本必须是：

```text
connected_components_4(mask, minimum_area=1)
```

### 选择性上下文性能Smoke

```bash
PROFILE_TIMING=1 \
PROGRESS_EVERY=2 \
SMOKE_IMAGES=10 \
NUM_WORKERS=4 \
bash experiments/spi_pseudo_instance_mainline_v1/run.sh selective-smoke
```

### 重跑完整Selective Context（只有协议变化时才需要）

```bash
PROFILE_TIMING=1 \
NUM_WORKERS=4 \
BOOTSTRAP_ITERATIONS=10000 \
bash experiments/spi_pseudo_instance_mainline_v1/run.sh selective-audit
```

已经完成且P3复现通过的450张结果，不需要仅仅为了性能优化重复运行。

### 从现有JSON补算S2-S1统计比较

```bash
SELECTIVE_RESULT=/root/data/BRIGHT-eval-results/SPI_SELECTIVE_CONTEXT_VAL450/SPI_SELECTIVE_CONTEXT_validation_full_v1_seed3407.json \
SUMMARY_OUTPUT=/root/data/BRIGHT-eval-results/SPI_PSEUDO_INSTANCE_MAINLINE/mainline_summary_seed3407.json \
BOOTSTRAP_ITERATIONS=10000 \
bash experiments/spi_pseudo_instance_mainline_v1/run.sh summarize
```

若已有带RGID的几何结果，可额外提供：

```bash
GEOMETRY_RESULT=/root/data/BRIGHT-eval-results/SPI_GEOMETRY_CAUSAL_VAL450/SPI_GEOMETRY_CAUSAL_validation_full_with_rgid_g3_v1_seed3407.json
```

### RGID因果链

```bash
bash experiments/spi_pseudo_instance_mainline_v1/run.sh geometry-controls

RGID_STEPS=500 \
RGID_BATCH_SIZE=1 \
RGID_ACCUMULATION=4 \
CALIBRATION_SAMPLES=256 \
NUM_WORKERS=4 \
RGID_OUTPUT_DIR=/root/data/BRIGHT-checkpoints/RGID_PSEUDO_ONLY/seed3407 \
bash experiments/spi_pseudo_instance_mainline_v1/run.sh rgid-train

RGID_CKPT=/root/data/BRIGHT-checkpoints/RGID_PSEUDO_ONLY/seed3407/rgid_step_500_calibrated.pth \
RGID_PROVENANCE_JSON=/root/data/BRIGHT-checkpoints/RGID_PSEUDO_ONLY/seed3407/provenance.json \
bash experiments/spi_pseudo_instance_mainline_v1/run.sh rgid-audit
```

RGID只允许改变区域划分，A6门控、P3概率、语义回写规则和Val450必须固定。

## 下一阶段边界

下一阶段不是继续调Watershed，也不是恢复MICB广播。先完成：

1. S2-S1配对Bootstrap；
2. LocCC与RGID在完全相同反馈规则下的几何因果比较；
3. 在训练集内部校准的参数匹配Local Adapter、Region Adapter、Region+Gate三组Pilot；
4. 单seed通过后才做3000步和三seed；
5. 所有结构锁定后才访问原始Test。
