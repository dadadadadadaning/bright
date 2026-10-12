# Cross-Event Metric Memory V2

## 目标

该实验逐项排除 PI-HSM v1 失败的五种原因，不修改P3、不重新生成图像缓存，也不访问meta_confirm、Val450或Test。

数据固定复用：

```text
SPI_CLUSTER_DETAIL_D1_V1/seed7341/meta_train.pt
SPI_CLUSTER_DETAIL_D1_V1/seed7341/meta_calibration.pt
```

## D0：标签、覆盖与裸检索

输出每类数量、事件数、图像数、标签纯度和前景占比，并比较：

```text
允许同事件检索
严格跨事件检索
严格跨事件 + purity>=0.8
严格跨事件 + 打乱标签
```

这一步没有Attention。如果裸检索失败，不能把原因归咎于融合网络。

## M1：跨事件Metric Encoder

三个模型使用完全相同的输入、结构、步数和随机种子策略，只改变损失：

```text
m1_ce                         软标签语义CE
m2_cross_event_contrast       CE + 同类跨事件监督对比
m3_contrast_event_prototype   CE + 对比 + 同类事件原型对齐
```

正对必须类别相同且事件不同。训练采用事件/类别均衡采样，低纯度条目自动降权。

## M2：Memory层级与置信门

只有M1通过才运行：

```text
B0 无Memory参数分类器
B1 只替换Occurrence
B2 只替换Severity
B3 Occurrence+Severity
B4 B1 + 查询置信门
B5 B2 + 查询置信门
B6 B3 + 查询置信门
```

置信门显式使用类别相似度margin、检索熵和Top-K可用性；不在calibration上拟合阈值。

## M1R：M1失败后的选择性救援

当M1得到 `STOP_CROSS_EVENT_METRIC_M1` 时，不允许强行运行M2。M1R只回答：

```text
整体Memory虽然不如参数分类器，是否存在一个可识别的小范围可信子集？
```

默认始终使用参数分类器。只有Memory与参数头意见不同，并且Memory的置信优势、
类别margin和Top-K可用性同时达标时才替换预测。阈值只在meta_train的严格跨事件
OOF检索上选择，meta_calibration只评估一次。

## 服务器同步与检查

同步 `SERVER_SYNC_FILES.txt` 中全部文件，然后：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_cross_event_metric_memory_v2/run.sh preflight
```

预期：

```text
6 passed
Cross-Event Metric Memory V2 preflight passed
```

## 推荐逐步运行

### 第一步：D0

```bash
env \
SEED=7341 \
D1_CACHE_ROOT=/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341 \
OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/SPI_CROSS_EVENT_METRIC_MEMORY_V2/seed7341 \
RESULT_ROOT=/root/data/BRIGHT-eval-results/SPI_CROSS_EVENT_METRIC_MEMORY_V2 \
bash experiments/spi_cross_event_metric_memory_v2/run.sh d0
```

结果：

```text
/root/data/BRIGHT-eval-results/SPI_CROSS_EVENT_METRIC_MEMORY_V2/d0_label_retrieval_seed7341.json
```

### 第二步：M1

只有D0输出 `ADMIT_CROSS_EVENT_METRIC_M1` 才运行：

```bash
nohup env \
SEED=7341 \
D1_CACHE_ROOT=/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341 \
OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/SPI_CROSS_EVENT_METRIC_MEMORY_V2/seed7341 \
RESULT_ROOT=/root/data/BRIGHT-eval-results/SPI_CROSS_EVENT_METRIC_MEMORY_V2 \
METRIC_HIDDEN=128 \
METRIC_STEPS=1000 \
BATCH_SIZE=512 \
MEMORY_CAPACITY_PER_CLASS=2048 \
MEMORY_TOPK=4 \
CONTRAST_TEMPERATURE=0.10 \
RETRIEVAL_TEMPERATURE=0.10 \
MINIMUM_MEMORY_PURITY=0.0 \
bash experiments/spi_cross_event_metric_memory_v2/run.sh m1 \
> /root/data/BRIGHT-train-logs/SPI_CROSS_EVENT_METRIC_MEMORY_M1_seed7341.log 2>&1 &
```

不要看完D0后临时调整 `MINIMUM_MEMORY_PURITY`。默认0.0表示软标签训练保留全部条目；D0中的pure80只是预注册诊断对照。

### 第三步：M2

只有M1输出 `ADMIT_HIERARCHICAL_MEMORY_M2` 才运行：

```bash
env \
SEED=7341 \
D1_CACHE_ROOT=/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341 \
OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/SPI_CROSS_EVENT_METRIC_MEMORY_V2/seed7341 \
RESULT_ROOT=/root/data/BRIGHT-eval-results/SPI_CROSS_EVENT_METRIC_MEMORY_V2 \
MEMORY_TOPK=4 \
RETRIEVAL_TEMPERATURE=0.10 \
bash experiments/spi_cross_event_metric_memory_v2/run.sh m2
```

### M1失败时：只运行一次M1R

```bash
env \
SEED=7341 \
D1_CACHE_ROOT=/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341 \
OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/SPI_CROSS_EVENT_METRIC_MEMORY_V2/seed7341 \
RESULT_ROOT=/root/data/BRIGHT-eval-results/SPI_CROSS_EVENT_METRIC_MEMORY_V2 \
MEMORY_TOPK=4 \
RETRIEVAL_TEMPERATURE=0.10 \
MINIMUM_MEMORY_PURITY=0.0 \
bash experiments/spi_cross_event_metric_memory_v2/run.sh rescue
```

结果：

```text
/root/data/BRIGHT-eval-results/SPI_CROSS_EVENT_METRIC_MEMORY_V2/m1r_selective_rescue_seed7341.json
```

## 如何读取最终结论

```text
STOP_MEMORY_LABEL_OR_COVERAGE_FATAL
```

标签或事件覆盖存在致命问题，停止训练Metric Encoder。

```text
STOP_CROSS_EVENT_METRIC_M1
```

显式跨事件训练仍不能建立可靠语义距离，停止当前Memory路线。

```text
STOP_HIERARCHICAL_MEMORY_M2
```

语义Key有一定改善，但Occurrence/Severity Memory没有带来稳定、类别安全的额外收益。

```text
ADMIT_MEMORY_TO_P5_INTEGRATION_E1
```

只有该结论允许编写P5集成。集成时仍需保持零初始化、严格恒等和完整mIoU重建。

```text
STOP_MEMORY_NO_IDENTIFIABLE_SAFE_SUBSET
STOP_MEMORY_AFTER_SELECTIVE_RESCUE
```

说明当前Memory连一个稳定的高置信子集也找不到，应停止推理期Memory；不再调Top-K。

```text
ADMIT_SELECTIVE_MEMORY_AS_AUXILIARY_E1
```

只允许把Memory作为参数分类器之后的小范围辅助残差，不能替代主分类器。
