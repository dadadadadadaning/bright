# Memory 只读审计：使用说明

本脚本只读取明确指定的 Train 支持库和 Val 查询，不训练、不改阈值、不修改原始文件。结果写到一个必须不存在的新目录。带 Test 文件名或 Test 元数据的输入会拒绝；该检查不能识别伪装成 Val 的文件，输入来源仍须人工确认。

## 1. 先运行支持数量和覆盖检查

在已安装 NumPy、PyTorch 的 BRIGHT 环境执行。下面是 Windows PowerShell 命令，输入文件名已在本地目录确认。首次运行建议只做此项，不开检索重放。

```powershell
python D:/software/dowmload/BRIGHT-master/tools/memory_readonly_audit.py `
  --memory D:/software/dowmload/BRIGHT-checkpoints/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2/train_superpixel_memory.pt `
  --query D:/software/dowmload/BRIGHT-checkpoints/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2/val_slic_records.pt `
  --candidates D:/software/dowmload/BRIGHT-checkpoints/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2/train_slic_records.pt `
  --exclude-same-event --topk 8 `
  --output D:/software/dowmload/BRIGHT-memory-audit-exemplar-counts-01
```

K、温度、同事件限制、每事件限额必须与待审计运行的配置一致，示例不是参数调优建议。无 `--max-per-event` 时不设事件限额；同图像默认排除。

若在 Linux 训练服务器运行，把仓库和缓存路径换成服务器实际路径，用反斜杠换行，或写成单行。不要直接复制 PowerShell 的反引号换行。

## 2. 可选：重放近邻投票

复制第一条命令，换一个新的输出目录，并追加：

```text
--replay --metric D:/software/dowmload/BRIGHT-checkpoints/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2/metric_key_step5000.pth --temperature 0.1 --allow-unverified-key-space
```

旧 Exemplar 库没有完整编码器哈希，因此必须显式允许未核验的特征来源，报告会标为 UNVERIFIED。这只是诊断性参考重放，不能证明历史运行完全复现。应人工确认选用的是原建库编码器；不要为绕过明确的哈希不匹配使用此选项。

重放计算三种三类近邻投票：原参考规则、固定邻居的去重重算、完整合法候选检索。逐查询扫描库，可能较慢，时间与库大小和查询数有关。NumPy 并列排序可能与原 PyTorch 顺序不同。

对于 factorized 系列，输入库仍用 `--memory`，查询必须有 `keys` 或 `query_keys`；按历史配置设置 `--max-per-event 2 --topk 8`。参考旧规则只搜索前 `50*K` 候选，邻居不足时重复最后一个；完整合法对照搜索全库且不重复补齐。它不重放检索注意力模型，也不把近邻投票当成该模型输出。只有原始分组特征而无 keys 的缓存，当前脚本不支持自动编码。

## 3. Evidence relation：先查每类合法支持

使用该实验原运行的 Stage1 Train、Stage1 Val、specialist train_scores。下列文件名对应实现中的默认命名，目录请填写实际归档位置：

```text
python tools/memory_readonly_audit.py --mode evidence --train-cache <stage1_train_all_buildings.pt> --query <stage1_val_all_buildings.pt> --fit-scores <specialist_c1/train_scores.pt> --topk 50 --min-purity 0.7 --output <新的输出目录>
```

此模式按 fit_mask、纯度和有效标签恢复支持池，仅分析基础预测为 Intact 的 Val 建筑，自动排除同事件。原缓存的 0/1/2 自动转为审计中的 1/2/3。不需要 `--label-base 0`。

这里检验的是“每类合法支持不足”的触发条件，不是细重排序违规的实际次数。当前脚本不执行局部细重排序，也不接收该模式的邻居 trace；不足支持不能直接等同于已污染最终预测。fit_mask 缺少来源哈希时只按行关联，会明确警告，需核对原归档。

## 4. 有历史邻居索引时，直接检查历史 trace

bank 模式可用 `--neighbors <trace.json或pt>` 代替 `--replay`。trace 必须与查询、支持库逐行对应，结构为：

```json
{
  "metadata": {"query_sha256": "查询文件SHA256", "memory_sha256": "支持库文件SHA256"},
  "indices": [[0, 1, 1]],
  "scores": [[0.9, 0.89, 0.89]],
  "valid_mask": [[true, true, true]]
}
```

valid_mask 表示历史计算真正使用的槽位，不能事后把错误槽位改成 false 来掩盖历史影响。未提供时所有非负有效索引均按活动槽位检查。mask=false 的补齐槽位不参与投票。只有邻居特征、没有索引和分数的缓存，不能冒充历史 trace；改用标记为参考重放的模式。

SHA256 可在 PowerShell 用 `Get-FileHash -Algorithm SHA256 <文件>` 获取，填入小写十六进制字符串。不要更改输入缓存本身。

## 5. Damaged→Intact 子集和标签口径

bank 默认标签为 1=Intact、2=Damaged、3=Destroyed；若原缓存是 0/1/2，显式传 `--label-base 0`。不要凭数值猜测编码。

只有查询中保存了同一行对应的原基础模型预测，才能用：

```text
--base-pred-field <真实字段名> --base-label-base 1
```

例如确有 `p3_region_labels` 时可填写该字段名；0/1/2 预测需设置 `--base-label-base 0`。不得用另一个分类头的预测代替基础预测。旧 Exemplar val_slic_records 缺少该字段，因此其 Damaged→Intact 子集输出为 null，不是零漏检。将来补充字段须另存新缓存并保证行身份对应，不能改写旧缓存。

## 6. 输出解读

| 文件 | 内容 |
| --- | --- |
| manifest.json | 输入路径、SHA256、参数、审计脚本哈希 |
| coverage.csv | 事件×类别的入库数、独立案例数、图像覆盖、最大/前五图占比、候选池对照 |
| queries.csv | 每个查询各类合法支持、事件限额后支持、可核验的重复/违规及投票结果 |
| summary.json | 全体、Damaged、Damaged→Intact、各事件汇总与局限 |

`legal_*` 是支持库行数；存在可靠案例标识时 `unique_legal_*` 才是独立案例数。跨类别检索的每事件限额共享，不能把各类单独限额后的数量直接相加。`global_event_capped_support` 给出共享限额下总数。

同图像的不同区域不是重复案例。优先用 source_indices，否则用事件、图像、region_id/预测建筑ID组合；缺少可靠身份时独立案例重复率为 null。旧 Exemplar 库恰有此限制。

无邻居 trace 且未开重放时，重复与违规汇总为 null，表示没检查，不是零。无支持时返回弃权，不虚构 Memory 类别；指标同时报告有效评价数和弃权数，不同覆盖率的 F1 不能直接比较。混淆矩阵顺序为 Intact、Damaged、Destroyed；这是区域/建筑指标，不是像素 mIoU。

去重投票只隔离重复计票，不同时删除来源违规，避免混合归因；完整合法检索另作对照。最终须查看“重复触发比例、预测改变、Damaged救回、Intact新增误报”，不能仅凭重复存在解释精度下降。

本版未实现：重新选库的多种子覆盖干预、无GT筛选的全量前向提取、注意力/局部细检索固定权重重放、像素回填评估。现有查询的GT筛选偏差仍然保留，报告不能推广成完整验证集结论。

## 7. 测试与安全加载

```text
python -m unittest discover -s tools -p test_memory_readonly_audit.py -v
```

合成测试只需 NumPy；读取真实 pt/pth 需要 PyTorch。本次开发环境没有 PyTorch，未在真实缓存执行，不能把合成测试当作实测审计结果。

默认使用安全的权重加载方式。若旧缓存包含自定义对象导致加载失败，仅对来源可信的本人缓存显式添加 `--trusted-pickle`；不要对下载或未知来源文件使用，它允许反序列化执行代码。脚本从不自动切换为不安全加载。
