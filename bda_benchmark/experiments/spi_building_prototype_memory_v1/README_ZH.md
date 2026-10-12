# 建筑条件原型 Memory V1

## 这套实验在做什么

它不再把 SLIC 超像素当成 Memory 地址，而是让完整训练集提供少量、可靠的“建筑损毁范例”。

```text
完整 train_set.txt
  -> 冻结 A0 提取 RGB、SAR 和 |RGB-SAR| 特征
  -> 只保留 GT 建筑、A0 也认为是建筑、类别纯度 >= 80% 的 Token
  -> 每张图每个类别求一个原型
  -> 只用 Train 拟合 PCA，将原型固定压缩到 64 维
  -> 建立 Intact / Damaged / Destroyed 训练集原型库

训练或验证图片
  -> A0 输出
  -> 建筑内部不确定 Token 查询原型库
  -> 小 MLP 判断是否应增加或减少 Affected 证据
  -> 同时修改 Damaged 与 Destroyed logit，不破坏二者严重度差值
```

验证集只做前向计算，绝不更新模型或 Memory；代码拒绝任何文件名含 `test` 的列表。

## 消融含义

| 变体 | 含义 | 要回答的问题 |
|---|---|---|
| MM0 | 冻结 A0 | 原始基线是多少？ |
| MM1 | 同样的 64 维投影和小 MLP，但不给 Memory | 提升是否只是多了参数？ |
| MM2 | 查询完整训练 Memory | Memory 是否有信号？ |
| MM4 | 训练时排除与当前样本同事件的 Memory | 不拆训练集时，能否避免记住同一灾害事件？ |
| MM5 | 与 MM4 相同，但故意轮换类别标签 | 正确类别语义是否真的被使用？ |

MM4 仍然使用完整训练集。所谓“排除同事件”只发生在每次检索时：训练样本来自哪个事件，就临时不看该事件的原型；下一批样本会换一套可见原型。没有创建新的永久数据划分。

## 推荐运行顺序

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/spi_building_prototype_memory_v1/run.sh preflight
bash experiments/spi_building_prototype_memory_v1/run.sh build_memory
bash experiments/spi_building_prototype_memory_v1/run.sh smoke mm2_memory
```

先用 500 步检查损失、`gain` 和残差是否正常打开。这个阶段不读取验证集：

```bash
mkdir -p /root/data/BRIGHT-train-logs/SPI_BUILDING_PROTOTYPE_MEMORY_V1
nohup bash experiments/spi_building_prototype_memory_v1/run.sh screen \
  > /root/data/BRIGHT-train-logs/SPI_BUILDING_PROTOTYPE_MEMORY_V1/screen500.log 2>&1 &
```

如果四个变体都能健康训练，再运行 5000 步正式链。只有所有训练完成后，程序才统一读取一次验证集并比较：

```bash
nohup bash experiments/spi_building_prototype_memory_v1/run.sh formal \
  > /root/data/BRIGHT-train-logs/SPI_BUILDING_PROTOTYPE_MEMORY_V1/formal5000.log 2>&1 &
```

断线不会影响 `nohup` 后台进程。重新连接后查看：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_BUILDING_PROTOTYPE_MEMORY_V1/formal5000.log
```

最终报告位于：

```text
/root/data/BRIGHT-eval-results/SPI_BUILDING_PROTOTYPE_MEMORY_V1/
MM0_MM5_probe5000_admission.json
```

## 准入标准

- MM4 相对 A0 提升至少 `+0.30pp`；
- Damaged F1 不下降；
- Destroyed IoU 下降不超过 `0.10pp`；
- 至少 10/14 个事件不下降；
- 事件 Bootstrap 95% CI 下界大于 0；
- MM4 必须优于错误类别 Memory（MM5）。

如果 MM2 不优于 MM1，说明原型检索本身没有价值，应停止 Memory 路线。如果 MM2 有效但 MM4 无效，说明收益可能依赖同事件近邻，未见事件泛化论点不能成立。
