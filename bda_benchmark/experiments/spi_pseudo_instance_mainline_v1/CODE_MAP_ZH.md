# 当前代码地图

## A. 论文主线：保留并继续使用

| 层级 | 唯一源文件 | 用途 |
|---|---|---|
| P3模型 | `model/DamageFormer_S8_MICB_DSMIL.py` | RGB/SAR编码、定位、损毁像素头、MICB辅助头 |
| LocCC与A6 | `script/standard_ML/spi_region_feedback/core.py` | 预测建筑区域、区域证据、门控、Majority、Bootstrap |
| 无训练结构验证 | `experiments/spi_selective_context_v1/` | S0-S4与局部匹配对照 |
| 几何因果验证 | `script/standard_ML/spi_geometry_causal/` | C0-C3，只允许LocCC/RGID分区变化 |
| 伪监督RGID训练 | `experiments/spi_rgid_semantic_v1/` | 在线P3特征、Center/Boundary训练、来源证明 |
| RGID解码 | `experiments/rgid_s9ic_clean/rgid/` | Center/Boundary Head、Watershed、Gate、Rollback |

## B. 已完成诊断：保留结果，不再作为下一步训练入口

| 代码 | 已回答的问题 | 结论 |
|---|---|---|
| `script/standard_ML/spi_stage0/` | 正确实例归属是否有语义价值 | E4上限明显为正 |
| `script/standard_ML/spi_t1/` | 组件等权损失能否直接提高P3 | 未超过P3，停止该路线 |
| `script/standard_ML/spi_t1/bag_audit.py` | MICB Bag是否具备纠错信号 | 有局部信号，但硬广播有害 |
| `script/standard_ML/spi_region_feedback/audit.py` | LocCC区域一致性是否有效 | Majority约提升0.877 pp |
| `experiments/spi_selective_context_v1/` | 结构上下文是否超过局部平滑 | 是，最高提升1.0696 pp |

## C. 只作历史/Oracle诊断，不得混入主训练

| 目录 | 限制 |
|---|---|
| `rgid_diagnostics_isolated/` | 用于热点、Center、Oracle或可视化诊断 |
| `rgid_gt_center_experiments_isolated/` | 使用真实/Oracle Center，不是可部署训练结果 |
| `script/standard_ML/audit_H9_building_oracle.py` | 官方实例加载和建筑级诊断 |

## D. 已实现但尚未获准进入主模型

`experiments/spi_rgid_semantic_v1/`中的以下文件属于候选模块，不代表已经跑出
论文结果：

- `method.py`：split-on-demand风险评分；
- `heads.py`：高/低/双尺度Center Head；
- `objectives.py`：质量加权区域损失、等变损失与P3锚定；
- `event_protocol.py`：事件不交叉协议。

只有前一层因果门槛通过，才能启用下一层。代码存在不等于实验结论成立。

## E. 为什么本目录不复制核心实现

此前服务器错误正是部分文件更新、部分文件仍旧导致的：调用方传入
`minimum_area`，被调用方却仍是旧签名。若再复制一套P3、LocCC和RGID代码，
同样的问题会扩大。

因此本目录采用：

```text
一个统一运行入口
一个组件清单
一个运行时接口检查
多个唯一源实现
```

而不是：

```text
复制出第二套P3/LocCC/RGID并独立修改
```

这既满足实验隔离，也保持单一事实来源。

