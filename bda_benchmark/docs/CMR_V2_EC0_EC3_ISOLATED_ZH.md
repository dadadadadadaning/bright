# CMR-V2：事件因子化反事实 Memory（EC0–EC3）

## 1. 范围

本目录只实现无训练、无融合的 EC0–EC3 审计：

```text
EC0  冻结 Raw-Mean 跨事件 Top-20 基线
EC1  同事件 I/D 反事实专家（仅作为 closed-set upper bound）
EC2  排除查询事件的 I/D 反事实事件专家
EC3  在 EC2 上比较 raw / event-centered / diagonal-whitened 特征
```

不会实现或触发：

```text
EC4 类别内多原型
EC5 软事件混合
EC6 P3 预测修改
```

只有 EC0–EC3 全部满足准入标准后，才允许另开实验实现 EC4。

## 2. 隔离目录

```text
model/cmr_v2/
script/standard_ML/cmr_v2/
tests/cmr_v2/
configs/cmr_v2/
run/run_CMR_V2_EC0_EC3_isolated.sh
```

V2 只读已有 P0 与 M3 cache，不修改旧 CMR、V0、V1 或 P3 文件。

## 3. 事件因子分解

事件中心严格定义为：

```text
mu_event = mean(mu_event_intact, mu_event_damaged, mu_event_destroyed)
```

三个类别权重各为 1/3，不使用事件内样本数加权。缺少任一类别的事件会标记为
`reliable_three_class_center=false`，查询变换回退到全局三类等权中心，不会把
不完整事件中心伪装成可靠估计。

白化使用三类等权的逐维残差尺度，不做高维全协方差求逆。

## 4. 评分协议

每个事件专家分别计算：

```text
Delta_event = score(Damaged supports) - score(Intact supports)
```

跨事件分数是可用事件专家 Delta 的等权平均，不按事件样本数量加权。额外输出：

```text
事件专家数量
Delta 的事件间标准差
支持 Damaged 的专家数量
支持 Damaged 的专家比例
至少 3 个专家且支持比例 >= 70% 的样本数
```

这些一致性指标在 EC0–EC3 中只做诊断，不用于修改预测。

## 5. 运行

```bash
bash run/run_CMR_V2_EC0_EC3_isolated.sh preflight
bash run/run_CMR_V2_EC0_EC3_isolated.sh audit
```

也可一次执行：

```bash
CMR_V2_ENV=/path/to/cmr_v2.env \
  bash run/run_CMR_V2_EC0_EC3_isolated.sh all
```

## 6. 固定超参数

第一轮固定：

```text
case_k       = 5
temperature  = 0.10
scale_floor  = 0.001
```

不得在 Validation 上扫描这些参数。

## 7. 当前 seed3407 真实结果

```text
EC0 frozen cross-event Top-20       AUROC 0.7120  event 0.5687
EC1 same-event raw                 AUROC 0.7504  event 0.7360
EC2 exclude-same-event raw         AUROC 0.7969  event 0.7268
EC3 exclude-same-event centered    AUROC 0.7332  event 0.7424
EC3 exclude-same-event whitened    AUROC 0.7290  event 0.7302
```

事件中心化未达到 AUROC 0.82，且相对 EC2 raw 下降，因此当前决策为：

```text
STOP_BEFORE_EC4_EVENT_FACTORIZATION_INSUFFICIENT
```

该结果意味着：当前等权事件中心并未成功分离事件域与损毁语义，不应继续用它构建
K=3 损毁子原型或接入 P3。
