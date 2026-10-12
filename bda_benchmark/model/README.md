# BRIGHT 模型目录

本目录现在分为三层：

1. `registry.py`：完整任务模型的唯一逻辑入口，模型名、实现文件、输入形式和输出形式都在这里登记。
2. `distillation/`：教师/学生通用输出、冻结教师、权重加载等蒸馏基础设施。
3. 原有 `DamageFormer_*.py`、`UNet.py` 等实现：暂不大规模移动，以保证旧训练命令和已有权重仍可使用。

## 新代码的模型创建方式

```python
from bda_benchmark.model import create_model

teacher = create_model("damageformer_iarm", offline=True)
student = create_model("damageformer_s1c", offline=True)
```

也兼容文件名、类名和短别名：

```python
create_model("DamageFormer_S1C", offline=True)
create_model("s1c", offline=True)
```

查看全部可用模型：

```bash
python -m bda_benchmark.script.distillation.inspect_pair --list
```

## 蒸馏输出约定

不同旧模型可能返回单个 Tensor、`(loc, damage)`、`(loc, damage, aux)` 或字典。
`distillation.normalize_model_output` 将它们统一为：

- `damage_logits`
- `localization_logits`
- `feature_maps`
- `auxiliary`
- `raw`

蒸馏损失只依赖这个统一结果，不再针对每个实验文件重复写分支。

## 兼容原则

- 旧的 `from model.X import Y` 暂时保留可用。
- 新代码使用 `bda_benchmark.model`。
- 只有完整任务模型进入注册表；几何头、损失、区域操作和检索组件不伪装成完整模型。
- S7 SIRM 是带外部 memory/config 的包装器，创建时必须显式传入依赖。

