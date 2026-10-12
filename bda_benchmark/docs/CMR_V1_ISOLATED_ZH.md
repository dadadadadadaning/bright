# CMR-V1 隔离式集成说明

## 1. 目标

CMR-V1 直接接入原 BRIGHT/CMR 工程，但不覆盖、不改写任何旧实验文件。

V1 只读取已有上游产物：

- Frozen P3 checkpoint
- CMR-P0 train/val pooling cache
- CMR-M3 train/val Raw-Mean Top-20 candidate cache

V1 自己生成的 token cache、audit JSON、matcher 结果全部写入独立结果目录 `CMR_V1_ISOLATED`。

## 2. 新增目录

```text
BRIGHT-master/
├─ model/
│  └─ cmr_v1/
│     ├─ __init__.py
│     ├─ set_matcher.py
│     ├─ replay.py
│     └─ region_cache.py
│
├─ script/standard_ML/
│  └─ cmr_v1/
│     ├─ __init__.py
│     ├─ audit_top20_complement.py
│     ├─ build_region_cache.py
│     ├─ build_token_cache.py
│     ├─ audit_set_verifier.py
│     └─ summarize_matchers.py
│
├─ tests/
│  └─ cmr_v1/
│     ├─ __init__.py
│     └─ test_set_matcher.py
│
├─ configs/
│  └─ cmr_v1/
│     └─ paths.example.env
│
├─ run/
│  └─ run_CMR_V1_isolated.sh
│
└─ docs/
   └─ CMR_V1_ISOLATED_ZH.md
```

## 3. 不修改的旧文件

V1 **不会修改**下列原文件：

```text
model/cmr/*
script/standard_ML/build_CMR_pooling_cache.py
script/standard_ML/audit_CMR_M3_oracle.py
script/standard_ML/cmr_pairwise_utils.py
script/standard_ML/cmr_metric_utils.py
script/standard_ML/build_CMR_V0_token_cache.py
script/standard_ML/audit_CMR_V0_local_verifier.py
run/run_CMR_M3_pairwise.sh
run/run_CMR_V0_local_verifier.sh
```

V1 允许 `import` 旧 runtime/helper，但只读调用，不改其实现。

## 4. 当前 Memory / Top-20 的上游关系

```text
Frozen P3
   ↓
CMR-P0 pooling cache
   ↓
class-event balanced train reference bank
   ↓
Raw Mean key = P3 fused_roi 在预测建筑区域内的平均特征
   ↓
排除同 event
   ↓
cosine Top-20
   ↓
CMR-M3 candidate cache
   ↓
CMR-V1（只读上述结果）
```

所以 V1 第一阶段不重新建立 Memory，也不重新召回 Top-20。

## 5. Windows 工程目录

你的工程目录：

```text
D:\software\dowmload\BRIGHT-master
```

将本补丁中的新增目录按相对路径复制进去即可。因为所有文件都是新路径，不会覆盖旧 CMR 文件。

如果实际训练在 Linux/WSL/服务器上，请在 Linux 项目根目录执行 shell 脚本；Windows 本地目录仅作为代码仓库时可正常保存这些文件。

## 6. 推荐路径配置

复制：

```text
configs/cmr_v1/paths.example.env
```

为你自己的配置文件，例如：

```text
/root/configs/cmr_v1_server.env
```

修改里面：

```bash
DATASET_ROOT=...
P3_CKPT=...
P0_ROOT=...
M3_ROOT=...
RESULT_ROOT=...
```

然后运行：

```bash
CMR_V1_ENV=/root/configs/cmr_v1_server.env \
  bash run/run_CMR_V1_isolated.sh audit_complement
```

## 7. 第一阶段：只做 Top-20 × P3 互补性

```bash
bash run/run_CMR_V1_isolated.sh preflight
bash run/run_CMR_V1_isolated.sh audit_complement
```

关键输出：

```text
P3-missed Damaged Recall@1/3/5/10/20
first correct rank
unused_headroom_at_maxk
完整 3×3 GT→P3 转移的 Recall@1/3/5/10/20
P3 正确样本的 Harm Exposure@K
候选类别计数、相似度 margin 与跨事件集中度
```

这一步不重新跑 P3，不重建 Memory，不生成 local token。

## 8. 第二阶段：冻结预测 Region Cache

```bash
bash run/run_CMR_V1_isolated.sh build_regions
```

Region Cache 只保存通过严格身份校验的 **P3 predicted component mask**，采用
bbox 内 bit-pack 压缩；GT mask 从不写入。无法唯一确认的行会进入
`unresolved_rows`，并受 `MAX_UNRESOLVED_FRACTION` 上限约束，不会通过降低阈值
或替换成邻近建筑来完成构建。

`verified replay fraction` 是运行环境重放稳定性的总体诊断，不是单实例身份阈值。
默认超过 2% 写出预警，超过 5% 才硬停止；无论总体比例是多少，每个 fallback
仍必须满足 bbox IoU、面积比、RawMean cosine 和唯一性 margin 的原始严格阈值。
诊断写入同目录的 `*.identity_diagnostics.json`。

从此后的 token 构建开始，不再运行 connected-components 或 component replay。

## 9. 第三阶段：双支持集 token cache

仅当 Recall@20 有足够 headroom 再运行：

```bash
bash run/run_CMR_V1_isolated.sh build
```

V1 的 proposal 条件是：

```text
P3 predicts Intact
AND
Raw-Mean Top-20 contains at least one Damaged case
```

与 V0 最大区别：**不要求 Top1=Damaged**。

同时构造：

```text
Intact support tokens
Damaged support tokens
```

## 10. 第四阶段：Top-20 set-to-set verifier

```bash
METHOD=chamfer bash run/run_CMR_V1_isolated.sh audit_set
METHOD=dn4     bash run/run_CMR_V1_isolated.sh audit_set
METHOD=emd     bash run/run_CMR_V1_isolated.sh audit_set
METHOD=frn     bash run/run_CMR_V1_isolated.sh audit_set
```

一次运行并汇总全部 matcher：

```bash
bash run/run_CMR_V1_isolated.sh audit_all
```

汇总文件：

```text
CMR_V1_ISO_B_matcher_comparison_seed3407.json
```

核心比较：

```text
S_Damaged - S_Intact
```

而不是 V0 的“Damaged support 自己像不像”。

## 11. 隔离原则

- 旧 CMR 结果目录不写入。
- 旧 cache 不覆盖。
- 旧 Python 文件不修改。
- V1 protocol 名称使用 `CMR_V1_ISOLATED_*`，避免误读旧 cache。
- V1 失败时，删除 `model/cmr_v1/`、`script/standard_ML/cmr_v1/`、`tests/cmr_v1/`、`configs/cmr_v1/` 和独立 run 文件即可完全回滚。
