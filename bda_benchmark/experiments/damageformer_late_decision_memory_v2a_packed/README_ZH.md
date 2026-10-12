# DamageFormer Late Decision Memory V2-A (Packed)

该版本已经按当前真实 Stage1-3000 cache 适配。

## 已确认标签

Building-level cache:

```text
-1 = unknown / impure
 0 = Intact
 1 = Damaged
 2 = Destroyed
```

Pixel confusion matrix:

```text
0 = Background
1 = Intact
2 = Damaged
3 = Destroyed
```

这两套编号不同，代码中已明确分开，不能混用。

## 当前真实 Region 存储

```text
region_offsets [N+1]
region_tokens  [R,128]
region_critical[R]
region_logits  [R,3]
```

第 i 个建筑的 Region：

```python
s = region_offsets[i]
e = region_offsets[i+1]
tokens   = region_tokens[s:e]
critical = region_critical[s:e]
logits   = region_logits[s:e]
```

不需要重新 padding 后保存。

## Global key

使用当前 C1 最稳定的 Region Mean：

```python
global_key = normalize(features[:,0:128])
```

因为：

```text
features [512]
= region_mean[128]
+ region_std[128]
+ top1_critical[128]
+ top2_critical[128]
```

## Local matching

第一版使用：

```text
global cosine Top-50
+ exclude same event
+ symmetric local Region matching
+ critical/logit-based Region weighting
+ final Top-20
```

Region weight:

```text
critical
+ 0.5 * (p_D - p_I)
- 0.25 * entropy(region_logits)
```

这些系数应作为固定初始协议；后续若调参只能用 Train-Cal/Val，不能使用 Test。

## Memory support filtering

默认 Train support：

```text
label >= 0
label_purity >= 0.70
```

I/D/X 全保留，其中 X 用于 Destroyed veto。

## 安装

把整个目录复制到：

```text
/root/data/BRIGHT-master/bda_benchmark/experiments/
damageformer_late_decision_memory_v2a_packed/
```

先运行：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash \
experiments/damageformer_late_decision_memory_v2a_packed/run.sh \
audit
```

该 audit 不访问 Test。

## 下一实验顺序

```text
E2  Specialist + Prototype
E3A Specialist + Global Memory
E3B Specialist + unweighted Local Region Memory
E3C Specialist + critical/logit-weighted Local Memory
E4  + Utility
```

不要直接跳到 Utility。先证明 Real Memory 在固定 coverage 下优于 Specialist-only 与 controls。
