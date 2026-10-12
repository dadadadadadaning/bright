# Candidate Cut-Band Evidence E0 / Verifier E1

## 研究问题

当前 Selective Strip-RGID 的风险分数主要是整栋 LocCC 汇总，无法描述候选切割发生的位置。本实验保持 P3、建筑支持、Strip-RGID 解码器和 fixed S2 不变，只在 RGID 候选子区之间的内部切割带上提取证据：

- RGB Scharr 边缘：灾前几何是否支持切割；
- SAR robust Scharr：灾后结构是否支持切割；
- 同位边缘对照：证明增益不是普通像素对齐；
- RGB→SAR 5×5 局部匹配：允许配准残差；
- 空间打乱对照：排除全局事件/纹理偏差。

E0 只做无参数 AUROC 审计。只有 `ADMIT_CUT_BAND_VERIFIER_E1` 才允许运行低容量 E1。E1 不修改 P3 logits，只判断某个 LocCC 应保留原分区还是采用 RGID 子分区。

## 插入位置

```text
P3 frozen -> LocCC -> Strip-RGID candidate children
                            |
             internal candidate cut band
             RGB edge + SAR local matching
                            |
              E0 audit / E1 small verifier
                            |
              keep LocCC or use RGID split
                            |
                      fixed S2 unchanged
```

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
bash experiments/spi_cut_band_evidence_v1/run.sh preflight
bash experiments/spi_cut_band_evidence_v1/run.sh smoke

nohup env NUM_WORKERS=1 BAND_RADIUS=2 MATCHING_MAX_SIDE=64 MATCHING_CHANNELS=32 MATCHING_WINDOW=5 SEED=7341 \
  PREREQUISITE_JSON=/root/data/BRIGHT-eval-results/SPI_SELECTIVE_STRIP_RGID_E0_V1_FAST2/selective_rgid_e0_seed7341.json \
  bash experiments/spi_cut_band_evidence_v1/run.sh audit \
  > /root/data/BRIGHT-train-logs/SPI_CUT_BAND_E0_seed7341.log 2>&1 &
```

仅当日志决策为 `ADMIT_CUT_BAND_VERIFIER_E1`：

```bash
nohup env SEED=7341 VERIFIER_STEPS=1000 VERIFIER_HIDDEN=32 \
  bash experiments/spi_cut_band_evidence_v1/run.sh train \
  > /root/data/BRIGHT-train-logs/SPI_CUT_BAND_E1_seed7341.log 2>&1 &
```

`run.sh all` 会在 E0 失败时由训练脚本主动拒绝 E1，不会绕过准入规则。

## E0 准入

- local matching AUROC >= 0.60；
- event-macro AUROC >= 0.58；
- 至少 10 个事件方向正确；
- 比 geometry/RGB/same-position 最强对照高至少 0.02；
- 比 spatial shuffle 高至少 0.02；
- permutation p < 0.05。

E0 失败时停止，不训练 E1，也不访问 meta_confirm、Val450 或 test。

完整 audit 还会检查前置 Selective Strip-RGID 结果：仅当 `R4-R0 >= 0.20pp` 且原 R2 selector 失败时运行。若 R4 本身不足，说明候选切割没有理论空间；若 R2 已通过，应冻结当前方法而不是继续增加模块。
