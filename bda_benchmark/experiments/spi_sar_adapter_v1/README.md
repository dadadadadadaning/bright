# SPI SAR Adapter V1

This experiment adds a switchable SAR-only adapter after P3 multi-scale fusion.
It does not edit the parent P3 model or checkpoint.

Variants:

- `none`: exact frozen-parent control;
- `fsa`: frequency-strip pilot;
- `rcm`: rectangular-context pilot;
- `fsa_rcm`: reserved combination, prohibited until both single modules pass.

Required isolation:

1. instantiate and strictly load `DamageFormer_S8_MICB_DSMIL` first;
2. set `p3.eval()` and `p3.requires_grad_(False)`;
3. optimize only `SARFeatureAdapter.parameters()`;
4. keep `mil_residual_scale=0` and do not train MICB/DSMIL, Memory, S2, or a
   Utility Gate in the same pilot;
5. save the adapter state and parent checkpoint SHA separately;
6. after selecting a new parent, regenerate every downstream S2 change audit.

The local FSA/RCM files are transfer references.  This package intentionally
contains dependency-free safety adaptations instead of importing code from an
external Chinese-path module directory.

Server commands:

```bash
cd /root/data/BRIGHT-master/bda_benchmark
bash experiments/spi_sar_adapter_v1/run.sh preflight

nohup env NUM_WORKERS=4 BATCH_SIZE=4 ACCUMULATION_STEPS=4 PILOT_STEPS=500 \
  bash experiments/spi_sar_adapter_v1/run.sh pilot \
  > /root/data/BRIGHT-train-logs/SPI_SAR_ADAPTER_V1_seed3407.log 2>&1 &
```

The `pilot` mode runs FSA and RCM sequentially on one GPU and audits each on
the 291-image internal calibration split.  It never accesses Val450.  Use
`fsa` or `rcm` instead of `pilot` to run only one variant.

The pilot command is already a sequential controller.  To wait for an
existing pilot and run one explicitly selected follow-up program only after
both audit artifacts have been validated:

```bash
nohup env PILOT_STEPS=500 SEED=3407 \
  bash experiments/spi_sar_adapter_v1/watch_then_run.sh PILOT_PID -- \
  bash experiments/spi_s2_evidence_ladder_v1/run.sh status \
  > /root/data/BRIGHT-train-logs/SPI_SAR_AFTER_QUEUE_seed3407.log 2>&1 &
```

Omit everything after `--` to monitor, validate and summarize without
launching another program.  Never pass an RCM command after a running `pilot`,
because `pilot` already runs RCM after the FSA audit.
