# Fixed-front modulation controls

This directory is isolated. It imports, but does not modify, existing V2 and evidence code. It is NOT an SRAC reproduction and does NOT enable Memory or end-to-end updates.

## First experiment (implemented)

Six identical, independently trained fresh MLPs share batches and initial weights:

| Source | Meaning | Heads |
|---|---|---|
| raw | first original-image fusion output, without pixel Soft ROI | pixel / region |
| normal | second modulated-image fusion output, without Soft ROI | pixel / region |
| roi | exact input of existing pixel classifier, with Soft ROI | pixel / region |

One frozen model forward supplies all features and the SAME Q. Region heads use Q-weighted mean pooling, not the old attention/statistics embedding. Original pixel classifier replay is checked on every batch. Both original outputs are reported for reference. All frozen modules stay in eval mode; no BN updates, no Test, no feature cache, no full-model checkpoint copies. Each head saves only best and last weights, not optimizer state (no resume support). A small CPU test is not a real-data CUDA smoke test.

Commands after uploading the ENTIRE directory:

```bash
cd /root/shared-nvme/BRIGHT-master
export PATH="/root/shared-nvme/conda-envs/bright4090/bin:$PATH"
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth
test -s "$FRONT_CKPT" || exit 1
bash bda_benchmark/experiments/damageformer_modulation_controls_v1/run.sh preflight
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/MODULATION_CONTROLS_V1/smoke_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_modulation_controls_v1/run.sh smoke
```

After smoke passes and GPU resources are available:

```bash
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/MODULATION_CONTROLS_V1/formal_$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/MODULATION_CONTROLS_V1
LOG="/root/shared-nvme/BRIGHT-train-logs/MODULATION_CONTROLS_V1/$(basename "$OUTPUT_ROOT").log"
nohup bash bda_benchmark/experiments/damageformer_modulation_controls_v1/run.sh train > "$LOG" 2>&1 &
echo "PID=$! OUTPUT=$OUTPUT_ROOT LOG=$LOG"
```

Defaults: 5000 updates PER head, shared 80000 crops, EBS16=batch2*accum8, LR1e-4 AdamW wd.005, warmup100 + cosine, NLL+Lovasz, Val450 every500. Six heads do not multiply data exposure per head; they do add runtime. Single GPU/process only. Existing source-hash gate is preserved: mismatch must be investigated, never bypassed. Snapshot step3000 measures that checkpoint only, not the final state of an ongoing run.

## Interpretation

- raw__pixel minus normal__pixel: input modulation's effect on learnability with matched heads on frozen features.
- raw__region minus normal__region: same test with a fixed regional readout.
- source__region minus source__pixel: effect of mean aggregation, conditional on source.
- roi__pixel minus raw__pixel: contribution of existing Soft ROI to this fresh probe.
- roi__pixel versus old_pixel: still differs in head architecture, training history and upsampling order; NOT pure capacity proof.
- Same-step comparisons are primary; also report best and last. Repeat paired seeds after a signal. No gain is guaranteed.

## Later steps (NOT enabled here)

`core.BoundedFeatureModulation` is a tested FiLM-inspired prototype: bounded spatial feature scale/shift, predicted condition detached, identity initialization, no image painting. It is NOT wired into the training runner. It needs no-modulation and equal-capacity content-independent controls, then full-model training. Identity at initialization does not prevent later harm.

For subsequent region-context or retrieval training use the research plan in notes/insights/20261003-damageformer-evidence-first-roadmap.md. Those new full-model ablations are proposals, not implemented commands.
