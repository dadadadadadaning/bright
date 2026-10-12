# Frozen-Q building-route experiment

This is an isolated **region readout** experiment. It does not train A1, Q,
Memory, J1/J2/J3 or a reliability gate. It compares four independent copies
of the same E0 region embedding/classifier on each identical Train image:

| Arm | Route sent to pooling and probability backprojection |
| --- | --- |
| `unsplit` (S0) | original nine-way soft Q |
| `capacity_control` (C2-like) | 2K slots and 18 supports, split by a fixed spatial checkerboard unrelated to predicted B |
| `hard` (S2-like) | each original seed becomes predicted-background and predicted-building slots; B threshold 0.5 |
| `soft` | same two slots, using `Q(1-B)` and `QB` continuously |

Predicted B comes from the frozen A1 in both Train and Val. No truth is used
to form a route. The same route weights are used for feature pooling and
probability readout. A region can still contain multiple buildings or damage
classes. The current `hard` route tests building/background separation, not
building-instance segmentation.

All readouts copy the exact same initialized E0 embedding and classifier.
Only these private readouts enter separate AdamW optimizers. The source
front/Q and historical heads are immutable. Train uses one shared full-image
Train2918 stream, **1 image per update**, NLL + Lovasz, 200/500 warmup then cosine
to zero (depending on the 2000/5000 budget), learning rate `1e-4`, weight
decay `.005`, clipping `1`. The primary comparison is at the **same final
step**; periodic route-only Val is a diagnostic curve, not formal evidence.
This is matched **among the four route heads** and across the two Q-source
route fits, following the earlier S2 readout protocol. It is not the Q-factorial
training exposure of 16 crops per update; do not compare their step counts as
equal amounts of data.

The **initial and final** Val audits each replay all historical E0 five arms
(`a1_pixel`, `direct_region`, `j1`, `j2`, `j3`) across all 450 images with
the unchanged absolute `1e-5` class/summary metric gate. They also verify
native A1 replay, original weights/buffers, source files, code, runtime, Q
checkpoint if used, full image order, and per-image artifact hashes. Failure
leaves `summary.json: complete=false` and cannot produce a formal checkpoint.
The prior failed `building_reconstruction/last.pth` is deliberately rejected.

## Server commands: first compare S0 / capacity / S2 / soft on original E0 Q

Copy this new package to the identical path in the server repository. The
server must have the exact archived E0 files and original code. Use the
Python interpreter and GPU setup that passed E0; do not set a nonexistent
`CUDA_VISIBLE_DEVICES=1` on a single-GPU machine.

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export E0_PROTOCOL=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/protocol
export ROUTE_EXP=bda_benchmark/experiments/damageformer_srac_building_route_v1
unset CUDA_VISIBLE_DEVICES MATMUL_TF32 CUDNN_TF32 DETERMINISTIC ROUTE_Q_CHECKPOINT
bash "$ROUTE_EXP/run.sh" preflight

export ROUTE_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_BUILDING_ROUTE_V1/original_q_$(date +%Y%m%d_%H%M%S)
export ROUTE_FIT="$ROUTE_ROOT/fit"
export ROUTE_STEPS=2000
bash "$ROUTE_EXP/run.sh" train
export ROUTE_EVAL="$ROUTE_ROOT/independent_val"
bash "$ROUTE_EXP/run.sh" evaluate
bash "$ROUTE_EXP/run.sh" report
```

For a quick diagnostic before a long run, use a **different fresh** output
and `bash "$ROUTE_EXP/run.sh" smoke` (sets two steps and two Val images).
Smoke outputs cannot be used as formal evidence. Every output path must be
new; the scripts never overwrite an older failed report.

Read `fit/summary.json`, `fit/initial_val/result.json`,
`fit/final_val/result.json`, `fit/validation_curve.jsonl` and
`independent_val/result.json`. Proceed only if the fit and independent Val
both show `complete=true`, `full_validation_evidence=true`,
`reproduction.passed=true`, and all `immutable` checks are true. Compute
S2-like gain as `hard - unsplit` in **percentage points**; `hard - capacity_control`
checks whether the improvement exceeds a non-building 2K-slot split; `soft - hard`
isolates the continuous versus threshold route. The spatial control is not a
perfect removal of all building information because the frozen source Q and
adapted features already use predicted B. Also inspect each class IoU,
per-event IoU and false-positive transitions from the saved prediction maps.

To test whether 2000 updates were too short, rerun all four readouts together
with `ROUTE_STEPS=5000` and a **new** `ROUTE_FIT`. Do not compare a 5000-step
hard route against a 2000-step unsplit route. The current loss schedule reaches
zero at the final step, so extending an old checkpoint without restarting a
matched schedule is a different experiment.

## Stage 4: crossed Q and route comparison

Only after `building_reconstruction` Q training has passed its frozen A1,
downstream, runtime and 1e-5 checks, run a second route fit with its completed
`last.pth`. A failed/incomplete Q checkpoint is not accepted.

```bash
export ROUTE_Q_CHECKPOINT=/absolute/path/to/completed_building_reconstruction/last.pth
export ROUTE_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_BUILDING_ROUTE_V1/building_q_$(date +%Y%m%d_%H%M%S)
export ROUTE_FIT="$ROUTE_ROOT/fit"
export ROUTE_STEPS=2000
bash "$ROUTE_EXP/run.sh" train
export ROUTE_EVAL="$ROUTE_ROOT/independent_val"
bash "$ROUTE_EXP/run.sh" evaluate
bash "$ROUTE_EXP/run.sh" report
```

This yields a matched **2×2**: original Q / building-reconstruction Q ×
unsplit / hard route, plus spatial-capacity and soft-route controls in both Q sources. The
readout training budget, source E0 and Val order stay the same. Compare the
two Q sources *within the same route mode*, and hard versus unsplit *within
the same Q source*. If the Q-factorial checkpoint's own source hash changes
after an A1 bug fix, retrain Q; do not bypass the hash gate.

After both independent Val runs pass, compare the crossed Q/route effects with
the built-in matched-run check:

```bash
export ROUTE_ORIGINAL_FIT=/absolute/path/to/original_q_run/fit
export ROUTE_ORIGINAL_EVAL=/absolute/path/to/original_q_run/independent_val
export ROUTE_BUILDING_FIT=/absolute/path/to/building_q_run/fit
export ROUTE_BUILDING_EVAL=/absolute/path/to/building_q_run/independent_val
bash "$ROUTE_EXP/run.sh" cross_q
```

It requires the same E0 protocol, code, readout update count, threshold,
soft-floor, train-image order and parameter counts. It prints the Q gain within
each fixed route and the Q×hard-route interaction in percentage points.

No Test images are accessed by this package. Select designs on Train/Val;
the existing Test878 results remain the frozen prior report.
