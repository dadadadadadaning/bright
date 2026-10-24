# Calibration 800 → 2500 recovery

This package continues the failed `calibration` control from its saved step800
checkpoint. It creates a new directory and does not overwrite the installed
severity study or its failed run. It is not a new Memory method and does not
establish that the historical CUDA fault is fixed.

## Known failure and recovery boundary

Parent:
`/root/shared-nvme/BRIGHT-checkpoints/SRAC_SEVERITY_MEMORY_V1/night_3407_z_yxvq4s/train_calibration`

- Last fully completed update: 803.
- Failure: backward at update804, `la_palma-volcano_00000270`.
- Saved recovery state: checkpoint800.
- Pinned SHA256: `23766189b371dff7c62eb373baaf78c0e216721e13be23623255e4771955a747`.

The worker restores the model, AdamW, RNG and raw data position. It consumes
the first800 raw batches without recomputing frozen features or updates, then
restores checkpoint RNG. Updates801–803 are performed again and their sample
identities and scalar loss differences are recorded. A scalar difference is
descriptive, not proof of bitwise historical CUDA replay. The known804 image
is checked explicitly.

## Only execution change

Published calibration computes its auxiliary label evidence and multiplies
its learned gain by zero. Autograd nevertheless executes the evidence branch.
In `lean_adapter.py`, calibration alone uses `evidence.detach()` in the
correction. The gain remains connected with a zero gradient, rather than
`None`, retaining AdamW state/weight decay semantics.

All forward readout values and diagnostics remain computed. The original
visual Reader, its trainable label embedding and gradients, supports, routes,
NLL + Lovasz objective, parameter schema and training schedule are retained.
Other modes are unchanged, but this runner starts only calibration.

The change removes18 zero-contribution GatherBackward nodes in the tested
small CPU fixture (63 → 45). It does not remove the original Reader or Lovasz
backward operations. The failing CUDA node at804 has not been identified.
Neither a successful CPU test nor a subsequent successful GPU continuation
proves the original fault's software/hardware cause.

The unchanged parent plan is retained for checkpoint compatibility. The new
manifest, recovery lineage, training rows, evaluation reports and checkpoint
sidecars explicitly declare `execution_change=calibration_evidence_detached`.
The new immutable runner is saved in its output directory and hash-bound to
its launch configuration. Do not detach sidecars/lineage from archived models.

## Execution scope

- 1700 new optimizer updates:801–2500.
- 34 new checkpoints:850–2500, every50; each scheduled validation follows a
 saved checkpoint.
- Full Val450 at1000,1500,2000,2500, then the original fixed Train450 at2500.
- The parent's initial Val and Val500 are verified, read-only references.
 Their pending final-integrity flags are preserved. This recovery does not
 promote those old reports or declare the whole multi-method study complete.
- No Test, other variants, automatic retry, skipped samples, index repair,
 probability normalization repair or CUDA-error continuation.

## Run

After downloading the pinned zipapp, use the existing server Python:

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_calibration_resume_v1.pyz start
```

`start` verifies parent/checkpoint/source evidence, checks GPU availability and
disk headroom, acquires the shared study lock, creates `calibration_resume_*`,
and launches a detached supervisor. The default time limit is12 hours, with
heartbeats every30 seconds. It exports `CUDA_LAUNCH_BLOCKING=1` before Torch
loads. Allocation estimation reserves twice34 checkpoint payloads plus512MiB
for metadata; this is a preflight estimate, not a disk usage guarantee.

The output includes exact run-local commands. General commands select this
package's latest calibration recovery:

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_calibration_resume_v1.pyz logs
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_calibration_resume_v1.pyz status
```

To verify without starting a GPU job, use `check`. Do not invoke the internal
worker/supervisor commands directly. Closing SSH does not stop the background
job. `logs` prints current tails; it does not start or resume training.

On completion inspect `phase=complete`, `training_complete=true`,
`validation_complete=true`, `fixed_train_complete=true`,
`last_complete_step=2500`, `new_complete_updates=1700`, `error=null`, and
the supervisor's returncode0. `study_complete=false` and
`historical_failure_resolved=false` are intentional.

## Comparing the experiment

Compare recovered calibration with the completed original at matching1000,
1500,2000,2500 nodes, particularly fixed2500. A later severity_memory study
must beat calibration to establish added value beyond class-bias adjustment;
correct-vs-shuffled additional readout must then support label correspondence.
Do not compare a Val score to an older Test score as evidence of improvement.
One seed cannot establish stable gains.

If this continuation fails, preserve `first_failure.json`, `traceback.txt`,
summary and the latest checkpoint. It stops; it does not repeatedly relaunch
or silently recover bad tensors. No guarantee of GPU fault resolution is made.
