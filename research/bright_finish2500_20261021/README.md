# Finish the six-arm experiment at2500, then evaluate Test878

The previous training process reached checkpoint2000 and failed during validation. The subsequent synchronous Val450 program completed evaluation only; it did not update parameters or continue training. This workflow completes the original2500-step budget and then evaluates the fixed2500 endpoint on Test878.

This is a continuation of the existing seed3407 six-arm experiment, not a new model or loss. It does not imply the historical asynchronous CUDA fault is fixed.

## Pinned inputs

- Original experiment: `SRAC_SLOT_MEMORY_ABLATION_V1/run_20261020_025710_3840142`.
- Continuation parent: `SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/resume_20261021_112222_495713`, checkpoint2000.
- Completed boundary Val450 certificate: `SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1/sync_val2000_hkpnci6r`.
- Certificate evaluator SHA256: `14a3cb440ce49899da7d7b81061b9b2df66ab88526203919f28823bf8b3617ab`.
- Source root on the server: `/root/shared-nvme/BRIGHT-checkpoints`.

Preflight verifies the original failed-run lineage, checkpoint, code releases, boundary evaluation journal, protocol, hashes and integrity flags. It never rewrites or promotes old failed runs.

## Execution

Use the original server interpreter:

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_finish2500_test_v1.pyz start
```

The launcher makes a new `SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/finish2500_*` directory, copies the archive into it, and starts a detached coordinator. The new root and PID are printed. Training and assessment locks prevent overlap with cooperating existing launchers.

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_finish2500_test_v1.pyz logs
```

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_finish2500_test_v1.pyz status
```

These commands resolve the saved latest root, so they do not depend on shell variables surviving an SSH reconnect. Ctrl+C exits log viewing; it does not terminate the detached workflow. Do not repeat `start` to query progress.

Stages are sequential, in separate Python/CUDA processes:

1. Restore six heads, AdamW states, RNG, original shuffled sample order, and update steps2001–2500. Original NLL+Lovasz, learning-rate schedule, clip threshold, data, bank and execution settings remain unchanged. Save a complete paired checkpoint every50 steps and at2500. No validation runs inside this process.
2. Evaluate the original fixed-Train subset at checkpoint2000, with evaluation transforms.
3. Evaluate all450 Val images at checkpoint2500; this is the original planned primary endpoint.
4. Evaluate the same fixed-Train subset at checkpoint2500.
5. Verify the completed2500 Val/fixed-Train certificates, then score canonical Test878 at checkpoint2500 for Grid and E/F/B/D/R/L. No optimizer updates, loss-based fitting, calibration or checkpoint selection occur on Test.

Assessment processes explicitly use `CUDA_LAUNCH_BLOCKING=1`, matching the successful standalone Val2000 check. The original training process retains its recorded setting. Evaluation is RNG neutral; deferring it preserves the intended sample/update schedule, but no bitwise equivalence to the historical live CUDA process is claimed. A CUDA fault still stops the failing stage; no images are skipped, probabilities clamped, validation checks removed, or automatic retries performed. Checkpoint2500 survives a later assessment failure.

## Reports

`workflow_summary.json` is the completion authority for the new workflow. Success requires all of:

```json
{"phase":"complete","complete":true,"training_complete":true,"validation_complete":true,"test_complete":true,"last_complete_step":2500}
```

The top-level `summary.json` is the immutable training-component record. Its `training_complete_evaluation_pending` phase records the point at which the training process exited; later assessment completion is recorded in `workflow_summary.json`. Keeping this record unchanged preserves its fingerprint in assessment provenance. It does not indicate that the coordinator is still waiting after workflow completion.

Results:

- `train.jsonl`: copied verified2000-row prefix followed by500 new complete paired rows.
- `checkpoints/checkpoint_002500.pt`: full six-head/optimizer/RNG checkpoint.
- `assessments/val2500/val_step_002500/result.json`: full Val450 metrics and inherited objective/mechanism diagnostics.
- `assessments/test2500/test_step_002500/result.json`: Test878 metrics, class confusion matrices, per-event results, same-pixel rescue/harm counts.
- `assessments/*/images.jsonl` is under each stage's `*_step_*` directory; identities/order and count matrices are verified by reopening the journal.
- `metrics.csv`: side-by-side Val2000, Val2500 and Test2500 with four-class mIoU, foreground mIoU and each arm's within-split difference from Grid.
- `workflow.log`, each assessment's `summary.json`, and any `traceback.txt` identify the failing stage without hiding it behind CUDA cleanup errors.

Test admission is explicit and confined to the Test process. It verifies canonical Train/Val membership, exactly878 unique Test names and split disjointness, then hashes Test input files. The runtime's `val_names` remains Val450; only the evaluation-query admission set is extended. Train-only memory banks, support exclusion rules and query-GT separation checks remain active. Test reports have `split=test`, `test_accessed=true`, `exploratory_test=true`, and `full_validation_evidence=false`.

Both mIoUs use the original class order background/intact/damaged/destroyed. Four-class mIoU averages all four IoUs; foreground mIoU averages classes1/2/3. Missing-class IoU contributes zero to these fixed-denominator means. Test also reports pixel OA from the same confusion matrices. No object-level F1 is inferred.

The fixed2500 endpoint is selected before reading these Test scores. Because this Test set has already informed earlier discussions, report this round as exploratory Test evaluation. Val2000 vs Val2500 diagnoses training-length effects; Test2500 compares methods at the common endpoint. Do not mix either with the older Extended V2 experiment's Test values as if they were the same run.

## Build and validation

Run `python build.py` in this directory to rebuild the deterministic archive from the pinned synchronous-Val archive. `FINISH_BASE.json` records byte-identical inherited helpers. See `VALIDATION.md` for CPU checks and limits; real CUDA completion must be established on the server.
