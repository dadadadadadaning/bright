# Real-model tensor lifetime diagnostic

The preceding observer pair (`observer_ab1500_co1ab40o`) completed one image
under memcheck in each case, with zero sanitizer errors. It nevertheless retained
13.2634 GiB (`light`) and 13.2653 GiB (`watched`) in the PyTorch allocator after
scoring. The roughly 1.94 MiB difference does not explain the earlier multi-GiB
growth. Both previous cases shared passive predicate wrappers, failure wrappers
and scalar forward hooks. Basic synthetic tensor lifetime tests had passed.

This tool narrows that remaining distinction. Both cases use fresh processes,
Compute Sanitizer memcheck, checkpoint1500 and the same first canonical Test
image (`bata-explosion_00000007`), with all original six arms and model validators:

- `minimal`: no added forward hooks, route predicate wrappers, route snapshots
  or successful-call wrappers around runtime.prepare / factory.forward. Original
  model feature-capture hooks remain. CPU stage labels in the scoring loop and
  scalar memory observations before/after the whole evaluation remain.
- `instrumented`: the previous `light` passive predicate/failure wrappers and
  scalar forward hooks, without the additional route snapshot observers.

After scoring returns and wrappers have been restored, both cases record scalar
allocator counters, enumerate Python-visible tensor metadata using weakrefs,
record counters again, call `gc.collect()` once, and record the response. There
is no `empty_cache()`, new forward, optimizer update, route repair, normalization,
dtype change or retry. Python-visible tensor metadata is not a complete census
of tensors owned only by C++. Inventory enumeration can itself affect garbage
collection, so pre-inventory and pre-explicit-GC readings are separate.

The same predictions, source/inputs, checkpoint, head/optimizer state, RNG and
pool guard integrity checks remain. A CUDA/other inference exception propagates;
the failed case performs no subsequent RNG restoration or GC/memory sampling.
No diagnostic result is promoted to full Test878 evidence. A pass does not
resolve previous route bit changes or prove/clear a hardware fault.

## Commands

Run with the server Python, using the downloaded immutable archive:

```bash
python -B bright_model_lifetime_v1.pyz start
python -B bright_model_lifetime_v1.pyz logs
python -B bright_model_lifetime_v1.pyz status
python -B bright_model_lifetime_v1.pyz stop
```

The launcher requires an idle GPU, reuses the existing job leases, creates a new
`model_lifetime1500_*` directory and detaches from SSH. Each subprocess is limited
to 600 seconds by default. The pair stops at its first unsuccessful case.
Ctrl+C in `logs` stops the tail; `stop` stops the job. `status` reports the final
pair, peak/end memory and `gc_observation`. No automatic full Test or training.
Original outputs remain untouched. New metadata/journals occupy disk; no model
checkpoint or per-image GPU tensor dump is written during successful execution.

## Interpretation

- Minimal pre-GC occupancy much lower: added wrappers/hooks or their interaction
  with this runtime are implicated; isolate the particular component next.
- Both high but GC releases substantial allocation: Python object lifetimes /
  cycles become a supported explanation for that retained memory. It does not
  establish why the old route bits changed, or validate GC as a production fix.
- Both high and GC releases little: inspect remaining original capture hooks,
  model/operator/native runtime lifetimes. A Python inventory cannot exclude
  C++ retention or allocator/runtime issues.

This is deliberately one image, not the historical image355 replay. Full missing
Test1500 and Test2000 remain separate work after the abnormal path is understood.

The build preserves the preceding archive's modules byte-for-byte and adds new
`life_*` modules. `MODEL_LIFETIME_BASE.json` records every inherited member hash.
