# Q × Memory completed-run report

Independent, read-only interpretation of `damageformer_srac_q_memory_samepass_v1` output. It does not run the network, read BRIGHT images, or alter E0/E1 and the same-pass result. It rejects incomplete, non-reproduced, non-immutable, or non-Val450 runs. The report compares fixed-J1 rescue and harm rates within hard Q-owner strata and reports separate soft-Q mixing exposure; these associations are **not causal** and do not count buildings.

```bash
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export Q_SAMEPASS_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/q_memory_samepass_YYYYMMDD_HHMMSS
bash bda_benchmark/experiments/damageformer_srac_q_memory_report_v1/run.sh preflight
bash bda_benchmark/experiments/damageformer_srac_q_memory_report_v1/run.sh report
```

The human report appears on stdout, and `q_memory_report.json` is written without overwriting an existing file. Source outputs `summary.json` and `audit.json` are never modified. Re-running the report should use a different `--json-output` path, or run the module with only `--run` to print the JSON to stdout as well.
