# Validation record

2026-10-16, cloud workspace CPU only. Python 3.12, PyTorch 2.14.1+cpu.

```
/workspace/bright-lr-test-env/bin/python -m unittest discover -s /workspace/bright-followup/joint_loss_after500_v1 -p 'test_*.py' -q
Ran 54 tests
OK
```

All 54 tests executed; no skipped tests in this local run. `bash -n start_server.sh`, Python compileall and CLI help also passed.

Evidence covers:

- NLL and present-class Lovasz value and gradient parity, including functions extracted from the current BRIGHT source via AST without importing its model dependencies.
- Candidate removes exactly the background numerator, leaves the original present-class denominator and all valid pixels intact. Absent classes, all-background and all-ignore cases are exercised.
- Independent D model/AdamW/RNG forks from real 500-step tiny CPU optimizer states. Exact control update matches the inherited reference update. Candidate differs with background present and agrees without background.
- Same original LR schedule for both branches, actual optimizer steps, RNG advancement once per shared update, no shared tensor storage.
- Common/candidate objective aggregation, per-class and per-event reporting, paired transition counts, saved predictions reopened, split/support exclusion and evaluation RNG/module-mode restoration.
- Original artifact hashes and completed-source requirements, duplicate lock, non-overwriting output directories.
- Simulated worker lifecycle preserves first501 data/RNG after initial Val and fixedTrain evaluation. Runtime/evaluation mocks in this lifecycle test are not real CUDA execution.
- Failed smoke stops formal training; zero exit alone cannot promote incomplete evidence; smoke and pending integrity cannot claim full formal evidence.

Not validated here: BRIGHT Train2918/Val450 data, production checkpoint contents, the server's PyTorch 2.2.1+cu121 stack, CUDA kernels, GPU memory peak, elapsed training time, or metric improvement. `full` performs real CUDA smoke first and fails closed on its source/protocol/replay checks.

This delivery changes only its new helper package. Existing BRIGHT model/source code, original experiment files and case banks remain unchanged.
