# Validation

Local environment: Python3.10 and CPU PyTorch2.1.2; no user GPU or BRIGHT data available.

`PYTHONWARNINGS=ignore python -B -m unittest discover -q`:24 CPU tests passed.

Coverage:

- Paired preflight accepts a complete observed-window fixture and rejects changed runner bytes, incomplete updates, missing restore evidence, mismatched checkpoint provenance and contradictory index coverage.
- The plain wrapper directly calls the installed original update function. CPU model/AdamW updates match direct calls; anomaly detection and new autograd observer hooks are not installed.
- Exceptions retain the original error and do not claim a failed update had zero parameter changes when a post-step check could have failed.
- Scalar comparison is descriptive, supports a failed partial window, and never claims full gradient/parameter identity or historical fault resolution.
- Launcher tests cover inherited leases, occupied GPU refusal, one attempt,30second heartbeat,1800second timeout, owned-process-group termination and preservation of original failure evidence.
- The supervisor log FD is checked for O_APPEND; both supervisor and worker append to the shared text log.
- CLI help, Python compilation and reproducible standard-library zipapp build checked.

These tests validate the diagnostic implementation, not CUDA stability. The planned server run is limited to151–158 and writes no new formal checkpoint or Val/Test result.
