# Validation record

2026-10-16: 29 CPU tests passed using Python 3.12 and PyTorch 2.14.1+cpu / NumPy 2.5.2.
The user's server uses a different existing environment (Python 3.10, PyTorch 2.2.1+cu121).
No actual BRIGHT data, saved user checkpoint tensors or CUDA execution were available here.

Checked with real CPU tensors/AdamW updates:

- Complete step-500 optimizer moments, counters and RNG; independent copies for all four branches.
- Original warmup/cosine horizon and strict step-501 multiplier boundary.
- Matched stochastic gradients, RNG advance, finite updates and actual optimizer-step counts.
- Source-file/checkpoint manifests and contiguous Train order; no promotion of incomplete old runs.
- Split/event exclusion, evaluation mode/RNG restoration, paired pixel transitions.
- Saved prediction reopening, tamper rejection, smoke vs full-validation evidence separation.

`study.py` byte compilation and `start_server.sh` Bash syntax check passed.
The original fix2 package was verified independently against its release manifest and the
recorded source hash `b84e2867a4117911b8279e2f97ccf9c5fbd021a7dc539f59e6bbb06396546142`.

These checks validate the new driver contracts, not model quality or server compatibility.
The server's real smoke, checkpoint-500 Val450 replay, training and final integrity checks
remain required. No Test is included and no numerical gain is promised.
