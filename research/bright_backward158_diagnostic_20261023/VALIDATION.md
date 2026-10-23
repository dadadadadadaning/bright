# Validation scope

Validated locally with Python3.10 / CPU PyTorch2.1.2. No user dataset or RTX4090 was available.

- `python -m unittest discover -q`:36 tests passed with no skips, covering observer, launcher and replay contracts.
- Model and AdamW restore produce the same next CPU update as uninterrupted execution.
- Saved Gather/Sort index mutation, including a valid-index mutation, stops before the respective node kernel.
- Native PyTorch version errors remain enabled; no saved-tensor replacement hooks.
- Backward failure leaves parameters and optimizer state unchanged.
- Background launcher checks occupancy/leases, enforces a single attempt and1800second limit, and never promotes diagnostic evidence to experiment completion.
- Additional CPU integration used the actual original classifier/slots at small dimensions and a synthetic NLL+sort objective:46 index nodes observed; parameters and AdamW state exactly matched uninstrumented experiment.update.
- CLI help, Python compilation and reproducible zipapp build checked.

The eight-step GPU replay remains unexecuted here. Passing this diagnostic cannot prove the historical CUDA failure fixed or clear hardware. A node recorded as last observed is not necessarily the cause if CUDA fails after observation.
