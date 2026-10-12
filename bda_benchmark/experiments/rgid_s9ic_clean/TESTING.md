# Testing

Run only the isolated files:

```bash
python -m pytest -v \
  tests/test_targets_losses.py \
  tests/test_geometry.py

python scripts/smoke_test_synthetic.py
bash run_all_tests.sh
```

Do not run an unscoped `pytest` command from the BRIGHT repository root,
because that would collect legacy S9 tests outside this experiment.

For BF16 safety, the head forward may use autocast while all center and
boundary loss arithmetic is explicitly converted to FP32.
