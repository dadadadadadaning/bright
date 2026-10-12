# Clean installation

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/rgid_s9ic_clean

python -m pip uninstall -y rgid-s9ic || true
python -m pip install -e . --no-deps --no-build-isolation

python - <<'PY'
import rgid
print(rgid.__file__)
assert "experiments/rgid_s9ic_clean/rgid" in rgid.__file__
PY
```

The import must resolve to this experiment's `rgid/__init__.py`, never to
the BRIGHT repository root.
