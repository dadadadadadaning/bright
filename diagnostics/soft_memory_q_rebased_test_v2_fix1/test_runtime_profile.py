"""CPU regression tests for the exact recorded Q runtime restoration contract.

Fresh processes are intentional: environment and inter-op thread settings have
process-lifetime effects. These tests do not load a model or any Test images.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import runtime_profile as profile


HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("BRIGHT_TEST_REPO", "/workspace/bright"))
PYTHON = sys.executable
NUMERIC = "bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.numeric"


def run_python(source, *args, env=None):
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join((str(HERE), str(REPO)))
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        child_env.update(env)
    result = subprocess.run([PYTHON, "-c", source, *map(str, args)],
                            text=True, capture_output=True, env=child_env, timeout=90)
    if result.returncode:
        raise AssertionError("Fresh process failed:\n" + result.stdout + result.stderr)
    return result.stdout


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class RecordedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Obtain the actual original runtime schema from the installed CPU
        # Torch build, including immutable version/platform/hardware fields.
        raw = run_python("""
import os, json
for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_TF32_OVERRIDE','OMP_NUM_THREADS',
            'MKL_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG'):
    os.environ.pop(key, None)
import torch
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1 import numeric
torch.set_num_threads(3)
torch.set_num_interop_threads(2)
assert not torch.cuda.is_initialized()
print(json.dumps(numeric.configure_runtime('legacy', deterministic=False)))
""")
        cls.full = json.loads(raw)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="runtime_profile_test_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cp = self.root / "train/checkpoints/checkpoint_003000.pt"
        self.cp.parent.mkdir(parents=True)
        self.cp.write_bytes(b"Synthetic bootstrap-only checkpoint; never torch.load")
        self.side_path = self.cp.with_suffix(".pt.json")
        self.manifest_path = self.root / "train/manifest.json"
        self.e0_path = self.root / "e0.json"
        self.source_path = self.root / "source.json"
        self.e0 = {"profile": "legacy", "numeric_overrides": {"deterministic": False},
                   "runtime_settings": copy.deepcopy(self.full)}
        self.compact = {short: self.full[long] for short, long in profile.OVERLAP.items()}
        self.compact["cuda_launch_blocking"] = "1"
        self.bind()

    def bind(self):
        write_json(self.e0_path, self.e0)
        source = {"format": "SOFT_MEMORY_Q_SOURCE_V1",
                  "e0_metadata_protocol": str(self.e0_path),
                  "e0_metadata_protocol_sha256": profile.digest(self.e0_path)}
        write_json(self.source_path, source)
        info = {"source_protocol": str(self.source_path),
                "source_protocol_sha256": profile.digest(self.source_path),
                "execution_settings": self.compact}
        self.side = {"file": self.cp.name, "step": 3000, "smoke": False,
                     "mode": "train", "sha256": profile.digest(self.cp), "protocol_info": info}
        self.manifest = {"protocol_info": copy.deepcopy(info),
                         "arguments": {"protocol": str(self.source_path)}}
        write_json(self.side_path, self.side)
        write_json(self.manifest_path, self.manifest)

    def test_stdlib_bootstrap_has_no_torch_or_numpy_import(self):
        run_python("""
import sys
import runtime_profile as p
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
value = p.read_bound_profile(sys.argv[1], 3000)
assert value['full']['num_threads'] == 3
assert value['full']['num_interop_threads'] == 2
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
""", self.root)

    def test_restored_profile_passes_original_full_exact_comparison(self):
        # Reproduce a poisoned launcher environment. Restore before importing
        # Torch, then exercise the untouched original numeric configure/gate.
        run_python("""
import sys, os
import runtime_profile as p
value = p.bootstrap_cli(['run','--train-root',sys.argv[1],'--step','3000'], '/missing')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
assert os.environ.get('OMP_NUM_THREADS') is None
assert os.environ.get('MKL_NUM_THREADS') is None
assert os.environ.get('CUDA_LAUNCH_BLOCKING') == '1'
import torch
from unittest.mock import patch
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1 import numeric
# Full runtime_settings may initialize CUDA on a GPU machine. Host restoration
# must not call it, cuda.is_available, current_device, or property inspection.
with patch.object(numeric, 'runtime_settings', side_effect=AssertionError('early full profile')), \
     patch.object(torch.cuda, 'is_available', side_effect=AssertionError('early CUDA availability')), \
     patch.object(torch.cuda, 'current_device', side_effect=AssertionError('early CUDA device')):
    p.restore_torch(value, torch)
assert not torch.cuda.is_initialized()
actual = numeric.configure_runtime('legacy', deterministic=False)
assert actual == value['full'], p.differences(value['full'], actual)
assert not torch.cuda.is_initialized()
""", self.root, env={"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                         "NVIDIA_TF32_OVERRIDE": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})

    def test_old_two_one_thread_policy_reproduces_gate_mismatch(self):
        run_python("""
import sys, os
import runtime_profile as p
value = p.read_bound_profile(sys.argv[1], 3000)
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '2'
import torch
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1 import numeric
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
actual = numeric.configure_runtime('legacy', deterministic=False)
assert actual != value['full']
fields = {row['field'] for row in p.differences(value['full'], actual)}
assert {'num_threads','num_interop_threads','environment.OMP_NUM_THREADS',
        'environment.MKL_NUM_THREADS'} <= fields, fields
""", self.root)

    def test_recorded_nonnull_environment_restored_before_import(self):
        self.e0["runtime_settings"]["environment"].update(OMP_NUM_THREADS="3", MKL_NUM_THREADS="3")
        self.bind()
        run_python("""
import sys, os
import runtime_profile as p
value = p.read_bound_profile(sys.argv[1], 3000)
p.restore_environment(value)
assert os.environ['OMP_NUM_THREADS'] == os.environ['MKL_NUM_THREADS'] == '3'
import torch
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1 import numeric
p.restore_torch(value, torch)
assert numeric.configure_runtime('legacy', deterministic=False) == value['full']
""", self.root)

    def test_nondefault_recorded_flags_and_dtype_are_restored(self):
        full = self.e0["runtime_settings"]
        full.update(default_dtype="torch.float64", deterministic_algorithms=True,
                    deterministic_warn_only=True, cudnn_deterministic=True,
                    cudnn_benchmark=False, cudnn_enabled=False, cudnn_allow_tf32=False,
                    cuda_matmul_allow_tf32=False, float32_matmul_precision="highest",
                    fp16_reduced_precision_reduction=False,
                    bf16_reduced_precision_reduction=False)
        self.e0["numeric_overrides"] = {}
        self.compact.update({short: full[long] for short, long in profile.OVERLAP.items()})
        self.bind()
        run_python("""
import sys
import runtime_profile as p
value = p.read_bound_profile(sys.argv[1], 3000)
p.restore_environment(value)
import torch
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1 import numeric
p.restore_torch(value, torch)
actual = numeric.configure_runtime('legacy')
assert actual == value['full'], p.differences(value['full'], actual)
assert torch.get_default_dtype() is torch.float64
assert torch.are_deterministic_algorithms_enabled()
assert torch.is_deterministic_algorithms_warn_only_enabled()
assert not torch.cuda.is_initialized()
""", self.root)

    def test_fresh_cuda_context_is_required(self):
        run_python("""
import sys
import runtime_profile as p
value = p.read_bound_profile(sys.argv[1], 3000)
p.restore_environment(value)
import torch
from unittest.mock import patch
with patch.object(torch.cuda, 'is_initialized', return_value=True):
    try:
        p.restore_torch(value, torch)
    except ValueError as error:
        assert 'fresh CUDA process' in str(error)
    else:
        raise AssertionError('Initialized CUDA context was accepted')
""", self.root)

    def test_environment_cannot_be_changed_after_library_import(self):
        for library in ("torch", "numpy"):
            with self.subTest(library=library):
                run_python("""
import sys
import runtime_profile as p
value = p.read_bound_profile(sys.argv[1], 3000)
sys.modules[sys.argv[2]] = object()
try:
    p.restore_environment(value)
except ValueError as error:
    assert 'before importing Torch or NumPy' in str(error)
else:
    raise AssertionError('Late environment change was accepted')
""", self.root, library)

    def test_compact_full_profile_disagreement_rejected(self):
        for field in ("matmul_allow_tf32", "torch_version", "float32_matmul_precision"):
            with self.subTest(field=field):
                original = self.compact[field]
                self.compact[field] = not original if isinstance(original, bool) else "contradictory"
                self.bind()
                with self.assertRaisesRegex(ValueError, "profiles disagree: " + field):
                    profile.read_bound_profile(self.root, 3000)
                self.compact[field] = original

    def test_bound_hash_tampering_is_rejected(self):
        for path, message in ((self.cp, "checkpoint identity/hash"),
                              (self.source_path, "source hash"),
                              (self.e0_path, "E0 metadata hash")):
            with self.subTest(path=path.name):
                before = path.read_bytes()
                path.write_bytes(before + b" ")
                with self.assertRaisesRegex(ValueError, message):
                    profile.read_bound_profile(self.root, 3000)
                path.write_bytes(before)

    def test_sidecar_manifest_disagreement_is_rejected(self):
        self.manifest["protocol_info"]["execution_settings"]["cuda_launch_blocking"] = None
        write_json(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, "checkpoint/manifest protocol differs"):
            profile.read_bound_profile(self.root, 3000)

    def test_environment_schema_never_accepts_arbitrary_keys(self):
        self.e0["runtime_settings"]["environment"]["LD_PRELOAD"] = "/untrusted.so"
        self.bind()
        with self.assertRaisesRegex(ValueError, "Unexpected recorded environment schema"):
            profile.read_bound_profile(self.root, 3000)
        run_python("""
import runtime_profile as p
env = dict.fromkeys((*p.ENV_KEYS, 'CUBLAS_WORKSPACE_CONFIG','CUDA_LAUNCH_BLOCKING'))
env['LD_PRELOAD'] = '/untrusted.so'
try:
    p.restore_environment({'environment': env})
except ValueError as error:
    assert 'only recorded numerical keys' in str(error)
else:
    raise AssertionError('Arbitrary environment mutation accepted')
""")

    def test_unmodifiable_platform_difference_is_preserved_and_reported(self):
        self.e0["runtime_settings"]["platform"] = "a different recorded operating system"
        self.bind()
        run_python("""
import sys
import runtime_profile as p
value = p.read_bound_profile(sys.argv[1], 3000)
p.restore_environment(value)
import torch
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1 import numeric
p.restore_torch(value, torch)
actual = numeric.configure_runtime('legacy', deterministic=False)
assert actual != value['full']
rows = p.differences(value['full'], actual)
assert len(rows) == 1 and rows[0]['field'] == 'platform', rows
assert actual['platform'] != 'a different recorded operating system'
""", self.root)

    def test_difference_report_distinguishes_missing_from_null(self):
        rows = profile.differences({"environment": {"OMP_NUM_THREADS": None}, "version": "old"},
                                   {"environment": {}, "version": "new", "device": None})
        indexed = {row["field"]: row for row in rows}
        self.assertEqual(indexed["environment.OMP_NUM_THREADS"]["missing_from"], "actual")
        self.assertEqual(indexed["device"]["missing_from"], "expected")
        self.assertEqual(indexed["version"]["actual"], "new")

    def test_profile_binding_rechecked_after_bootstrap(self):
        value = profile.read_bound_profile(self.root, 3000)
        formal = SimpleNamespace(checkpoint=self.cp, checkpoint_manifest=self.side,
                                 saved={"protocol_info": self.side["protocol_info"]})
        profile.check_binding(value, formal)
        self.e0_path.write_text(self.e0_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "source changed after bootstrap"):
            profile.check_binding(value, formal)

    def test_help_and_report_bootstrap_require_no_source_inputs(self):
        run_python("""
import os, sys
import runtime_profile as p
before = dict(os.environ)
for args in ([], ['--help'], ['run','--help'], ['plan','-h'],
             ['report','--output','/missing-result'], ['--report','/missing-result']):
    assert p.bootstrap_cli(args, '/definitely-missing-training-root') is None
assert dict(os.environ) == before
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
""")

    def test_launcher_help_requires_no_training_directory(self):
        result = subprocess.run([PYTHON, str(HERE / "start.py"), "--help",
                                 "--train-root", "/missing-training-directory"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--step", result.stdout)


if __name__ == "__main__":
    unittest.main()
