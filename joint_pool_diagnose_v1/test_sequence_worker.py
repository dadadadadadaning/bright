"""CPU contracts for no-training replay orchestration, not real BRIGHT/CUDA."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import sequence_worker as worker


def module_hash(module):
    import hashlib
    result = hashlib.sha256()
    for name, value in module.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().numpy().tobytes())
    return result.hexdigest()


class Fixture:
    def __init__(self, base):
        self.base = Path(base)
        self.repo, self.tool, self.old = (self.base / name for name in ("repo", "old_tool", "old_run"))
        for path in (self.repo, self.tool, self.old / "train/pool_debug"):
            path.mkdir(parents=True)
        (self.tool / "train.py").write_text("# immutable source\n")
        self.train_names = ["train_%04d" % n for n in range(2918)]
        self.val_names = ["event_%04d" % n for n in range(450)]
        self.protocol = {"source_runtime_settings": {"cublas_workspace_config": None}, "paired_bank_sha256": "frozenbank"}
        self.config = dict(repo=str(self.repo), mode="train", steps=5000, seed=3407,
                           lr=1e-4, weight_decay=5e-3, chunk_size=64,
                           output=str(self.old / "train"))
        self.seed_calls, self.evaluate_calls, self.runtime_instances = [], [], []
        self.checked_verify_count = 0
        self.module_hash_calls = 0
        self.fail_evaluate = False
        self.change_old_file = False
        self.change_protocol = False
        self.change_val_order = False
        self.change_code = False
        self.fail_coverage = False
        self.fake_input = self.base / "frozen_image.dat"
        self.fake_input.write_bytes(b"sample")
        self.initial_heads, self.hashes = self.new_heads(SimpleNamespace(**self.config), "cpu")
        self.manifest = dict(arguments=self.config, source_hash="codehash",
                             protocol_info=self.protocol, train_names=self.train_names,
                             val_names=self.val_names, initial_head_hashes=self.hashes)
        worker.write_json(self.old / "train/manifest.json", self.manifest)
        worker.write_json(self.old / "train/summary.json", dict(phase="initial_val", updates_per_arm=dict.fromkeys(worker.ARMS, 0)))
        worker.write_json(self.old / "train/pool_debug/failure.json", {"image": self.val_names[240], "error": "old failure"})
        fixture = self

        class Runtime:
            def __init__(self, args, output):
                self.args, self.output = args, output
                self.device = "cpu"
                self.train_names, self.val_names = list(fixture.train_names), list(fixture.val_names)
                if fixture.change_val_order:
                    self.val_names.reverse()
                self.protocol_info = copy.deepcopy(fixture.protocol)
                if fixture.change_protocol:
                    self.protocol_info["paired_bank_sha256"] = "changed"
                self.input_ledger, self.guard_report = {}, None
                fixture.runtime_instances.append(self)
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                self.guard_report = {"restored": True, "restore_conflicts": [], "failures": []}
                return False

        def source_hash():
            return "changed" if fixture.change_code else "codehash"

        self.modules = SimpleNamespace(
            torch=torch,
            common=SimpleNamespace(source_hash=source_hash, module_hash=self.hash_module),
            run=SimpleNamespace(verify_release=lambda: {"files": 14, "code_sha256": "codehash"}),
            train=SimpleNamespace(_seed=lambda seed: self.seed_calls.append(seed),
                _names=lambda runtime, smoke: (runtime.train_names, runtime.val_names),
                _new_heads=self.new_heads, _evaluate=self.evaluate,
                _checked_verify=self.verify, _restored_guard=lambda runtime: runtime.guard_report),
            server_runtime=SimpleNamespace(Runtime=Runtime))

    def hash_module(self, head):
        self.module_hash_calls += 1
        return module_hash(head)

    def new_heads(self, args, device):
        torch.manual_seed(args.seed)
        first = torch.nn.Linear(2, 4)
        heads = {arm: copy.deepcopy(first) for arm in worker.ARMS}
        return heads, {arm: module_hash(head) for arm, head in heads.items()}

    def verify(self, runtime):
        self.checked_verify_count += 1
        return {"files": True, "models": True, "runtime": True}

    def evaluate(self, runtime, heads, names, folder, **kwargs):
        self.evaluate_calls.append((list(names), kwargs))
        folder.mkdir()
        runtime.input_ledger[self.fake_input] = worker.digest(self.fake_input)
        if self.fail_evaluate:
            (folder / "images.jsonl").write_text('{"image": "earlier"}\n')
            (runtime.output / "pool_debug").mkdir()
            worker.write_json(runtime.output / "pool_debug/failure.json", {"image": names[0], "kind": "cuda error"})
            raise RuntimeError("synthetic CUDA failure")
        if self.change_old_file:
            (self.tool / "train.py").write_text("# accidentally changed\n")
        result = {"image_count": len(names), "whole_val450_processed": len(names) == 450,
                  "initial_probability_checks": {a: {"passed": True} for a in worker.ARMS},
                  "coverage": {a: {"supported_queries": 0 if self.fail_coverage else len(names)} for a in worker.ARMS},
                  "full_validation_evidence": False, "final_integrity_pending": True}
        worker.write_json(folder / "result.json", result)
        return result

    def args(self, scope="failed_image", **kw):
        values = dict(tool=str(self.tool), repo=str(self.repo), failed_run=str(self.old),
                      output=str(self.base / "diagnostic"), scope=scope, image=None)
        values.update(kw)
        return SimpleNamespace(**values)


class SequenceWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.fixture = Fixture(self.directory.name)
        self.env = patch.dict(os.environ, {"CUDA_LAUNCH_BLOCKING": "1"}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.directory.cleanup()

    def test_failed_image_replays_same_seed_initialization_and_never_steps(self):
        f = self.fixture
        original = f.old / "train/summary.json"
        old_digest = worker.digest(original)
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Training forbidden")):
            result = worker.run(f.args(), f.modules)
        self.assertTrue(result["diagnostic_complete"])
        self.assertTrue(result["gates_passed"])
        self.assertFalse(result["training"])
        self.assertFalse(result["formal_training_completed"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertFalse(result["whole_val450_processed"])
        self.assertEqual(result["updates_per_arm"], dict.fromkeys(worker.ARMS, 0))
        self.assertEqual(f.seed_calls, [3407, 3407])
        names, options = f.evaluate_calls[0]
        self.assertEqual(names, [f.val_names[240]])
        self.assertEqual(options, {"seed": 3407, "smoke": True, "initial": True, "save_arrays": True})
        self.assertEqual(result["optimizer_objects_created"], list(worker.ARMS))
        self.assertEqual(result["input_ledger_after"], {str(f.fake_input): worker.digest(f.fake_input)})
        self.assertEqual(worker.digest(original), old_digest)

    def test_val450_preserves_order_and_cannot_claim_trained_evidence(self):
        f = self.fixture
        result = worker.run(f.args(scope="val450"), f.modules)
        self.assertEqual(f.evaluate_calls[0][0], f.val_names)
        self.assertFalse(f.evaluate_calls[0][1]["smoke"])
        self.assertTrue(result["whole_val450_processed"])
        self.assertFalse(result["full_validation_evidence"])
        saved = worker.read_json(Path(f.args().output) / "initial_val/result.json")
        self.assertTrue(saved["diagnostic_only"])
        self.assertFalse(saved["training"])
        self.assertFalse(saved["formal_training_completed"])

    def test_cuda_failure_records_cpu_evidence_without_gpu_verification(self):
        f = self.fixture
        f.fail_evaluate = True
        with self.assertRaisesRegex(RuntimeError, "synthetic CUDA failure"):
            worker.run(f.args(), f.modules)
        self.assertEqual(f.checked_verify_count, 0)
        self.assertEqual(f.module_hash_calls, 0)
        report = worker.read_json(Path(f.args().output) / "summary.json")
        self.assertFalse(report["gates_passed"])
        self.assertEqual(report["persisted_image_rows"], 1)
        self.assertIn("pool_debug/failure.json", report["cpu_failure_observations"])
        self.assertTrue(report["cpu_failure_observations"]["pool_guard_report"]["restored"])
        self.assertTrue((Path(f.args().output) / "traceback.txt").is_file())

    def test_unknown_image_is_rejected_before_runtime(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            worker.run(self.fixture.args(image="not_a_val_image"), self.fixture.modules)
        self.assertFalse(self.fixture.runtime_instances)

    def test_source_code_drift_is_rejected_before_runtime(self):
        self.fixture.change_code = True
        with self.assertRaisesRegex(ValueError, "source hash"):
            worker.run(self.fixture.args(), self.fixture.modules)
        self.assertFalse(self.fixture.runtime_instances)

    def test_runtime_val_order_drift_is_rejected(self):
        self.fixture.change_val_order = True
        with self.assertRaisesRegex(ValueError, "identities or order"):
            worker.run(self.fixture.args(scope="val450"), self.fixture.modules)
        self.assertFalse(self.fixture.evaluate_calls)

    def test_bank_binding_drift_is_rejected(self):
        self.fixture.change_protocol = True
        with self.assertRaisesRegex(ValueError, "bindings differ"):
            worker.run(self.fixture.args(), self.fixture.modules)
        self.assertFalse(self.fixture.evaluate_calls)

    def test_preserved_evidence_drift_fails_end_gate(self):
        self.fixture.change_old_file = True
        with self.assertRaisesRegex(ValueError, "changed during replay"):
            worker.run(self.fixture.args(), self.fixture.modules)
        report = worker.read_json(Path(self.fixture.args().output) / "summary.json")
        self.assertFalse(report["diagnostic_complete"])
        self.assertFalse(report["preserved_inputs_unchanged"][str(self.fixture.tool / "train.py")])

    def test_empty_historical_support_cannot_pass_diagnostic(self):
        self.fixture.fail_coverage = True
        with self.assertRaisesRegex(ValueError, "no legal historical support"):
            worker.run(self.fixture.args(), self.fixture.modules)

    def test_requires_preimport_cuda_blocking_and_source_cublas(self):
        os.environ.pop("CUDA_LAUNCH_BLOCKING")
        with self.assertRaisesRegex(ValueError, "CUDA_LAUNCH_BLOCKING"):
            worker.environment_binding(self.fixture.manifest)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with self.assertRaisesRegex(ValueError, "CUBLAS_WORKSPACE_CONFIG"):
            worker.environment_binding(self.fixture.manifest)

    def test_existing_output_or_original_tree_never_overwritten(self):
        args = self.fixture.args(output=str(self.fixture.old / "train/new"))
        with self.assertRaisesRegex(ValueError, "must not be inside"):
            worker.run(args, self.fixture.modules)
        self.assertFalse(Path(args.output).exists())
        args = self.fixture.args(output=str(self.fixture.old))
        with self.assertRaises(FileExistsError):
            worker.run(args, self.fixture.modules)

    def test_failed_run_with_updates_not_treated_as_initial_failure(self):
        worker.write_json(self.fixture.old / "train/summary.json", {"phase": "training", "updates_per_arm": {"r0": 1, "rt": 1, "rf": 1}})
        with self.assertRaisesRegex(ValueError, "zero updates"):
            worker.run(self.fixture.args(), self.fixture.modules)


if __name__ == "__main__":
    unittest.main()
