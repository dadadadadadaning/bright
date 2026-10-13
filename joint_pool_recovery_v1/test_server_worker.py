"""CPU-only tamper tests for the checkpoint boundary, without real Runtime imports."""
from __future__ import annotations

from collections import OrderedDict
import copy
import importlib.util
import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from server_worker import state_hash, validate_loaded


def fixture():
    arguments = dict(mode="train", steps=5000, eval_every=500, repo="/repo",
                     v3_root="/bank", protocol="/e0/protocol.json", device="cuda:0",
                     workers=4, seed=3407, topk=64, chunk_size=64,
                     lr=1e-4, weight_decay=5e-3, output="/old/train")
    source_hash = "b" * 64
    protocol = {"source_protocol_sha256": "c" * 64,
                "paired_bank_sha256": "d" * 64,
                "source_runtime_settings": {"deterministic_algorithms": False}}
    heads = {}
    optimizers = {}
    for number, arm in enumerate(("r0", "rt", "rf")):
        weight = torch.arange(12, dtype=torch.float32).reshape(4, 3) + number
        bias = torch.arange(4, dtype=torch.float32) + number
        heads[arm] = OrderedDict(weight=weight, bias=bias)
        optimizers[arm] = {
            "state": {index: {"step": torch.tensor(1500.),
                              "exp_avg": torch.zeros_like(value),
                              "exp_avg_sq": torch.ones_like(value)}
                      for index, value in enumerate((weight, bias))},
            "param_groups": [{"lr": 8.83022221559489e-5, "betas": (0.9, 0.999),
                              "eps": 1e-8, "weight_decay": 5e-3, "amsgrad": False,
                              "foreach": None, "maximize": False, "capturable": False,
                              "differentiable": False, "fused": None, "params": [0, 1]}],
        }
    generator = torch.Generator(device="cpu").manual_seed(3407)
    saved = {
        "format": "SRAC_REGION_MEMORY_JOINT_TRAIN_V1", "mode": "train",
        "smoke": False, "training_complete": False, "step": 1500,
        "config": arguments, "source_hash": source_hash, "protocol_info": protocol,
        "initial_head_hashes": {arm: "1" * 64 for arm in heads},
        "updates_per_arm": {arm: 1500 for arm in heads}, "heads": heads,
        "optimizers": optimizers,
        "rng": {"python": random.Random(3407).getstate(),
                "numpy": np.random.RandomState(3407).get_state(),
                "torch_cpu": generator.get_state(),
                "torch_cuda": [generator.get_state().clone()]},
        "input_file_sha256": {"/dataset/train/a.tif": "a" * 64},
    }
    evidence = {"arguments": copy.deepcopy(arguments), "original_source_hash": source_hash,
                "protocol_info": copy.deepcopy(protocol), "checkpoint_step": 1500,
                "initial_head_hashes": copy.deepcopy(saved["initial_head_hashes"]),
                "sidecar": {"head_hashes": {arm: state_hash(state)
                                            for arm, state in heads.items()}}}
    return saved, evidence


class CPUCheckpointBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.saved, self.evidence = fixture()
        # Any accidental GPU initialization or availability probe is a failure.
        self.stack = []
        for name in ("_lazy_init", "is_available", "device_count"):
            mocked = patch.object(torch.cuda, name, side_effect=AssertionError("CUDA called: " + name))
            mocked.start()
            self.stack.append(mocked)

    def tearDown(self):
        for mocked in reversed(self.stack):
            mocked.stop()

    def reject(self):
        with self.assertRaises((ValueError, TypeError, RuntimeError)):
            validate_loaded(self.saved, self.evidence)

    def test_valid_trusted_cpu_payload(self):
        validate_loaded(self.saved, self.evidence)

    def test_state_hash_matches_original_module_hash(self):
        source = (Path(__file__).resolve().parent.parent / "joint_recovery_source" /
                  "region_memory_joint_train_v1" / "common.py")
        if os.environ.get('JOINT_ORIGINAL_TOOL'):
            source = Path(os.environ['JOINT_ORIGINAL_TOOL']) / 'common.py'
        elif not source.is_file():
            source = Path('/root/shared-nvme/BRIGHT-master/New/joint_train_ZAtNI2Wi/region_memory_joint_train_v1/common.py')
        if not source.is_file():
            self.skipTest("Original source available only in the review workspace")
        specification = importlib.util.spec_from_file_location("joint_hash_reference", source)
        reference = importlib.util.module_from_spec(specification)
        import sys
        sys.modules[specification.name] = reference
        try:
            specification.loader.exec_module(reference)
            module = torch.nn.Linear(3, 4)
            self.assertEqual(state_hash(module.state_dict()), reference.module_hash(module))
        finally:
            sys.modules.pop(specification.name, None)

    def test_state_hash_binds_order_names_shape_dtype_and_values(self):
        state = self.saved["heads"]["r0"]
        expected = state_hash(state)
        variations = [
            OrderedDict(reversed(list(state.items()))),
            OrderedDict(other_weight=state["weight"], bias=state["bias"]),
            OrderedDict(weight=state["weight"].reshape(3, 4), bias=state["bias"]),
            OrderedDict(weight=state["weight"].double(), bias=state["bias"]),
            OrderedDict(weight=state["weight"] + 1, bias=state["bias"]),
        ]
        for changed in variations:
            self.assertNotEqual(state_hash(changed), expected)

    def test_config_change_is_rejected(self):
        for key, changed in (("seed", 1), ("workers", 0), ("topk", 8),
                             ("lr", 2e-4), ("eval_every", 1000), ("repo", "/other")):
            with self.subTest(key=key):
                saved = copy.deepcopy(self.saved)
                saved["config"][key] = changed
                with self.assertRaises(ValueError):
                    validate_loaded(saved, self.evidence)

    def test_source_or_protocol_change_is_rejected(self):
        self.saved["source_hash"] = "e" * 64
        self.reject()
        self.saved["source_hash"] = self.evidence["original_source_hash"]
        self.saved["protocol_info"]["paired_bank_sha256"] = "f" * 64
        self.reject()

    def test_completed_or_smoke_checkpoint_rejected(self):
        self.saved["training_complete"] = True
        self.reject()
        self.saved["training_complete"] = False
        self.saved["smoke"] = True
        self.reject()

    def test_unequal_updates_or_step_rejected(self):
        self.saved["updates_per_arm"]["rf"] = 1499
        self.reject()
        self.saved["updates_per_arm"]["rf"] = 1500
        self.saved["step"] = 1499
        self.reject()

    def test_missing_head_or_optimizer_rejected(self):
        self.saved["heads"].pop("rt")
        self.reject()
        self.saved, self.evidence = fixture()
        self.saved["optimizers"].pop("rf")
        self.reject()

    def test_head_tampering_rejected(self):
        self.saved["heads"]["rt"]["weight"][0, 0] += 1
        self.reject()

    def test_nonfinite_head_rejected_even_with_matching_hash(self):
        self.saved["heads"]["rf"]["bias"][0] = torch.nan
        self.evidence["sidecar"]["head_hashes"]["rf"] = state_hash(self.saved["heads"]["rf"])
        self.reject()

    def test_empty_or_nonfinite_optimizer_rejected(self):
        self.saved["optimizers"]["rt"]["state"] = {}
        self.reject()
        self.saved, self.evidence = fixture()
        self.saved["optimizers"]["rt"]["state"][0]["exp_avg"][0, 0] = torch.inf
        self.reject()

    def test_optimizer_step_cannot_exceed_saved_update(self):
        self.saved["optimizers"]["rt"]["state"][0]["step"] = torch.tensor(1501.)
        self.reject()

    def test_initial_hash_binding_change_rejected(self):
        self.saved["initial_head_hashes"]["rt"] = "2" * 64
        self.reject()

    def test_optimizer_missing_moment_or_group_reference_rejected(self):
        self.saved["optimizers"]["rf"]["state"][0].pop("exp_avg_sq")
        self.reject()
        self.saved, self.evidence = fixture()
        self.saved["optimizers"]["rf"]["param_groups"][0]["params"] = [0, 99]
        self.reject()

    def test_missing_rng_rejected(self):
        self.saved["rng"].pop("torch_cuda")
        self.reject()

    def test_invalid_python_rng_rejected_without_global_rng_mutation(self):
        before = random.getstate()
        self.saved["rng"]["python"] = ("invalid",)
        self.reject()
        self.assertEqual(before, random.getstate())

    def test_invalid_numpy_rng_rejected(self):
        self.saved["rng"]["numpy"] = ("invalid",)
        self.reject()

    def test_invalid_torch_cpu_rng_rejected(self):
        self.saved["rng"]["torch_cpu"] = torch.ones(8, dtype=torch.float32)
        self.reject()

    def test_invalid_cuda_rng_schema_rejected_without_cuda(self):
        self.saved["rng"]["torch_cuda"] = [torch.ones(8, dtype=torch.float32)]
        self.reject()

    def test_missing_ledger_rejected(self):
        self.saved["input_file_sha256"] = {}
        self.reject()

    def test_malformed_ledger_rejected(self):
        self.saved["input_file_sha256"] = {"/dataset/train/a.tif": "not-a-sha256"}
        self.reject()


if __name__ == "__main__":
    unittest.main()
