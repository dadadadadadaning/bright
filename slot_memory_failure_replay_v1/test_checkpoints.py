"""CPU checkpoint evidence tests; no BRIGHT data, CUDA or formal run claims."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

import checkpoints
from inspect_failure import sha256


class State:
    """Small device-independent state helper with all production RNG fields."""

    @staticmethod
    def capture_rng():
        return {
            "python": copy.deepcopy(random.getstate()),
            "numpy": copy.deepcopy(np.random.get_state()),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": ([value.clone() for value in torch.cuda.get_rng_state_all()]
                           if torch.cuda.is_available() else []),
        }

    @staticmethod
    def restore_rng(value):
        random.setstate(copy.deepcopy(value["python"]))
        np.random.set_state(copy.deepcopy(value["numpy"]))
        torch.set_rng_state(value["torch_cpu"])
        if value["torch_cuda"]:
            torch.cuda.set_rng_state_all(value["torch_cuda"])

    @staticmethod
    def state_fingerprint(value):
        checksum = hashlib.sha256()

        def add(item):
            if torch.is_tensor(item):
                value = item.detach().cpu().contiguous()
                checksum.update(b"tensor")
                add(str(value.dtype))
                add(tuple(value.shape))
                checksum.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(item, np.ndarray):
                checksum.update(b"ndarray")
                add(str(item.dtype))
                add(tuple(item.shape))
                checksum.update(item.tobytes())
            elif isinstance(item, dict):
                checksum.update(b"dict")
                for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                    add(key)
                    add(item[key])
            elif isinstance(item, (tuple, list)):
                checksum.update(type(item).__name__.encode())
                add(len(item))
                for part in item:
                    add(part)
            elif item is None or isinstance(item, (bool, int, float, str, np.generic)):
                checksum.update(type(item).__name__.encode())
                checksum.update(repr(item).encode())
            else:
                raise TypeError("Unsupported test state: " + type(item).__name__)
            checksum.update(b"\0")

        add(value)
        return checksum.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def head():
    return torch.nn.Sequential(
        torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2))


def train_once(heads, optimizers):
    # All three CPU RNG families affect this update. Restoring a checkpoint
    # therefore tests more than equality of bytes immediately after loading.
    features = torch.randn(4, 3) + random.random() + float(np.random.uniform())
    target = torch.tensor([0, 1, 1, 0])
    losses = {}
    for arm, model in heads.items():
        optimizer = optimizers[arm]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(features), target)
        loss.backward()
        optimizer.step()
        losses[arm] = float(loss.detach())
    return losses


class CheckpointContracts(unittest.TestCase):
    def setUp(self):
        self.original_rng = State.capture_rng()
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "checkpoints"
        self.path = self.folder / "diagnostic_000002.pt"
        self.journal = self.root / "replay_train.jsonl"
        self.journal.write_text(
            '{"step":1,"image":"train0","arms":{"e":{"optimizer_step":1},"f":{"optimizer_step":1}}}\n'
            '{"step":2,"image":"train1","arms":{"e":{"optimizer_step":2},"f":{"optimizer_step":2}}}\n')
        self.used = self.root / "used_input.dat"
        self.unused = self.root / "inherited_unvisited_input.dat"
        self.used.write_bytes(b"frozen source observed in this worker")
        self.unused.write_bytes(b"parent input not visited in this bounded worker")
        self.inherited = {str(path): sha256(path) for path in (self.used, self.unused)}
        self.runtime = SimpleNamespace(
            input_ledger={self.used: self.inherited[str(self.used)]},
            protocol_info={"synthetic": True, "test_accessed": False},
            args=SimpleNamespace(seed=1729, device="cpu", output=str(self.root / "worker")))
        self.factory = SimpleNamespace(provenance=lambda: {
            "seed": 1729, "query_GT_used_for_supports": False,
            "diagnostic_fixture": True})
        self.plan = {"arms": ["e", "f"], "seeds": [1729], "endpoint_step": 2500}
        self.manifest = {"seed": 1729, "plan": copy.deepcopy(self.plan),
                         "runtime_config": vars(self.runtime.args).copy()}
        self.lineage = {
            "parent_checkpoint_sha256": "1" * 64,
            "original_journal_sha256": "2" * 64,
            "last_original_complete_step": 2,
        }
        random.seed(1729)
        np.random.seed(1729)
        torch.manual_seed(1729)
        prototype = head()
        self.heads = {arm: copy.deepcopy(prototype) for arm in self.plan["arms"]}
        self.optimizers = {
            arm: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.005)
            for arm, model in self.heads.items()}
        train_once(self.heads, self.optimizers)
        train_once(self.heads, self.optimizers)

    def tearDown(self):
        self.temp.cleanup()
        State.restore_rng(self.original_rng)
        torch.set_num_threads(self.threads)

    def save(self, **kwargs):
        arguments = dict(
            step=2, plan=self.plan, manifest=self.manifest, inherited=self.inherited,
            lineage=self.lineage, journal_path=self.journal, write_json=write_json)
        arguments.update(kwargs)
        return checkpoints.save(
            torch, self.folder, self.heads, self.optimizers, State,
            self.runtime, self.factory, **arguments)

    def load(self, **kwargs):
        arguments = dict(
            plan=self.plan, manifest=self.manifest, lineage=self.lineage,
            journal_path=self.journal, expected_step=2)
        arguments.update(kwargs)
        return checkpoints.load(torch, self.path, State, **arguments)

    def assert_unpublished(self):
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_suffix(".pt.json").exists())
        self.assertFalse(self.path.with_suffix(".pt.tmp").exists())

    def test_complete_pair_roundtrips_head_adamw_and_all_rng(self):
        expected_rng = State.capture_rng()
        expected_heads = State.state_fingerprint(
            {arm: model.state_dict() for arm, model in self.heads.items()})
        expected_optimizers = State.state_fingerprint(
            {arm: optimizer.state_dict() for arm, optimizer in self.optimizers.items()})
        record = self.save()
        saved, reopened_record = self.load()
        self.assertEqual(record, reopened_record)
        self.assertEqual(State.state_fingerprint(saved["heads"]), expected_heads)
        self.assertEqual(State.state_fingerprint(saved["optimizers"]), expected_optimizers)
        self.assertEqual(set(saved["rng"]), {"python", "numpy", "torch_cpu", "torch_cuda"})
        self.assertEqual(State.state_fingerprint(saved["rng"]), State.state_fingerprint(expected_rng))
        self.assertEqual(State.state_fingerprint(State.capture_rng()), State.state_fingerprint(expected_rng))
        self.assertEqual(record["payload_fingerprint"], State.state_fingerprint(saved))
        self.assertEqual(record["sha256"], sha256(self.path))
        self.assertEqual(record["replay_journal_sha256"], sha256(self.journal))
        self.assertEqual(saved["updates_per_arm"], {"e": 2, "f": 2})
        self.assertTrue(saved["diagnostic_only"])
        self.assertTrue(record["roundtrip_verified"])
        self.assertFalse(saved["test_accessed"])
        self.assertFalse(saved["exact_historical_state_claim"])
        for values in saved["optimizers"].values():
            self.assertEqual(len(values["state"]), 4)
            for parameter in values["state"].values():
                self.assertEqual(int(parameter["step"]), 2)
                self.assertGreater(float(parameter["exp_avg"].abs().sum()), 0)
                self.assertGreater(float(parameter["exp_avg_sq"].sum()), 0)

        # Execute a further update, then restore into independently allocated
        # heads/optimizers and require exactly the same next update and RNG.
        expected_losses = train_once(self.heads, self.optimizers)
        expected_next = State.state_fingerprint({
            "heads": {arm: model.state_dict() for arm, model in self.heads.items()},
            "optimizers": {arm: optimizer.state_dict() for arm, optimizer in self.optimizers.items()},
            "rng": State.capture_rng()})
        restored_heads = {arm: head() for arm in saved["heads"]}
        restored_optimizers = {
            arm: torch.optim.AdamW(model.parameters(), lr=9e-3)
            for arm, model in restored_heads.items()}
        for arm, model in restored_heads.items():
            model.load_state_dict(saved["heads"][arm])
            restored_optimizers[arm].load_state_dict(copy.deepcopy(saved["optimizers"][arm]))
        State.restore_rng(saved["rng"])
        actual_losses = train_once(restored_heads, restored_optimizers)
        actual_next = State.state_fingerprint({
            "heads": {arm: model.state_dict() for arm, model in restored_heads.items()},
            "optimizers": {arm: optimizer.state_dict() for arm, optimizer in restored_optimizers.items()},
            "rng": State.capture_rng()})
        self.assertEqual(actual_losses, expected_losses)
        self.assertEqual(actual_next, expected_next)
        self.assertNotEqual(next(restored_heads["e"].parameters()).data_ptr(),
                            next(restored_heads["f"].parameters()).data_ptr())

    def test_preserves_inherited_unvisited_inputs_without_claiming_reverification(self):
        inherited_before = copy.deepcopy(self.inherited)
        self.save()
        saved, _ = self.load()
        self.assertEqual(saved["input_file_sha256"], inherited_before)
        self.assertEqual(self.inherited, inherited_before)
        self.assertEqual(saved["inputs_verified_this_worker"],
                         {str(self.used): inherited_before[str(self.used)]})
        self.assertNotIn(str(self.unused), saved["inputs_verified_this_worker"])
        self.assertFalse(saved["inherited_inputs_all_reverified"])

    def test_current_input_change_or_new_input_is_rejected_before_publication(self):
        for key in (self.used, self.root / "unbound.dat"):
            with self.subTest(path=key):
                self.runtime.input_ledger = {key: "f" * 64}
                with self.assertRaisesRegex(ValueError, "input differs"):
                    self.save()
                self.assert_unpublished()

    def test_missing_input_ledgers_are_rejected_before_publication(self):
        self.runtime.input_ledger = {}
        with self.assertRaisesRegex(ValueError, "Missing input ledger"):
            self.save()
        self.assert_unpublished()

    def test_lagging_arm_counter_is_rejected_before_publication(self):
        parameter = next(iter(self.optimizers["f"].state.values()))
        parameter["step"].fill_(1)
        with self.assertRaisesRegex(ValueError, "partial update"):
            self.save()
        self.assert_unpublished()

    def test_fractional_counter_is_rejected_before_publication(self):
        parameter = next(iter(self.optimizers["f"].state.values()))
        parameter["step"].fill_(2.5)
        with self.assertRaises(ValueError):
            self.save()
        self.assert_unpublished()

    def test_missing_optimizer_arm_is_rejected_before_publication(self):
        del self.optimizers["f"]
        with self.assertRaisesRegex(ValueError, "arms"):
            self.save()
        self.assert_unpublished()

    def test_missing_parameter_state_is_rejected_before_publication(self):
        parameter = next(iter(self.optimizers["f"].state))
        del self.optimizers["f"].state[parameter]
        with self.assertRaisesRegex(ValueError, "parameter states"):
            self.save()
        self.assert_unpublished()

    def test_nonfinite_adam_moment_is_rejected_before_publication(self):
        parameter = next(iter(self.optimizers["f"].state.values()))
        parameter["exp_avg"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Adam moments"):
            self.save()
        self.assert_unpublished()

    def test_existing_artifacts_are_not_overwritten(self):
        for suffix in (".pt", ".pt.json", ".pt.tmp"):
            with self.subTest(suffix=suffix):
                self.folder.mkdir(exist_ok=True)
                occupied = self.path.with_suffix(suffix)
                occupied.write_bytes(b"existing immutable evidence")
                with self.assertRaises(FileExistsError):
                    self.save()
                self.assertEqual(occupied.read_bytes(), b"existing immutable evidence")
                self.assertEqual(set(self.folder.iterdir()), {occupied})
                occupied.unlink()

    def test_binary_tampering_is_rejected_without_deserialization(self):
        self.save()
        with self.path.open("ab") as stream:
            stream.write(b"tampering")
        with self.assertRaisesRegex(ValueError, "file/sidecar mismatch"):
            self.load()

    def test_payload_mutation_still_fails_if_only_file_hash_is_recomputed(self):
        self.save()
        saved = torch.load(self.path, map_location="cpu", weights_only=False)
        saved["heads"]["e"]["0.weight"].add_(1)
        torch.save(saved, self.path)
        sidecar = self.path.with_suffix(".pt.json")
        record = json.loads(sidecar.read_text())
        record["sha256"] = sha256(self.path)
        write_json(sidecar, record)
        with self.assertRaisesRegex(ValueError, "payload fingerprint"):
            self.load()

    def test_requested_lineage_plan_manifest_or_step_mismatch_is_rejected(self):
        self.save()
        changes = {
            "lineage": dict(self.lineage, parent_checkpoint_sha256="f" * 64),
            "plan": dict(self.plan, endpoint_step=3000),
            "manifest": dict(self.manifest, seed=99),
            "expected_step": 3,
        }
        for key, value in changes.items():
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.load(**{key: value})

    def test_sidecar_lineage_and_journal_hash_mutations_are_rejected(self):
        self.save()
        sidecar = self.path.with_suffix(".pt.json")
        original = json.loads(sidecar.read_text())
        for key, value in (
                ("lineage", dict(self.lineage, parent_checkpoint_sha256="f" * 64)),
                ("replay_journal_sha256", "f" * 64)):
            write_json(sidecar, dict(original, **{key: value}))
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.load()
        write_json(sidecar, original)

    def test_replay_journal_mutation_is_rejected(self):
        self.save()
        with self.journal.open("a") as stream:
            stream.write('{"step":3,"image":"different"}\n')
        with self.assertRaisesRegex(ValueError, "file/sidecar|lineage/complete-step"):
            self.load()


if __name__ == "__main__":
    unittest.main()
