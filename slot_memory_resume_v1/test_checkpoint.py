"""CPU contracts for formal continuation checkpoint boundaries and provenance."""
from __future__ import annotations

from collections import OrderedDict
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

import checkpoint
from restore_state import restore_training_state, validate_restored_rng


class CPUState:
    def state_fingerprint(self, value):
        def canonical(obj):
            if torch.is_tensor(obj):
                tensor = obj.detach().cpu().contiguous()
                return ["tensor", str(tensor.dtype), list(tensor.shape),
                        tensor.reshape(-1).view(torch.uint8).numpy().tobytes().hex()]
            if isinstance(obj, np.ndarray):
                return ["numpy", str(obj.dtype), list(obj.shape), obj.tobytes().hex()]
            if isinstance(obj, dict):
                return ["dict", [[canonical(k), canonical(obj[k])] for k in
                                 sorted(obj, key=lambda k: (type(k).__name__, repr(k)))]]
            if isinstance(obj, (tuple, list)):
                return [type(obj).__name__, [canonical(v) for v in obj]]
            if obj is None or isinstance(obj, (bool, str, int, float, np.generic)):
                return [type(obj).__name__, repr(obj)]
            raise TypeError(type(obj).__name__)
        return hashlib.sha256(json.dumps(canonical(value)).encode()).hexdigest()

    def capture_rng(self):
        return {
            "python": copy.deepcopy(random.getstate()), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": [],
        }

    def restore_rng(self, value, *, require_cuda=False):
        if require_cuda or value["torch_cuda"]:
            raise ValueError("CPU fixture only")
        random.setstate(value["python"])
        np.random.set_state(value["numpy"])
        torch.set_rng_state(value["torch_cpu"])


def fresh(seed=29):
    torch.manual_seed(seed)
    prototype = torch.nn.Sequential(
        torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Dropout(0.25),
        torch.nn.Linear(5, 2),
    )
    heads = OrderedDict((a, copy.deepcopy(prototype)) for a in checkpoint.ARMS)
    optimizers = OrderedDict(
        (a, torch.optim.AdamW(h.parameters(), lr=0.003, weight_decay=0.005,
                             amsgrad=True, foreach=False)) for a, h in heads.items()
    )
    return heads, optimizers


def train(heads, optimizers, batches):
    for inputs, targets in batches:
        for arm, head in heads.items():
            optimizers[arm].zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(head(inputs), targets).backward()
            optimizers[arm].step()


class FormalCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "checkpoints"
        self.journal = self.root / "train.jsonl"
        self.state = CPUState()
        self.names = ["synthetic_%d" % n for n in range(1, 8)]
        self.plan = {
            "format": checkpoint.ORIGINAL_FORMAT, "arms": list(checkpoint.ARMS),
            "seeds": [3407], "endpoint_step": 7, "schedule_horizon": 5000,
        }
        self.manifest = {
            "format": checkpoint.ORIGINAL_FORMAT, "plan": copy.deepcopy(self.plan),
            "seed": 3407, "train_names": self.names,
            "protocol_info": {"execution_settings": {"cuda_launch_blocking": None}},
            "runtime_config": {"output": "/original/train", "lr": 0.003},
        }
        self.inherited = {"/synthetic/input": hashlib.sha256(b"input").hexdigest()}
        self.runtime = SimpleNamespace(
            input_ledger={Path(p): h for p, h in self.inherited.items()},
            protocol_info=copy.deepcopy(self.manifest["protocol_info"]),
            args=SimpleNamespace(output=str(self.root), lr=0.003),
        )
        self.factory = SimpleNamespace(provenance=lambda: {"fixture": "CPU"})
        self.lineage = {"original_run": "/synthetic/original", "parent_step": 1}
        generator = torch.Generator().manual_seed(74)
        self.batches = [(torch.randn(4, 3, generator=generator),
                         torch.randn(4, 2, generator=generator)) for _ in range(7)]
        self.heads, self.optimizers = fresh()
        train(self.heads, self.optimizers, self.batches[:3])
        self.rows = [self.row(n) for n in range(1, 4)]
        self.write_journal()

    def row(self, step):
        return {
            "step": step, "epoch": 0, "epoch_seed": 3407,
            "image": self.names[step - 1],
            "arms": {a: {"optimizer_step": step} for a in checkpoint.ARMS},
        }

    @staticmethod
    def write_json(path, value):
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(value) + "\n")
        temporary.replace(path)

    def write_journal(self):
        self.journal.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def save(self):
        return checkpoint.save(
            torch, self.folder, self.heads, self.optimizers, self.state,
            self.runtime, self.factory, step=3, plan=self.plan,
            original_manifest=self.manifest, inherited=self.inherited,
            lineage=self.lineage, journal_path=self.journal, write_json=self.write_json,
        )

    def load(self):
        return checkpoint.load(
            torch, self.folder / "checkpoint_000003.pt", self.state,
            plan=self.plan, original_manifest=self.manifest,
            lineage=self.lineage, journal_path=self.journal,
        )

    def rewrite_payload(self, mutate):
        path = self.folder / "checkpoint_000003.pt"
        value = torch.load(path, map_location="cpu", weights_only=False)
        mutate(value)
        torch.save(value, path)
        record = json.loads(path.with_suffix(".pt.json").read_text())
        record["sha256"] = checkpoint.sha256(path)
        record["payload_fingerprint"] = self.state.state_fingerprint(value)
        record["head_hashes"] = {
            a: self.state.state_fingerprint(value["heads"][a]) for a in checkpoint.ARMS
        }
        self.write_json(path.with_suffix(".pt.json"), record)

    def test_full_state_roundtrip_then_continuation_matches_uninterrupted(self):
        before_rng = self.state.state_fingerprint(self.state.capture_rng())
        record = self.save()
        self.assertEqual(record["format"], checkpoint.FORMAT)
        self.assertEqual(set(record["head_hashes"]), set(checkpoint.ARMS))
        self.assertEqual(record["journal_prefix_bytes"], self.journal.stat().st_size)
        self.assertEqual(record["journal_prefix_sha256"], checkpoint.sha256(self.journal))
        self.assertEqual(record["last_data_position"], {
            "epoch": 0, "epoch_seed": 3407, "step": 3,
            "consumed_in_epoch": 3, "image": "synthetic_3",
        })
        self.assertEqual(self.state.state_fingerprint(self.state.capture_rng()), before_rng)
        saved, loaded_record = self.load()
        self.assertEqual(record, loaded_record)
        self.assertNotIn("diagnostic_only", saved)
        self.assertEqual(saved["input_file_sha256"], self.inherited)
        train(self.heads, self.optimizers, self.batches[3:])
        expected = {
            arm: (self.state.state_fingerprint(head.state_dict()),
                  self.state.state_fingerprint(self.optimizers[arm].state_dict()))
            for arm, head in self.heads.items()
        }
        heads, optimizers = fresh(seed=91)
        restore_training_state(torch, heads, optimizers, saved, self.state)
        validate_restored_rng(self.state, saved["rng"])
        train(heads, optimizers, self.batches[3:])
        for arm, head in heads.items():
            self.assertEqual(self.state.state_fingerprint(head.state_dict()), expected[arm][0])
            self.assertEqual(self.state.state_fingerprint(optimizers[arm].state_dict()), expected[arm][1])

    def test_later_appends_are_accepted_but_checkpoint_prefix_tampering_is_not(self):
        record = self.save()
        self.rows.extend([self.row(4), self.row(5)])
        self.write_journal()
        saved, _ = self.load()
        self.assertEqual(saved["journal_prefix_bytes"], record["journal_prefix_bytes"])
        self.rows[0]["image"] = self.names[5]
        self.write_journal()
        with self.assertRaisesRegex(ValueError, "journal prefix differs"):
            self.load()

    def test_partial_arm_update_never_publishes_checkpoint(self):
        values = next(iter(self.optimizers["b"].state.values()))
        values["step"].fill_(2)
        with self.assertRaisesRegex(ValueError, "partial update"):
            self.save()
        self.assertFalse(self.folder.exists())

    def test_invalid_prefix_or_incomplete_arm_journal_never_publishes(self):
        for mutation in ("unterminated", "missing_arm", "wrong_step", "epoch"):
            with self.subTest(mutation=mutation):
                self.rows = [self.row(n) for n in range(1, 4)]
                if mutation == "missing_arm":
                    self.rows[2]["arms"].pop("l")
                elif mutation == "wrong_step":
                    self.rows[2]["arms"]["d"]["optimizer_step"] = 2
                elif mutation == "epoch":
                    self.rows[2]["epoch"] = 1
                self.write_journal()
                if mutation == "unterminated":
                    self.journal.write_bytes(self.journal.read_bytes().rstrip(b"\n"))
                with self.assertRaises(ValueError):
                    self.save()
                self.assertFalse(self.folder.exists())

    def test_nonfinite_head_or_moment_is_rejected(self):
        for mutation in ("head", "moment"):
            with self.subTest(mutation=mutation):
                self.heads, self.optimizers = fresh()
                train(self.heads, self.optimizers, self.batches[:3])
                if mutation == "head":
                    with torch.no_grad():
                        next(self.heads["e"].parameters()).reshape(-1)[0] = float("nan")
                else:
                    next(iter(self.optimizers["e"].state.values()))["exp_avg_sq"].reshape(-1)[0] = float("nan")
                with self.assertRaisesRegex(ValueError, "nonfinite|Nonfinite"):
                    self.save()
                self.assertFalse(self.folder.exists())

    def test_rejects_file_sha_or_payload_fingerprint_tampering(self):
        self.save()
        path = self.folder / "checkpoint_000003.pt"
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        with self.assertRaisesRegex(ValueError, "file/sidecar mismatch"):
            self.load()
        path.write_bytes(original)
        sidecar = json.loads(path.with_suffix(".pt.json").read_text())
        sidecar["payload_fingerprint"] = "0" * 64
        self.write_json(path.with_suffix(".pt.json"), sidecar)
        with self.assertRaisesRegex(ValueError, "payload fingerprint"):
            self.load()

    def test_diagnostic_format_is_rejected_even_with_refreshed_file_hash(self):
        self.save()
        self.rewrite_payload(lambda value: value.update(
            format="BRIGHT_SLOT_MEMORY_DIAGNOSTIC_CHECKPOINT_V1", diagnostic_only=True))
        with self.assertRaisesRegex(ValueError, "format, lineage"):
            self.load()

    def test_load_revalidates_complete_counters_and_position_after_roundtrip(self):
        self.save()
        self.rewrite_payload(lambda value: next(iter(
            value["optimizers"]["f"]["state"].values()))["step"].fill_(2))
        with self.assertRaisesRegex(ValueError, "partial update"):
            self.load()

    def test_original_protocol_endpoint_and_input_scope_cannot_change(self):
        for mutation in ("protocol", "endpoint", "input"):
            with self.subTest(mutation=mutation):
                self.runtime.protocol_info = copy.deepcopy(self.manifest["protocol_info"])
                self.runtime.input_ledger = {Path(p): h for p, h in self.inherited.items()}
                self.plan["endpoint_step"] = 7
                if mutation == "protocol":
                    self.runtime.protocol_info["execution_settings"]["cuda_launch_blocking"] = "1"
                elif mutation == "endpoint":
                    self.plan["endpoint_step"] = 6
                else:
                    self.runtime.input_ledger[Path("/unrecorded/input")] = "a" * 64
                with self.assertRaises(ValueError):
                    self.save()
                self.assertFalse(self.folder.exists())

    def test_existing_checkpoint_or_sidecar_is_never_overwritten(self):
        self.save()
        before = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        with self.assertRaises(FileExistsError):
            self.save()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.folder.iterdir()})
        checkpoint_path = self.folder / "checkpoint_000003.pt"
        checkpoint_path.unlink()
        sidecar = checkpoint_path.with_suffix(".pt.json")
        sidecar_before = sidecar.read_bytes()
        with self.assertRaises(FileExistsError):
            self.save()
        self.assertFalse(checkpoint_path.exists())
        self.assertEqual(sidecar.read_bytes(), sidecar_before)


if __name__ == "__main__":
    unittest.main()
