"""CPU contracts only: these tests do not reproduce the historical CUDA fault."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import recovery


def fingerprint(obj):
    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, torch.Tensor):
            item = value.detach().cpu().contiguous()
            digest.update(repr((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                add(key)
                add(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode())
            for item in value:
                add(item)
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")

    add(obj)
    return digest.hexdigest()


STATE = SimpleNamespace(state_fingerprint=fingerprint)


def prefix_fixture():
    manifest = dict(seed=3407, actual_training_names=["a", "b", "c"],
                    plan=dict(variant="read_aux"))
    rows = [dict(step=s, optimizer_step=s, epoch=(s-1)//3,
                 epoch_seed=3407+(s-1)//3, variant="read_aux",
                 image="abc"[(s-1)%3]) for s in range(1, 7)]
    return manifest, rows


class PrefixContracts(unittest.TestCase):
    def test_only_checkpoint_prefix_is_inherited(self):
        manifest, rows = prefix_fixture()
        before = copy.deepcopy(rows)
        result = recovery.validate_prefix(rows, manifest, 4)
        self.assertEqual(result, rows[:4])
        self.assertEqual(rows, before)
        self.assertEqual(result[-1]["step"], 4)

    def test_prefix_rejects_missing_or_wrong_update_identity(self):
        manifest, rows = prefix_fixture()
        for field, value in (("step", 6), ("optimizer_step", 1),
                             ("epoch_seed", 99), ("variant", "original"),
                             ("image", "test_image")):
            broken = copy.deepcopy(rows)
            broken[2][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.validate_prefix(broken, manifest, 4)
        with self.assertRaises(ValueError):
            recovery.validate_prefix(rows[:3], manifest, 4)

    def test_reconstruction_returns_next_batch_without_preparing_images(self):
        manifest, rows = prefix_fixture()
        calls = []

        def batches(runtime, names, seed):
            self.assertEqual(names, manifest["actual_training_names"])
            self.assertEqual(seed, manifest["seed"])
            for i, row in enumerate(rows):
                calls.append(i + 1)
                yield row["epoch"], (None, [row["image"]])

        runtime = SimpleNamespace(prepare=lambda *a: self.fail("reconstruction prepared image"))
        iterator = recovery.reconstruct_iterator(SimpleNamespace(batches=batches), runtime,
                                                  manifest, rows, 4)
        self.assertEqual(calls, [1, 2, 3, 4])
        epoch, batch = next(iterator)
        self.assertEqual((epoch, batch[-1][0]), (1, "b"))

    def test_reconstruction_rejects_reordered_or_wrong_epoch_data(self):
        manifest, rows = prefix_fixture()
        for mutation in ("image", "epoch"):
            def batches(runtime, names, seed):
                for i, row in enumerate(rows):
                    image = "b" if i == 0 and mutation == "image" else row["image"]
                    epoch = 1 if i == 0 and mutation == "epoch" else row["epoch"]
                    yield epoch, (None, [image])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                recovery.reconstruct_iterator(SimpleNamespace(batches=batches), object(),
                                                manifest, rows, 4)


class RestoreContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def saved(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.005)
        model(torch.ones(1, 3)).square().sum().backward()
        optimizer.step()
        for value in optimizer.state.values():
            value["step"].fill_(800)
        return copy.deepcopy(dict(step=800, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  rng=torch.get_rng_state()))

    def fresh(self):
        model = torch.nn.Linear(3, 2)
        return model, torch.optim.AdamW(model.parameters(), lr=.7)

    def test_restores_exact_model_optimizer_and_preserves_payload(self):
        saved = self.saved()
        before = fingerprint(saved)
        model, optimizer = self.fresh()
        recovery.restore_model_optimizer(torch, STATE, model, optimizer, saved, 800)
        self.assertEqual(fingerprint(model.state_dict()), fingerprint(saved["model"]))
        self.assertEqual(fingerprint(optimizer.state_dict()), fingerprint(saved["optimizer"]))
        self.assertEqual(fingerprint(saved), before)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-5)

    def test_rejects_partial_or_empty_optimizer_state(self):
        for empty in (False, True):
            saved = self.saved()
            if empty:
                saved["optimizer"]["state"] = {}
            else:
                next(iter(saved["optimizer"]["state"].values()))["step"].fill_(799)
            model, optimizer = self.fresh()
            with self.subTest(empty=empty), self.assertRaises(ValueError):
                recovery.restore_model_optimizer(torch, STATE, model, optimizer, saved, 800)

    def test_rejects_model_shape_mismatch(self):
        saved = self.saved()
        saved["model"]["weight"] = torch.zeros(2, 4)
        model, optimizer = self.fresh()
        with self.assertRaises((ValueError, RuntimeError)):
            recovery.restore_model_optimizer(torch, STATE, model, optimizer, saved, 800)

    def test_rejects_missing_rng_or_nonfinite_model(self):
        for field in ("rng", "nonfinite"):
            saved = self.saved()
            if field == "rng":
                del saved["rng"]
            else:
                saved["model"]["weight"][0, 0] = float("nan")
            model, optimizer = self.fresh()
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.restore_model_optimizer(torch, STATE, model, optimizer, saved, 800)

    def test_rejects_silent_optimizer_conversion(self):
        saved = self.saved()
        model, optimizer = self.fresh()
        load = optimizer.load_state_dict

        def corrupt(payload):
            load(payload)
            optimizer.param_groups[0]["lr"] *= 2

        optimizer.load_state_dict = corrupt
        with self.assertRaises(ValueError):
            recovery.restore_model_optimizer(torch, STATE, model, optimizer, saved, 800)


class ParentPreflightContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.package, self.parent = root / "package", root / "parent"
        self.package.mkdir()
        self.parent.mkdir()
        (self.parent / "checkpoints").mkdir()
        for name in recovery.IDENTITY:
            (self.package / name).write_text("MARKER = 'verified fixture'\n")
        self.identity = {name: recovery.sha(self.package / name) for name in recovery.IDENTITY}
        self.checkpoint = self.parent / "checkpoints/checkpoint_000800.pt"
        self.checkpoint.write_bytes(b"not a pickle: preflight must never torch.load\n")
        self.checksum = recovery.sha(self.checkpoint)
        self.plan = dict(format=recovery.FORMAT, variant="read_aux", effective_mode="read_aux",
                         seed=3407, endpoint_step=2500, read_weight=.01, extension_files=self.identity,
                         scheduler_horizon=5000, maximum_lr=.0001, weight_decay=.005,
                         warmup_steps=500, grad_clip=1., optimizer="AdamW", support_arm="d",
                         post500_lr_multiplier=.2, test_accessed=False)
        names = ["train_" + str(i) for i in range(2918)]
        self.manifest = dict(plan=self.plan, seed=3407, actual_training_names=names,
                             train_names=names, val_names=["val_" + str(i) for i in range(450)])
        self.status = dict(phase="failed", last_complete_step=803, failure_phase="update_804")
        self.sidecar = dict(step=800, sha256=self.checksum, roundtrip_verified=True)
        self.rows = [dict(step=s, optimizer_step=s, epoch=0, epoch_seed=3407,
                          variant="read_aux", image=names[s-1]) for s in range(1, 804)]
        self.save_fixture()
        old_path = sys.path[:]
        self.addCleanup(lambda: setattr(sys, "path", old_path))
        for name, value in (("IDENTITY", self.identity), ("CHECKPOINT_SHA", self.checksum)):
            guard = patch.object(recovery, name, value)
            guard.start()
            self.addCleanup(guard.stop)

    def save_fixture(self):
        for name, value in (("plan.json", self.plan), ("manifest.json", self.manifest),
                            ("summary.json", self.status)):
            (self.parent / name).write_text(json.dumps(value))
        self.checkpoint.with_suffix(".pt.json").write_text(json.dumps(self.sidecar))
        (self.parent / "train.jsonl").write_text("".join(json.dumps(r)+"\n" for r in self.rows))

    def test_valid_preflight_reads_only_and_retains_separate_fingerprints(self):
        before = {p: p.read_bytes() for p in self.parent.rglob("*") if p.is_file()}
        with patch.object(torch, "load") as load, patch.object(recovery.subprocess, "Popen") as launch:
            exp, plan, manifest, rows, checkpoint, digests = recovery.read_parent(self.package, self.parent)
        load.assert_not_called()
        launch.assert_not_called()
        self.assertEqual(exp.MARKER, "verified fixture")
        self.assertEqual(plan, self.plan)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(len(rows), 803)
        self.assertEqual(checkpoint, self.checkpoint)
        self.assertEqual(digests[str(checkpoint)], self.checksum)
        self.assertIn(str(self.parent / "train.jsonl"), digests)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def assert_pre_import_rejection(self):
        with patch.object(torch, "load") as load, patch.object(recovery.importlib.util, "spec_from_file_location") as importer:
            with self.assertRaises(ValueError):
                recovery.read_parent(self.package, self.parent)
        load.assert_not_called()
        importer.assert_not_called()

    def test_modified_checkpoint_rejected_before_loading(self):
        self.checkpoint.write_bytes(b"different checkpoint")
        self.assert_pre_import_rejection()

    def test_modified_code_rejected_before_import(self):
        (self.package / "experiment.py").write_text("raise RuntimeError('must not execute')\n")
        self.assert_pre_import_rejection()

    def test_wrong_recipe_sidecar_and_trajectory_rejected(self):
        cases = ("recipe", "sidecar", "status", "journal")
        for case in cases:
            with self.subTest(case=case):
                original = copy.deepcopy((self.plan, self.manifest, self.status, self.sidecar, self.rows))
                if case == "recipe":
                    self.plan["read_weight"] = .05
                elif case == "sidecar":
                    self.sidecar["roundtrip_verified"] = False
                elif case == "status":
                    self.status["last_complete_step"] = 804
                else:
                    self.rows[400]["optimizer_step"] = 400
                self.save_fixture()
                self.assert_pre_import_rejection()
                self.plan, self.manifest, self.status, self.sidecar, self.rows = original
                self.save_fixture()


class ProtocolContracts(unittest.TestCase):
    def protocol(self):
        return dict(execution_settings=dict(cuda_launch_blocking=None, cudnn_deterministic=True,
                                             cudnn_benchmark=False), train_names=["a", "b"], val_names=["v"])

    def test_allows_only_recorded_launch_blocking_change_without_mutating_inputs(self):
        before = self.protocol()
        after = copy.deepcopy(before)
        after["execution_settings"]["cuda_launch_blocking"] = "1"
        old = copy.deepcopy((before, after))
        delta = recovery.protocol_delta(before, after)
        self.assertEqual(delta, {"execution_settings.cuda_launch_blocking": {"before": None, "after": "1"}})
        self.assertEqual((before, after), old)

    def test_denies_backend_or_membership_changes(self):
        before = self.protocol()
        for field in ("cudnn_benchmark", "train_names", "val_names", "cuda_launch_blocking"):
            after = copy.deepcopy(before)
            after["execution_settings"]["cuda_launch_blocking"] = "1"
            if field == "cudnn_benchmark":
                after["execution_settings"][field] = True
            elif field == "cuda_launch_blocking":
                after["execution_settings"][field] = "0"
            else:
                after[field] = ["different_image"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.protocol_delta(before, after)


if __name__ == "__main__":
    unittest.main()
