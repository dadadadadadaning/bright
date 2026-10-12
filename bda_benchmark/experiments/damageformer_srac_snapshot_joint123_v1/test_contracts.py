"""CPU provenance and protocol tests; no BRIGHT images or CUDA required."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from bda_benchmark.experiments.damageformer_srac_cause_audit_v1 import support
from bda_benchmark.experiments.damageformer_srac_transformer_v1 import common as parent
from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import RetrievalProjector
from . import common


def payload():
    config = dict(dim=128, decoder_width=64, stride=8, iterations=3,
                  temperature=0.2, spatial_weight=0.1, building_weight=0.5, region_context=False)
    return dict(format=parent.FRONT_FORMAT, source_hash="parent-source", a1_sha256=support.KNOWN_A1,
                test_accessed=False, valid_miou=0.614220, step=13000, smoke=False,
                arguments=dict(steps=50000, smoke=False, **config), config=config,
                adapter={"test.weight": torch.ones(2)}, train_sha256="train", val_sha256="val",
                load_report={"expected_valid_miou": 0.616867})


class SnapshotContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.front = self.root / "snapshot" / "front_snapshot.pth"
        self.front.parent.mkdir()
        torch.save(payload(), self.front)
        self.meta = dict(format="SRAC_CAUSE_DIAGNOSTIC_SNAPSHOT_V1", complete=True,
                         diagnostic_only=True, snapshot_file=self.front.name,
                         snapshot_sha256=common.digest(self.front), smoke=False, test_accessed=False,
                         step=13000, declared_steps=50000, run_complete=False,
                         a1_sha256=support.KNOWN_A1, source_hash="parent-source")
        common.save_json(self.front.parent / "snapshot.json", self.meta)

    def test_incomplete_parent_is_not_relabelled(self):
        data = payload()
        before = copy.deepcopy(data)
        with mock.patch.object(support, "source_hash", return_value="parent-source"):
            common.validate_front_payload(data, "train", "val")
        self.assertNotIn("run_complete", data)
        self.assertEqual(data["arguments"], before["arguments"])
        self.assertEqual(data["step"], 13000)
        self.assertEqual(data["arguments"]["steps"], 50000)
        with self.assertRaisesRegex(ValueError, "did not complete"):
            parent.load_front(self.front, "cpu", "unused-a1.pth")

    def test_payload_rejects_smoke_test_and_split_mismatches(self):
        for changes in ({"smoke": True}, {"test_accessed": True}, {"source_hash": "wrong"}):
            with self.subTest(changes=changes), mock.patch.object(support, "source_hash", return_value="parent-source"):
                with self.assertRaises(ValueError):
                    common.validate_front_payload(dict(payload(), **changes), "train", "val")
        with mock.patch.object(support, "source_hash", return_value="parent-source"):
            with self.assertRaisesRegex(ValueError, "split"):
                common.validate_front_payload(payload(), "different", "val")

    def test_manifest_required_and_hash_checked(self):
        self.assertEqual(common.snapshot_metadata(self.front)["step"], 13000)
        common.save_json(self.front.parent / "snapshot.json", dict(self.meta, snapshot_sha256="wrong"))
        with self.assertRaises(ValueError):
            common.snapshot_metadata(self.front)
        with self.assertRaises(ValueError):
            common.snapshot_metadata(self.root / "live" / "best.pth")

    def test_loader_keeps_front_frozen_and_metadata_truthful(self):
        data = payload()
        frozen = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        with mock.patch.object(common, "load_snapshot", return_value=(frozen, data)), \
                mock.patch.object(support, "source_hash", return_value="parent-source"):
            model, loaded = common.load_front(self.front, "cpu", "a1.pth")
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            self.assertIs(loaded, data)
            self.assertNotIn("run_complete", loaded)
            common.save_json(self.front.parent / "snapshot.json", dict(self.meta, step=20000))
            with self.assertRaisesRegex(ValueError, "metadata conflicts"):
                common.load_front(self.front, "cpu", "a1.pth")

    def make_cache(self):
        root = self.root / "bank"
        root.mkdir()
        bank = dict(tokens=torch.ones(4, 8), distribution=torch.eye(4),
                    checkpoint_sha256=common.digest(self.front))
        torch.save(bank, root / "bank.pt")
        manifest = dict(format=common.MEMORY_FORMAT, snapshot_protocol=True, complete=True,
                        split="train", test_accessed=False, source_weights_and_buffers_unchanged=True,
                        checkpoint_sha256=common.digest(self.front), source_hash="experiment",
                        smoke=True, image_count=4, shards=[{}] * 4, parent_checkpoint_step=13000,
                        bank_sha256=common.digest(root / "bank.pt"))
        common.save_json(root / "manifest.json", manifest)
        return root, manifest

    def test_cache_does_not_accept_original_format_or_false_freeze(self):
        root, manifest = self.make_cache()
        with mock.patch.object(common, "source_hash", return_value="experiment"):
            self.assertEqual(len(common.read_cache(root, self.front)[2]["tokens"]), 4)
            for changes in ({"format": "SRAC_TRANSFORMER_MEMORY_V1"}, {"complete": False},
                            {"source_weights_and_buffers_unchanged": False}, {"image_count": 3},
                            {"checkpoint_sha256": "wrong"}, {"parent_checkpoint_step": 20000},
                            {"test_accessed": True}):
                with self.subTest(changes=changes):
                    common.save_json(root / "manifest.json", dict(manifest, **changes))
                    with self.assertRaises(ValueError):
                        common.read_cache(root, self.front)

    def test_metric_binding_and_rng_preservation(self):
        projector = RetrievalProjector(8)
        data = dict(format=common.METRIC_FORMAT, complete=True, arguments={"steps": 2000},
                    step=2000, additional_optimizer_updates=2000, smoke=False, train_only=True,
                    test_accessed=False, source_hash="experiment", front_sha256=common.digest(self.front),
                    bank_sha256="bank", dim=8, model=projector.state_dict())
        path = self.root / "metric.pth"
        torch.save(data, path)
        cache = dict(bank_sha256="bank", smoke=False)
        with mock.patch.object(common, "source_hash", return_value="experiment"):
            rng = torch.get_rng_state().clone()
            loaded, _ = common.load_metric(path, cache, self.front, "cpu")
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            with self.assertRaises(ValueError):
                common.load_metric(path, dict(cache, bank_sha256="other"), self.front, "cpu")
            torch.save(dict(data, step=1999), path)
            with self.assertRaises(ValueError):
                common.load_metric(path, cache, self.front, "cpu")

    def test_reproduction_gate_distinguishes_probe_and_full_val(self):
        result = dict(image_count=450, metrics={"a1_pixel": {"valid_miou": 0.616867},
                                              "direct_region": {"valid_miou": 0.614220}})
        self.assertTrue(common.reproduction_gate(result, payload())["passed"])
        self.assertIsNone(common.reproduction_gate(result, payload(), True)["passed"])
        result["image_count"] = 2
        self.assertFalse(common.reproduction_gate(result, payload())["passed"])
        result["image_count"] = 450
        result["metrics"]["direct_region"]["valid_miou"] = 0.62
        self.assertFalse(common.reproduction_gate(result, payload())["passed"])


if __name__ == "__main__":
    unittest.main()
