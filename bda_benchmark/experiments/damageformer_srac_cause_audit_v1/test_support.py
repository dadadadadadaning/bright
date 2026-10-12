"""CPU tests of diagnostic provenance, immutable snapshots, and logged evidence."""
from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from . import logs, snapshot, support


CONFIG = dict(dim=4, decoder_width=4, stride=2, iterations=1, temperature=0.2,
              spatial_weight=0.1, building_weight=0.5, region_context=False)


def payload():
    return dict(
        format=support.FRONT_FORMAT,
        source_hash="fixture-source-hash",
        a1_sha256=support.KNOWN_A1,
        a1_checkpoint="unused-a1.pth",
        arguments=dict(steps=50000, smoke=False, **CONFIG),
        config=dict(CONFIG),
        adapter={"region_classifier.weight": torch.ones(4, 4), "region_classifier.bias": torch.zeros(4)},
        step=1000,
        valid_miou=0.5,
        smoke=False,
        test_accessed=False,
    )


class TinyFront(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.base = nn.Linear(4, 4)
        self.region_classifier = nn.Linear(4, 4)

    def adapter_state_dict(self):
        return {name: value for name, value in self.state_dict().items() if not name.startswith("base.")}

    def load_adapter_state_dict(self, state):
        self.load_state_dict(state, strict=False)


class SnapshotSupportTests(unittest.TestCase):
    def test_intermediate_allowed_without_weakening_parent_gate(self):
        from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import load_front
        item = payload()
        support.validate_snapshot_payload(item, expected_source_hash="fixture-source-hash")
        self.assertNotIn("run_complete", item)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "front.pth"
            torch.save(item, path)
            with self.assertRaisesRegex(ValueError, "did not complete"):
                load_front(path, "cpu")

    def test_changed_source_and_a1_hash_rejected(self):
        item = payload()
        with self.assertRaisesRegex(ValueError, "Source changed"):
            support.validate_snapshot_payload(item, expected_source_hash="other-source")
        item["a1_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "audited A1"):
            support.validate_snapshot_payload(item, expected_source_hash="fixture-source-hash")

    def test_config_keys_and_completion_consistency_strict(self):
        item = payload()
        item["config"].pop("stride")
        with self.assertRaisesRegex(ValueError, "config keys"):
            support.validate_snapshot_payload(item, expected_source_hash="fixture-source-hash")
        item = payload()
        item.update(run_complete=True, completed_steps=1000)
        with self.assertRaisesRegex(ValueError, "completed_steps"):
            support.validate_snapshot_payload(item, expected_source_hash="fixture-source-hash")

    def test_snapshot_copy_is_exact_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(support, "source_hash", return_value="fixture-source-hash"):
            root = Path(directory)
            source, output = root / "best.pth", root / "diagnostic"
            torch.save(payload(), source)
            source_bytes = source.read_bytes()
            manifest = snapshot.create_snapshot(source, output)
            self.assertEqual((output / "front_snapshot.pth").read_bytes(), source_bytes)
            self.assertEqual(manifest["snapshot_sha256"], support.digest(source))
            self.assertFalse(manifest["run_complete"])
            self.assertFalse(manifest["a1_copied"])
            self.assertEqual({p.name for p in output.iterdir()}, {"front_snapshot.pth", "snapshot.json"})
            with self.assertRaises(FileExistsError):
                snapshot.create_snapshot(source, output)
            self.assertEqual(source.read_bytes(), source_bytes)

    def test_corrupt_snapshot_never_publishes_success_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "best.pth", root / "diagnostic"
            source.write_bytes(b"incomplete checkpoint")
            with self.assertRaises(Exception):
                snapshot.create_snapshot(source, output)
            self.assertFalse((output / "snapshot.json").exists())
            self.assertFalse((output / "front_snapshot.pth").exists())

    def test_loader_fully_freezes_model_and_preserves_incomplete_status(self):
        fake_module = types.SimpleNamespace(SRACTransformerFront=TinyFront)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, a1 = root / "front.pth", root / "a1.pth"
            torch.save(payload(), checkpoint)
            a1.write_bytes(b"fixture-a1")
            with mock.patch.object(support, "source_hash", return_value="fixture-source-hash"), \
                    mock.patch.object(support, "digest", return_value=support.KNOWN_A1), \
                    mock.patch.object(support, "load_a1_strict"), \
                    mock.patch.dict("sys.modules", {"bda_benchmark.experiments.damageformer_srac_transformer_v1.model": fake_module}):
                model, loaded = support.load_snapshot(checkpoint, a1, "cpu")
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertTrue(all(not m.training for m in model.modules()))
            self.assertNotIn("run_complete", loaded)
            torch.testing.assert_close(model.region_classifier.weight, torch.ones(4, 4))

    def test_snapshot_manifest_rejects_replaced_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(support, "source_hash", return_value="fixture-source-hash"):
            root = Path(directory)
            source, output = root / "best.pth", root / "diagnostic"
            torch.save(payload(), source)
            snapshot.create_snapshot(source, output)
            changed = payload()
            changed["step"] = 2000
            torch.save(changed, output / "front_snapshot.pth")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                support.load_snapshot(output / "front_snapshot.pth", root / "missing-a1.pth", "cpu")

    def test_fingerprint_covers_buffers_and_bfloat16_without_mutation(self):
        module = nn.Linear(2, 2).to(torch.bfloat16)
        module.register_buffer("diagnostic_counter", torch.tensor(0), persistent=False)
        before = support.module_fingerprint(module)
        self.assertEqual(before, support.module_fingerprint(module))
        with torch.no_grad():
            module.diagnostic_counter.add_(1)
        self.assertNotEqual(before, support.module_fingerprint(module))


class LogTests(unittest.TestCase):
    def write_log(self, directory, content):
        path = Path(directory) / "validation.jsonl"
        path.write_bytes(content)
        return path

    def test_only_unterminated_partial_last_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_log(directory, b'{"step":1}\n{"step":')
            self.assertEqual(logs.read_validation_jsonl(path), ([{"step": 1}], True))
            path.write_bytes(b'{"step":1}\n{"step":2}')
            self.assertEqual(logs.read_validation_jsonl(path), ([{"step": 1}, {"step": 2}], False))
            path.write_bytes(b'{"step":1}\n{"step":\n')
            with self.assertRaisesRegex(ValueError, "line 2"):
                logs.read_validation_jsonl(path)
            path.write_bytes(b'{"step":\n{"step":2}')
            with self.assertRaisesRegex(ValueError, "line 1"):
                logs.read_validation_jsonl(path)
            path.write_bytes(b'{"step":1}\n{"step":BROKEN')
            with self.assertRaisesRegex(ValueError, "line 2"):
                logs.read_validation_jsonl(path)

    def test_partial_tokens_from_append_writer_are_tolerated(self):
        encoded = json.dumps(dict(step=1000, test_accessed=False, value=1e-20, note="灾害"), ensure_ascii=True).encode()
        with tempfile.TemporaryDirectory() as directory:
            for length in range(1, len(encoded)):
                path = self.write_log(directory, encoded[:length])
                with self.subTest(length=length):
                    self.assertEqual(logs.read_validation_jsonl(path), ([], True))

    @staticmethod
    def scores(confusion):
        support_counts = [sum(row) for row in confusion]
        predicted = [sum(row[i] for row in confusion) for i in range(4)]
        iou = [confusion[i][i] / max(1, support_counts[i] + predicted[i] - confusion[i][i]) for i in range(4)]
        return dict(confusion=confusion, count=sum(support_counts), valid_miou=sum(iou[1:]) / 3)

    def test_class_counts_prf1_and_damaged_to_intact_are_explicit(self):
        confusion = [[10, 0, 0, 0], [0, 5, 1, 0], [0, 3, 4, 1], [0, 0, 2, 7]]
        result = logs.class_metrics(self.scores(confusion))
        damaged = result["classes"]["Damaged"]
        self.assertEqual(result["confusion"], confusion)
        self.assertEqual(result["damaged_to_intact"], dict(count=3, damaged_support=8, rate=3 / 8))
        self.assertAlmostEqual(damaged["precision"], 4 / 7)
        self.assertAlmostEqual(damaged["recall"], 4 / 8)
        self.assertAlmostEqual(damaged["f1"], 8 / 15)
        self.assertAlmostEqual(damaged["iou"], 4 / 11)

    def test_formal_val450_smoke_label_and_best_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(format=logs.FRONT_FORMAT, test_accessed=False, smoke=False,
                            arguments=dict(steps=50000, val_list="/data/val_set.txt"))
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            good = self.scores([[10, 0, 0, 0], [0, 8, 2, 0], [0, 2, 8, 0], [0, 0, 0, 10]])
            worse = self.scores([[10, 0, 0, 0], [0, 6, 4, 0], [0, 4, 6, 0], [0, 0, 0, 10]])
            rows = [dict(step=step, image_count=450, test_accessed=False,
                         metrics=dict(front_region=score, a1_pixel=good))
                    for step, score in ((1000, good), (2000, worse))]
            self.write_log(root, ("\n".join(json.dumps(row) for row in rows) + "\n").encode())
            result = logs.summarize_logs(root)
            self.assertEqual(result["best"]["step"], 1000)
            self.assertEqual(result["latest"]["step"], 2000)
            self.assertIn("not paired-pixel harm", result["limitations"])
            rows[0]["image_count"] = 2
            self.write_log(root, (json.dumps(rows[0]) + "\n").encode())
            with self.assertRaisesRegex(ValueError, "450"):
                logs.summarize_logs(root)
            manifest["smoke"] = True
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Smoke logs"):
                logs.summarize_logs(root)
            result = logs.summarize_logs(root, allow_smoke=True)
            self.assertFalse(result["formal_val450"])
            self.assertIn("smoke", result["scope"])


if __name__ == "__main__":
    unittest.main()
