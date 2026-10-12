"""CPU-only checks that incomplete Val evidence cannot be reported as formal."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
from bda_benchmark.experiments.damageformer_srac_memory_metric_v2.train import (
    load_metric_checkpoint)

from .evaluate import ALL_ARMS, CONTROL_ARMS, FORMAT, METRIC_ARMS, PAIRS
from .native import NativeA1LogitsCapture, native_grid_probq, protocol_record
from .report import read_formal


class FormalReportTests(unittest.TestCase):
    def test_native_a1_capture_uses_original_forward_and_q_probability(self):
        import torch
        from torch import nn
        from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture

        class Front(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Module()
                self.base.clf_2 = nn.Conv2d(2, 4, 1)

            def forward(self, pre, post):
                native = self.base.clf_2(post)
                return dict(old_pixel=native, q=torch.ones(1, 1, 4),
                            ids=torch.zeros(1, 4, dtype=torch.long),
                            count=1, shape=(2, 2))

        torch.manual_seed(3407)
        front = Front().eval()
        features = FrozenFrontCapture(front)
        logits = NativeA1LogitsCapture(front)
        try:
            logits.clear()
            result, feature = features(torch.zeros(1, 2, 2, 2),
                                       torch.ones(1, 2, 2, 2))
            native = logits.take()
            self.assertTrue(torch.equal(native, result["old_pixel"]))
            probability, replay = native_grid_probq(result, feature, native, (2, 2))
            expected = native.softmax(1).mean(dim=(2, 3), keepdim=True).expand_as(probability)
            self.assertTrue(replay["passed"])
            self.assertTrue(torch.allclose(probability, expected, atol=1e-6))
        finally:
            logits.close()
            features.close()

    def test_native_a1_capture_rejects_zero_or_multiple_calls(self):
        import torch
        from torch import nn

        class Front(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Module()
                self.base.clf_2 = nn.Conv2d(2, 4, 1)

        front = Front().eval()
        capture = NativeA1LogitsCapture(front)
        try:
            with self.assertRaisesRegex(RuntimeError, "exactly one.*got 0"):
                capture.take()
            capture.clear()
            front.base.clf_2(torch.zeros(1, 2, 2, 2))
            front.base.clf_2(torch.ones(1, 2, 2, 2))
            with self.assertRaisesRegex(RuntimeError, "exactly one.*got 2"):
                capture.take()
        finally:
            capture.close()

    def test_rejects_strict_or_failed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            base = dict(format="SRAC_HARDQUERY_PROTOCOL_V1", complete=True,
                        full_validation_evidence=True, reproduction={"passed": True},
                        image_count=450, scope="all", smoke=False,
                        profile="legacy", pool_mode="native",
                        input_files_unchanged=True, code_unchanged=True,
                        frozen_evidence_unchanged=True, numerical_unchanged=True,
                        selected_images=[f"image_{i:04d}" for i in range(450)])
            path.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(protocol_record(path)[0], path)
            for change in ({"profile": "strict"},
                           {"reproduction": {"passed": False}},
                           {"image_count": 449}):
                record = dict(base, **change)
                path.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "passed legacy/native"):
                    protocol_record(path)

    def test_rejects_changed_new_metric_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "metric_mislead_last.pth"
            path.write_bytes(b"synthetic checkpoint")
            other = root / "metric_j1_hard_last.pth"
            other.write_bytes(b"other synthetic checkpoint")
            summary = dict(format="SRAC_MEMORY_METRIC_V2", complete=True,
                           smoke=False, train_only=True, test_accessed=False,
                           checkpoint_sha256={"mislead": "wrong-checksum",
                                              "j1_hard": digest(other)})
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed metric run"):
                load_metric_checkpoint(path, {}, "mislead", smoke=False)

    def test_rejects_partial_and_changed_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "images.jsonl"
            predictions = root / "predictions"
            predictions.mkdir()
            lines = []
            for index in range(450):
                path = predictions / f"{index:04d}.npz"
                path.write_bytes(f"synthetic {index}".encode())
                lines.append(json.dumps(dict(index=index, image=f"image_{index:04d}",
                                             prediction_file=f"predictions/{index:04d}.npz",
                                             prediction_sha256=digest(path))))
            journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
            summary = dict(format=FORMAT, complete=True,
                           full_validation_evidence=True,
                           reproduction_passed=True, image_count=449,
                           audit_file="audit.json", immutable={"files": True})
            audit = dict(format=FORMAT, complete=True,
                         full_validation_evidence=True,
                         reproduction={"passed": True},
                         image_count=450, test_accessed=False,
                         immutable={"files": True},
                         image_journal_sha256=digest(journal),
                         metric_arms=list(METRIC_ARMS),
                         control_arms=list(CONTROL_ARMS),
                         comparison_pairs=list(PAIRS),
                         metrics={key: {} for key in ALL_ARMS},
                         transitions={key: {} for key in PAIRS},
                         support_diagnostics={key: {} for key in CONTROL_ARMS})
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (root / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed formal"):
                read_formal(root)
            summary["image_count"] = 450
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(read_formal(root)["image_count"], 450)
            (root / "audit.json").write_text(
                json.dumps(dict(audit, control_arms=[])), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "control evidence"):
                read_formal(root)
            (root / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
            (predictions / "0000.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA"):
                read_formal(root)


if __name__ == "__main__":
    unittest.main()
