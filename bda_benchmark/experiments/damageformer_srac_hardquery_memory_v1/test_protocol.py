"""CPU contracts: numeric policy, exact gates and cross-process evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from .protocol import FORMAT, a1_probability, compare_predictions, reproduction_for_scope


class ProtocolContractTests(unittest.TestCase):
    def test_legacy_preserves_startup_numeric_defaults(self):
        script = """
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.numeric import configure_runtime, runtime_settings
before = runtime_settings()
assert configure_runtime('legacy') == before
assert runtime_settings() == before
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    def test_strict_and_explicit_determinism_group(self):
        script = """
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.numeric import configure_runtime
strict = configure_runtime('strict')
assert strict['deterministic_algorithms'] and strict['cudnn_deterministic']
assert not strict['deterministic_warn_only'] and not strict['cudnn_benchmark']
assert not strict['cuda_matmul_allow_tf32'] and not strict['cudnn_allow_tf32']
assert strict['cublas_workspace_config'] in (':4096:8', ':16:8')
off = configure_runtime('legacy', deterministic=False, matmul_tf32=True, cudnn_tf32=True)
assert not off['deterministic_algorithms'] and not off['cudnn_deterministic']
assert not off['deterministic_warn_only'] and not off['cudnn_benchmark']
assert off['cublas_workspace_config'] is None
assert off['cuda_matmul_allow_tf32'] and off['cudnn_allow_tf32']
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    def test_a1_never_enters_region_front(self):
        class Front:
            def __call__(self, *args, **kwargs):
                raise AssertionError("Must not run the region front")

            def base(self, pre, post, return_aux=False):
                assert return_aux
                return None, pre + post, {}

        tensor = torch.randn(1, 4, 3, 3)
        actual = a1_probability(Front(), tensor, tensor)
        self.assertTrue(torch.equal(actual, (2 * tensor).softmax(1)))

    @staticmethod
    def _metrics():
        one = dict(valid_miou=.5, iou=[.5] * 4, precision=[.5] * 4,
                   recall=[.5] * 4, f1=[.5] * 4)
        return {key: json.loads(json.dumps(one)) for key in ("a1_pixel", "direct_region", "j1", "j2", "j3")}

    def test_all_85_metrics_and_fixed_threshold(self):
        expected = self._metrics()
        actual = self._metrics()
        actual["j3"]["recall"][2] += 2.97e-5
        report = reproduction_for_scope(actual, expected, "all", False, 450)
        self.assertFalse(report["passed"])
        self.assertEqual(report["tolerance"], 1e-5)
        self.assertEqual(len(report["checks"]), 85)
        self.assertEqual(sum(not row["passed"] for row in report["checks"]), 1)
        actual["j3"]["recall"][2] = expected["j3"]["recall"][2]
        self.assertTrue(reproduction_for_scope(actual, expected, "all", False, 450)["passed"])
        self.assertFalse(reproduction_for_scope(actual, expected, "all", False, 449)["passed"])
        self.assertIsNone(reproduction_for_scope(actual, expected, "all", True, 2)["passed"])

    def test_a1_only_checks_17_and_labels_partial_evidence(self):
        report = reproduction_for_scope(self._metrics(), self._metrics(), "a1", False, 450)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["checks"]), 17)
        self.assertIn("not_a_joint_training_gate", report["scope"])

    def _fixture(self, root, *, changed=False):
        root.mkdir()
        target = np.array([[[1, 2], [3, 255]]], dtype=np.int16)
        prediction = np.array([[[1, 2], [3, 0]]], dtype=np.uint8)
        if changed:
            prediction[0, 0, 1] = 1
        support = np.array([[7, 8], [-1, -1]])
        if changed:
            support[0] = [8, 7]
        cm = np.diag([0, 1, 1, 1])
        if changed:
            cm[2, 2] -= 1
            cm[2, 1] += 1
        artifact = root / "0000.npz"
        np.savez_compressed(artifact, target=target, prediction_a1_pixel=prediction,
                            support_j3=support, confusion_a1_pixel=cm)
        row = dict(image="event_x", prediction_file=artifact.name,
                   prediction_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        (root / "images.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        report = dict(format=FORMAT, inference_complete=True, complete=not changed,
                      frozen_evidence_unchanged=True, code_unchanged=True,
                      input_files_unchanged=True, numerical_unchanged=True,
                      source_hashes={"source": "same"}, source_head_hashes={}, reference_hashes={},
                      input_fingerprints={}, source_sha256="code", scope="all", selected_images=["event_x"],
                      runtime_settings={"tf32": changed}, pool_mode="native",
                      arguments=dict(dataset=str(root.parent / "dataset"), device="cpu"),
                      metrics={}, reproduction=dict(passed=not changed), confusion=dict(a1_pixel=cm.tolist()))
        (root / "protocol.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    def test_compare_counts_flips_rank_changes_and_failed_reproduction(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / "left", Path(temp) / "right"
            self._fixture(left)
            self._fixture(right, changed=True)
            result = compare_predictions(left, right)
            self.assertTrue(result["complete"])
            self.assertEqual(result["prediction_changes"]["a1_pixel"]["flips_valid_pixels"], 1)
            self.assertEqual(result["support_changes"]["j3"]["changed_slots"], 2)
            self.assertEqual(result["support_changes"]["j3"]["queries_with_set_change"], 0)
            self.assertEqual(result["confusion_right_minus_left"]["a1_pixel"][2][1], 1)
            self.assertEqual(result["confusion_right_minus_left"]["a1_pixel"][2][2], -1)

    def test_comparison_refuses_hash_tampering_and_incomplete_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / "left", Path(temp) / "right"
            self._fixture(left)
            report = self._fixture(right)
            with (right / "0000.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                compare_predictions(left, right)
            report["inference_complete"] = False
            (right / "protocol.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed inference"):
                compare_predictions(left, right)

    def test_comparison_rejects_different_dataset_roots_with_same_gt(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / "left", Path(temp) / "right"
            self._fixture(left)
            report = self._fixture(right)
            report["arguments"]["dataset"] = str(Path(temp) / "different_dataset")
            (right / "protocol.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Dataset root differs"):
                compare_predictions(left, right)

    def test_comparison_records_compute_device_difference_even_with_same_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / "left", Path(temp) / "right"
            self._fixture(left)
            report = self._fixture(right)
            report["arguments"]["device"] = "cuda"
            (right / "protocol.json").write_text(json.dumps(report), encoding="utf-8")
            comparison = compare_predictions(left, right)
            self.assertEqual(comparison["runtime_differences"]["compute_device"],
                             dict(left="cpu", right="cuda:0"))
            report["arguments"]["device"] = "cpu:0"
            (right / "protocol.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(compare_predictions(left, right)["runtime_differences"], {})


if __name__ == "__main__":
    unittest.main()
