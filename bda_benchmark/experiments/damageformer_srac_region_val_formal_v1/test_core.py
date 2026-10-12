"""CPU-only evidence contracts; no BRIGHT image or checkpoint is opened."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from .core import (ARMS, PAIRS, checked_journal, checked_protocol, formal_gate,
                   metric_deltas, reproduction_metrics, source_hash)
from .report import IMMUTABLE_KEYS, checked_result, format_report


class GateTests(unittest.TestCase):
    def test_only_full_reproduced_immutable_val450_is_formal(self):
        good = dict(image_count=450, expected_count=450,
                    reproduction={"passed": True},
                    immutable={"files": True, "probe_code": True},
                    pool_ok=True, native_a1_replay_ok=True)
        self.assertTrue(formal_gate(**good))
        for key, value in (("image_count", 449), ("expected_count", 878),
                           ("reproduction", {"passed": False}),
                           ("immutable", {"files": False}),
                           ("pool_ok", False), ("native_a1_replay_ok", False)):
            with self.subTest(key=key):
                case = dict(good, **{key: value})
                self.assertFalse(formal_gate(**case))

    def test_alias_and_paired_deltas_keep_formal_meaning(self):
        values = {name: {"valid_miou": 0.6} for name in ARMS}
        values["region"]["valid_miou"] = 0.5
        values["grid_probq"]["valid_miou"] = 0.61
        self.assertIs(reproduction_metrics(values)["direct_region"], values["region"])
        self.assertEqual(set(metric_deltas(values)), set(PAIRS))
        self.assertAlmostEqual(metric_deltas(values)["pixel_grid__grid_probq"], 1.0)
        with self.assertRaises(ValueError):
            reproduction_metrics({"a1_pixel": values["a1_pixel"]})

    def test_protocol_rejects_nonformal_or_wrong_numeric_profile(self):
        record = dict(
            format="SRAC_HARDQUERY_PROTOCOL_V1", complete=True,
            full_validation_evidence=True, image_count=450, scope="all",
            profile="legacy", pool_mode="native", smoke=False,
            reproduction={"passed": True},
            selected_images=[f"event_{i:03d}" for i in range(450)],
            input_files_unchanged=True, code_unchanged=True,
            frozen_evidence_unchanged=True, numerical_unchanged=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(checked_protocol(path)[1]["image_count"], 450)
            for key, value in (("profile", "strict"),
                               ("pool_mode", "grouped"),
                               ("reproduction", {"passed": False}),
                               ("selected_images", record["selected_images"][:-1]),
                               ("numerical_unchanged", False)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    path.write_text(json.dumps(dict(record, **{key: value})), encoding="utf-8")
                    checked_protocol(path)

    def test_per_image_journal_needs_order_and_hash(self):
        from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
            digest, safe_child,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "prediction.npz"
            file.write_bytes(b"small synthetic artifact")
            row = dict(index=0, image="val_0", prediction_file=file.name,
                       prediction_sha256=digest(file))
            log = root / "images.jsonl"
            log.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(len(checked_journal(root, ["val_0"], digest, safe_child)), 1)
            file.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                checked_journal(root, ["val_0"], digest, safe_child)
            row["prediction_sha256"] = digest(file)
            log.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                checked_journal(root, ["different_val"], digest, safe_child)

    def test_probe_source_has_dependency_hash(self):
        self.assertEqual(len(source_hash()), 64)

    def test_human_report_refuses_failed_evidence(self):
        metrics = {arm: {"valid_miou": 0.6, "iou": [0.9, 0.7, 0.5, 0.6]}
                   for arm in ARMS}
        metrics["grid_probq"] = {"valid_miou": 0.61,
                                 "iou": [0.9, 0.69, 0.52, 0.62]}
        base = dict(format="SRAC_REGION_VAL_FORMAL_V1", complete=True,
                    full_validation_evidence=True, image_count=450,
                    test_accessed=False, immutable={key: True for key in IMMUTABLE_KEYS},
                    native_a1_replay_ok=True, source_sha256=source_hash())
        summary = dict(base, reproduction_passed=True, result_file="result.json")
        result = dict(base, expected_image_count=450, training=False,
                      reproduction={"passed": True, "image_count": 450,
                                    "tolerance": 1e-5, "checks": [{"passed": True}]},
                      metrics=metrics,
                      transitions={pair: {} for pair in PAIRS},
                      pool_guard={"backend": "native", "restored": True, "failures": []},
                      per_event={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            self.assertIn("pixel_grid→grid_probq: +1.0000", format_report(checked_result(root)))
            result["reproduction"] = {"passed": False}
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaises(ValueError):
                checked_result(root)


if __name__ == "__main__":
    unittest.main()
