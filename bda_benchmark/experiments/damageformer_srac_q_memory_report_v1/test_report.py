"""CPU-only schema, denominator and formal-evidence checks."""

import copy
import unittest

import numpy as np

from bda_benchmark.experiments.damageformer_srac_q_memory_diagnostic_v1.stats import (
    ExposureAccumulator, StratifiedAccumulator,
)
from .report import FORMAT, build_report, check_formal, human_report


def fixture():
    y = np.array([[1, 2], [2, 1]])
    j1 = np.array([[1, 1], [2, 1]])
    candidates = {
        "j2": np.array([[2, 2], [1, 1]]),
        "j3": np.array([[2, 2], [2, 1]]),
    }
    strata = StratifiedAccumulator()
    strata.add("synthetic_event", y, j1, candidates, {
        "foreground_bin": np.full((2, 2), 2),
        "class_purity_bin": np.array([[1, 3], [1, 3]]),
    })
    # 450 is only a synthetic gate fixture, never a BRIGHT result.
    strata.images = 450
    exposure = ExposureAccumulator()
    exposure.add("synthetic_event", y, j1, candidates, {
        "background_foreground_mixed": np.full((2, 2), 0.2),
        "foreground_class_mixed": np.array([[0.8, 0.1], [0.6, 0.1]]),
    })
    immutable = {name: True for name in (
        "files", "inherited_code", "new_code", "frozen_models", "runtime",
        "protocol", "own_source",
    )}
    summary = {
        "format": FORMAT, "complete": True, "full_validation_evidence": True,
        "image_count": 450, "reproduction_passed": True,
        "test_accessed": False, "training": False,
        "audit_file": "audit.json", "immutable": immutable,
    }
    audit = {
        "format": FORMAT, "complete": True, "full_validation_evidence": True,
        "image_count": 450, "expected_image_count": 450,
        "test_accessed": False, "training": False,
        "reproduction": {"passed": True}, "immutable": immutable,
        "strata": strata.report(), "soft_q_mixed_exposure": exposure.report(),
        "region_inventory": {},
        "numerical_profile": "legacy", "pool_mode": "native",
        "metrics": {name: {"valid_miou": 0.5} for name in
                    ("a1_pixel", "j1", "j2", "j3")},
    }
    return summary, audit


class ReportTests(unittest.TestCase):
    def test_formal_gate_rejects_partial_failed_and_test_touched(self):
        summary, audit = fixture()
        check_formal(summary, audit)
        for target, key, value in (
            ("summary", "image_count", 449),
            ("audit", "full_validation_evidence", False),
            ("summary", "reproduction_passed", False),
            ("audit", "test_accessed", True),
        ):
            s, a = copy.deepcopy((summary, audit))
            (s if target == "summary" else a)[key] = value
            with self.assertRaises(ValueError):
                check_formal(s, a)
        summary["immutable"]["files"] = False
        with self.assertRaises(ValueError):
            check_formal(summary, audit)

    def test_report_uses_fixed_j1_denominators_and_strata(self):
        summary, audit = fixture()
        report = build_report(summary, audit)
        j2 = report["hard_owner_strata"]["all"]["j2"]
        self.assertEqual(j2["damaged_j1_wrong"], 1)
        self.assertEqual(j2["damaged_rescue"], 1)
        self.assertEqual(j2["damaged_j1_correct"], 1)
        self.assertEqual(j2["damaged_harm"], 1)
        self.assertEqual(j2["intact_j1_correct"], 2)
        self.assertEqual(j2["intact_correct_to_damaged"], 1)
        self.assertEqual(j2["intact_correct_to_damaged_rate"], 0.5)
        self.assertEqual(report["hard_owner_strata"]["fg_ge080/class_lt080"]["j2"]["intact_correct_to_damaged"], 1)
        self.assertEqual(report["hard_owner_strata"]["fg_ge080/class_ge095"]["j2"]["damaged_rescue"], 1)
        self.assertAlmostEqual(report["soft_q_exposure"]["j2"]["intact_correct_to_damaged"]["foreground_class_mixed"]["mean"], 0.8)
        self.assertIn("不能证明", human_report(report))

    def test_report_rejects_partition_and_soft_cohort_inconsistency(self):
        summary, audit = fixture()
        audit["strata"]["overall"]["foreground_fraction"]["fg_ge080"]["j2"]["valid_pixels"] += 1
        with self.assertRaisesRegex(ValueError, "partition"):
            build_report(summary, audit)
        summary, audit = fixture()
        audit["soft_q_mixed_exposure"]["overall"]["j2"]["damaged_rescued"]["foreground_class_mixed"]["count"] += 1
        with self.assertRaises(ValueError):
            build_report(summary, audit)


if __name__ == "__main__":
    unittest.main()
