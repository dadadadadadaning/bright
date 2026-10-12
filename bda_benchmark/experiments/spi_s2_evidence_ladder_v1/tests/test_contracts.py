import json
import tempfile
import unittest
from pathlib import Path

from bda_benchmark.experiments.spi_s2_evidence_ladder_v1.contracts import (
    decide_stage,
    load_protocol,
    sha256_file,
    stage_specs,
    validate_result,
)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol()
        self.specs = stage_specs(self.protocol)

    def test_stage_three_gate_accepts_only_when_all_checks_pass(self):
        summary = {
            "delta_miou_pp": 0.2,
            "harm_reduction_fraction": 0.25,
            "selector_auroc": 0.7,
            "event_nonnegative_fraction": 0.7,
            "minimum_foreground_delta_pp": -0.1,
        }
        decision, checks = decide_stage("s3_base_utility_gate", summary, self.protocol)
        self.assertEqual(decision, "ACCEPT")
        self.assertTrue(all(checks.values()))

    def test_parent_fingerprint_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            parent_path = Path(directory) / "parent.json"
            parent_path.write_text(json.dumps({"stage_id": "s1_s2_audit", "decision": "ACCEPT"}))
            result = {
                "protocol": "SPI_S2_EVIDENCE_RESULT_V1",
                "stage_id": "s2_direction_policy",
                "summary": {"accepted_directions": ["1->2"], "event_direction_stability": {"1->2": {}}},
                "parent_result_sha256": sha256_file(parent_path),
                "official_instances_accessed": False,
                "test_accessed": False,
            }
            validate_result(result, self.specs["s2_direction_policy"], parent_result_path=parent_path)
            result["parent_result_sha256"] = "wrong"
            with self.assertRaises(RuntimeError):
                validate_result(result, self.specs["s2_direction_policy"], parent_result_path=parent_path)

    def test_rgid_requires_support_and_random_control(self):
        summary = {
            "delta_miou_pp": 0.2,
            "event_nonnegative_fraction": 0.7,
            "minimum_foreground_delta_pp": -0.1,
            "support_conserved": True,
            "matched_random_split_control_pp": -0.01,
        }
        decision, checks = decide_stage("s6_selective_rgid", summary, self.protocol)
        self.assertEqual(decision, "REJECT")
        self.assertFalse(checks["beats_matched_random_split"])


if __name__ == "__main__":
    unittest.main()

