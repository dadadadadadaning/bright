import unittest

import torch

from bda_benchmark.script.standard_ML.audit_H6_information_funnel import (
    diagnosis,
    empty_counts,
    event_name,
    finalize_counts,
    update_counts,
)


class H6InformationFunnelTests(unittest.TestCase):
    def test_event_name_keeps_full_disaster_suffix(self):
        self.assertEqual(event_name("la_palma-volcano_00000010"), "la_palma-volcano")

    def test_flow_finds_fusion_and_hierarchy_harm(self):
        target = torch.tensor([[[2, 2, 2, 255]]])
        predictions = {
            # First two Damaged pixels are read correctly by SAR.
            "sar_aux": torch.tensor([[[2, 2, 1, 0]]]),
            # Fusion loses the second, but rescues the third.
            "base": torch.tensor([[[2, 1, 2, 0]]]),
            # Hierarchy loses the first while retaining the rescued third.
            "native": torch.tensor([[[1, 1, 2, 0]]]),
            "final": torch.tensor([[[1, 1, 2, 0]]]),
        }
        counts = empty_counts()
        update_counts(counts, target, predictions)
        report = finalize_counts(counts)
        flow = report["damaged_evidence_flow_counts"]
        rates = report["damaged_evidence_flow_rates"]
        self.assertEqual(report["damaged_pixels"], 3)
        self.assertEqual(flow["sar_correct_base_to_gate_miss"], 1)
        self.assertEqual(flow["sar_wrong_base_correct"], 1)
        self.assertEqual(flow["base_correct_native_to_gate_miss"], 1)
        self.assertAlmostEqual(
            rates["sar_correct_base_to_gate_miss_given_sar_correct"], 0.5
        )
        self.assertAlmostEqual(
            rates["base_correct_native_to_gate_miss_given_base_correct"], 0.5
        )
        findings = diagnosis(report)["findings"]
        self.assertIn(
            "FUSION_OR_DECODER_DISCARDS_CORRECT_SAR_DAMAGED_EVIDENCE", findings
        )
        self.assertIn(
            "HIERARCHY_TURNS_CORRECT_DAMAGED_INTO_NON_AFFECTED", findings
        )

    def test_damaged_prediction_fractions_are_complete(self):
        target = torch.tensor([[[2, 2, 2, 2]]])
        pred = torch.tensor([[[0, 1, 2, 3]]])
        counts = empty_counts()
        update_counts(
            counts,
            target,
            {"sar_aux": pred, "base": pred, "native": pred, "final": pred},
        )
        fractions = finalize_counts(counts)["stages"]["final"][
            "damaged_prediction_fraction"
        ]
        self.assertAlmostEqual(sum(fractions.values()), 1.0)
        self.assertTrue(all(value == 0.25 for value in fractions.values()))


if __name__ == "__main__":
    unittest.main()
