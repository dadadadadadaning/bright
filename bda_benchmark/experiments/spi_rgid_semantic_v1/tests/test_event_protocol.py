import unittest

from bda_benchmark.experiments.spi_rgid_semantic_v1.event_protocol import (
    CrossEventGate,
    assert_event_disjoint,
    balanced_event_folds,
    cross_event_decision,
    sample_event,
)


class EventProtocolTests(unittest.TestCase):
    def test_sample_event_preserves_internal_hyphens(self):
        self.assertEqual(sample_event("turkey-earthquake_00000123"), "turkey-earthquake")

    def test_event_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "event leakage"):
            assert_event_disjoint(
                ["event-a_0001", "event-b_0001"], ["event-a_0002"]
            )

    def test_balanced_folds_never_split_an_event(self):
        names = [
            "event-a_0001", "event-a_0002", "event-b_0001",
            "event-c_0001", "event-d_0001", "event-e_0001",
        ]
        folds = balanced_event_folds(names, fold_count=3)
        owner = {}
        for fold_index, rows in enumerate(folds):
            for row in rows:
                event = sample_event(row)
                owner.setdefault(event, fold_index)
                self.assertEqual(owner[event], fold_index)

    def test_cross_event_gate_passes_only_complete_evidence(self):
        deltas = {
            f"event-{index}": {"miou_pp": 0.7, "damaged_f1_pp": 1.2}
            for index in range(4)
        }
        gate = CrossEventGate(nonnegative_events_min=3)
        passed = cross_event_decision(deltas, bootstrap_ci_low=0.1, gate=gate)
        self.assertEqual(passed["decision"], "CROSS_EVENT_IMPROVEMENT_SUPPORTED")
        failed = cross_event_decision(deltas, bootstrap_ci_low=-0.1, gate=gate)
        self.assertEqual(failed["decision"], "CROSS_EVENT_EVIDENCE_INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
