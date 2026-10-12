# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.hmemory.occurrence_key import (
    extract_occurrence_key_variants,
    masked_mean_key,
    soft_occurrence_key,
)
from script.standard_ML.hm_occurrence_runtime import (
    event_class_capped_indices,
    event_probe_report,
    loeo_occurrence_scores,
    occurrence_auc_report,
)


class HierarchicalMemoryOccurrenceKeyTests(unittest.TestCase):
    def test_masked_mean_uses_only_component(self):
        tokens = torch.tensor(
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 9.0], [9.0, 9.0]]]
        )
        mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        key = masked_mean_key(tokens, mask)
        self.assertTrue(torch.allclose(key, torch.tensor([1.0, 0.0])))

    def test_soft_occurrence_concentrates_on_high_response(self):
        tokens = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]]]
        )
        response = torch.tensor([[8.0, -8.0]])
        mask = torch.ones(1, 2)
        key = soft_occurrence_key(tokens, response, mask, temperature=0.25)
        self.assertGreater(float(key[0]), 0.999)
        self.assertLess(abs(float(key[1])), 0.001)

    def test_extraction_contract_has_no_labels(self):
        auxiliary = {
            "stage2_local_tokens": torch.randn(1, 64, 3, 3),
            "stage2_raw_delta": torch.randn(1, 1, 3, 3),
        }
        keys = extract_occurrence_key_variants(
            auxiliary, torch.ones(6, 6), temperatures=(0.5, 1.0)
        )
        self.assertEqual(
            tuple(keys), ("s2o_masked_mean", "s2o_soft_t0p5", "s2o_soft_t1")
        )
        self.assertTrue(all(tuple(value.shape) == (64,) for value in keys.values()))

    def test_event_class_cap_never_repeats_or_upsamples(self):
        labels = torch.tensor([0, 0, 0, 1, 0, 1, 1])
        events = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        selected = event_class_capped_indices(
            labels, events, cap_per_event_class=2, seed=3407
        )
        self.assertEqual(selected.numel(), 7 - 1)
        self.assertEqual(torch.unique(selected).numel(), selected.numel())

    def test_loeo_scores_every_row_without_held_event_support(self):
        features = []
        labels = []
        events = []
        for event in range(3):
            features.extend([[1.0, 0.05 * event], [-1.0, 0.05 * event]])
            labels.extend([1, 0])
            events.extend([event, event])
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        events = torch.tensor(events)
        scores = loeo_occurrence_scores(
            features,
            labels,
            events,
            cap_per_event_class=2,
            seed=3407,
        )
        report = occurrence_auc_report(scores, labels, [str(v) for v in events.tolist()])
        self.assertTrue(torch.isfinite(scores).all())
        self.assertAlmostEqual(report["global_auroc"], 1.0)
        self.assertEqual(report["direction_consistent_events"], 3)

    def test_event_probe_reports_event_leakage(self):
        train_feature = torch.tensor([[1.0, 0.0], [0.9, 0.0], [0.0, 1.0], [0.0, 0.9]])
        train_event = torch.tensor([0, 0, 1, 1])
        report = event_probe_report(
            train_feature,
            train_event,
            train_feature,
            train_event,
            cap_per_event=2,
            seed=3407,
        )
        self.assertAlmostEqual(report["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
