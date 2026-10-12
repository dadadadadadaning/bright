# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from script.standard_ML.audit_CMR_M2_multiview import admission_checks
from script.standard_ML.cmr_multiview_utils import multiview_rerank_sweep


class CMRM2MultiviewTests(unittest.TestCase):
    def test_critical_view_can_promote_correct_coarse_candidate(self):
        memory_mean = torch.tensor([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
        ])
        memory_critical = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ])
        memory_labels = torch.tensor([0, 1, 2])
        memory_events = torch.tensor([0, 1, 0])
        query_mean = torch.tensor([[1.0, 0.0]])
        query_critical = torch.tensor([[1.0, 0.0]])
        query_labels = torch.tensor([1])
        query_events = torch.tensor([2])
        sweep = multiview_rerank_sweep(
            memory_mean,
            {"hard_critical": memory_critical},
            memory_labels,
            memory_events,
            query_mean,
            {"hard_critical": query_critical},
            query_labels,
            query_events,
            lambdas=(1.0,),
            coarse_k=3,
            k_values=(1, 2),
            ap_k=2,
            query_chunk=1,
            device="cpu",
            print_progress=False,
        )
        self.assertEqual(sweep["results"]["mean_only"]["cross_event_R@1"], 0.0)
        self.assertEqual(
            sweep["results"]["mean+hard_critical@1"]["cross_event_R@1"],
            1.0,
        )

    def test_same_event_reference_is_excluded_before_reranking(self):
        memory_mean = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
        memory_view = memory_mean.clone()
        memory_labels = torch.tensor([0, 1, 2])
        memory_events = torch.tensor([0, 1, 2])
        query = torch.tensor([[1.0, 0.0]])
        sweep = multiview_rerank_sweep(
            memory_mean,
            {"exist_embedding": memory_view},
            memory_labels,
            memory_events,
            query,
            {"exist_embedding": query},
            torch.tensor([1]),
            torch.tensor([0]),
            lambdas=(0.5,),
            coarse_k=2,
            k_values=(1, 2),
            ap_k=2,
            query_chunk=1,
            device="cpu",
            print_progress=False,
        )
        self.assertEqual(sweep["results"]["mean_only"]["cross_event_R@1"], 1.0)
        self.assertTrue(sweep["results"]["mean_only"]["same_event_excluded"])

    def test_macro_map_averages_classes_instead_of_query_frequency(self):
        memory_mean = torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]
        ])
        memory_labels = torch.tensor([0, 0, 1, 2])
        memory_events = torch.tensor([0, 1, 0, 1])
        query_mean = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        query_labels = torch.tensor([0, 0, 1])
        query_events = torch.tensor([2, 2, 2])
        sweep = multiview_rerank_sweep(
            memory_mean,
            {},
            memory_labels,
            memory_events,
            query_mean,
            {},
            query_labels,
            query_events,
            lambdas=(),
            coarse_k=3,
            k_values=(1, 2),
            ap_k=2,
            query_chunk=2,
            device="cpu",
            print_progress=False,
        )
        result = sweep["results"]["mean_only"]
        expected = sum(v["mAP@2"] for v in result["per_class"].values()) / 2
        self.assertAlmostEqual(result["macro_mAP@2"], expected)

    def test_admission_gate_protects_damaged_and_intact(self):
        baseline = {
            "macro_mAP@10": 0.70,
            "per_class": {
                0: {"R@1": 0.75, "R@5": 0.90},
                1: {"R@1": 0.70, "R@5": 0.90},
                2: {"R@1": 0.82, "R@5": 0.91},
            },
        }
        candidate = {
            "macro_mAP@10": 0.72,
            "per_class": {
                0: {"R@1": 0.76, "R@5": 0.91},
                1: {"R@1": 0.72, "R@5": 0.91},
                2: {"R@1": 0.81, "R@5": 0.90},
            },
        }
        checks = admission_checks(
            candidate,
            baseline,
            ap_k=10,
            minimum_damaged_gain=0.01,
        )
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()

