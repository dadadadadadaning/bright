# -*- coding: utf-8 -*-
"""Unit and synthetic rendering checks for the isolated CMR-C0 diagnosis."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib  # noqa: F401
except ImportError:
    HAS_MATPLOTLIB = False
else:
    HAS_MATPLOTLIB = True

from model.cmr_c0.visual_diagnostics import (
    VIS_SCORE_PROTOCOL,
    cosine_similarity_matrix,
    expanded_crop_box,
    score_validation_queries,
    select_diagnostic_samples,
    summarize_manual_observations,
    validate_visual_score_payload,
)
from model.cmr_v1.region_cache import pack_component

if HAS_MATPLOTLIB:
    from script.standard_ML.cmr_c0.visualize_critical_regions import _render_case_sheet


SELECTED = "di_top20_nms"
VARIANTS = ("all", "di_top10", "di_top20", "di_top30", SELECTED)


def _variant(delta: float, probability: float, intact_row: int = 0, damaged_row: int = 1):
    return {
        "local_feature": [0.2, 0.2 + delta, delta, 0.3, 0.3 + delta, delta],
        "s_i": 0.2,
        "s_d": 0.2 + delta,
        "delta_d_i": delta,
        "direct_prediction": int(delta > 0.0),
        "combined_probability": probability,
        "best_intact": {"support_row": intact_row, "rank": 1, "local_score": 0.3},
        "best_damaged": {"support_row": damaged_row, "rank": 2, "local_score": 0.3 + delta},
    }


def _record(row: int, label: int, all_delta: float, critical_delta: float,
            all_probability: float, critical_probability: float, event: str):
    return {
        "query_row": row,
        "source_region_index": row,
        "image_id": f"query_{row}",
        "event": event,
        "disaster_type": "earthquake",
        "label": label,
        "base_prediction": 0,
        "base_probabilities": [0.8, 0.2, 0.0],
        "bbox": [8, 8, 24, 24],
        "area": 256,
        "identity_mode": "exact",
        "region_fingerprint": f"fingerprint-{row}",
        "base_feature": [0.0] * 10,
        "variants": {
            "all": _variant(all_delta, all_probability),
            SELECTED: _variant(critical_delta, critical_probability),
        },
    }


def _case_cache(image_ids, labels, features):
    size = len(image_ids)
    indices = [torch.arange(features[row].shape[0]) for row in range(size)]
    return {
        "image_ids": list(image_ids),
        "events": ["event-a"] * size,
        "disaster_types": ["earthquake"] * size,
        "labels": torch.tensor(labels),
        "base_predictions": torch.zeros(size, dtype=torch.long),
        "base_probabilities": torch.tensor([[0.8, 0.2, 0.0]] * size),
        "source_region_indices": torch.arange(size),
        "bboxes": torch.tensor([[8, 8, 24, 24]] * size),
        "areas": torch.tensor([256] * size),
        "identity_mode": ["exact"] * size,
        "region_fingerprints": [f"fp-{row}" for row in range(size)],
        "image_shapes": torch.tensor([[32, 32]] * size),
        "normalized_xy": [
            torch.tensor([[0.30, 0.30], [0.45, 0.45], [0.60, 0.60], [0.70, 0.40]])
            for _ in range(size)
        ],
        "di_margin": [torch.tensor([-0.4, 0.2, 0.8, 0.5]) for _ in range(size)],
        "is_boundary": [torch.tensor([True, False, False, True]) for _ in range(size)],
        "features": features,
        "critical_indices": {variant: [item.clone() for item in indices] for variant in VARIANTS},
    }


def _region_cache(size: int):
    mask = np.zeros((32, 32), dtype=np.bool_)
    mask[8:24, 8:24] = True
    packed, bbox, shape = pack_component(mask)
    return {
        "source_region_indices": torch.arange(size),
        "packed_component_masks": [packed.clone() for _ in range(size)],
        "bboxes": torch.tensor([bbox] * size),
        "image_shapes": torch.tensor([shape] * size),
    }


class VisualSelectionTests(unittest.TestCase):
    def test_four_groups_are_selected_from_frozen_scores(self):
        records = [
            _record(0, 1, -0.1, 0.4, 0.4, 0.6, "event-a"),
            _record(1, 0, -0.2, 0.5, 0.2, 0.8, "event-a"),
            _record(2, 1, 0.3, -0.3, 0.8, 0.2, "event-b"),
            _record(3, 0, -0.1, -0.2, 0.2, 0.7, "event-b"),
        ]
        groups = select_diagnostic_samples(
            records, selected_variant=SELECTED, per_event=2, maximum_per_group=8
        )
        self.assertEqual(groups["rescue_target"], [0])
        self.assertEqual(groups["harm"], [1])
        # Groups are diagnostic views, not mutually exclusive: a Harm case can
        # also be an all-correct/critical-wrong degradation case.
        self.assertEqual(groups["critical_degradation"], [1, 2])
        self.assertEqual(groups["maximum_disagreement"][0], 1)

    def test_geometry_and_similarity_helpers(self):
        self.assertEqual(expanded_crop_box((10, 10, 20, 20), (32, 40)), (0, 0, 32, 32))
        features = torch.eye(2)
        similarity = cosine_similarity_matrix(
            features, features, torch.tensor([0, 1]), torch.tensor([0, 1])
        )
        self.assertTrue(torch.allclose(similarity, torch.eye(2)))

    def test_test_derived_score_payload_is_rejected(self):
        payload = {
            "protocol": VIS_SCORE_PROTOCOL,
            "test_accessed": True,
            "variants": ["all", SELECTED],
            "records": [{"query_row": 0}],
        }
        with self.assertRaises(RuntimeError):
            validate_visual_score_payload(payload)

    def test_manual_annotations_are_summarized_without_a_decision(self):
        rows = [
            {"group": "rescue_target", "event": "a", "hotspot": "interior"},
            {"group": "rescue_target", "event": "a", "hotspot": "interior"},
            {"group": "harm", "event": "b", "hotspot": "boundary"},
        ]
        summary = summarize_manual_observations(rows, ("hotspot",))
        self.assertEqual(
            summary["by_group"]["rescue_target"]["fields"]["hotspot"]["most_common"],
            "interior",
        )
        self.assertIn("no automatic scientific decision", summary["scope"])


class VisualScoringTests(unittest.TestCase):
    def test_frozen_candidates_produce_two_sided_scores(self):
        support_features = [torch.eye(2), torch.flip(torch.eye(2), dims=[0])]
        support = _case_cache(["intact", "damaged"], [0, 1], support_features)
        support["memory_local_indices"] = torch.tensor([10, 11])
        query = _case_cache(["query"], [1], [torch.eye(2)])
        query["candidate_indices"] = torch.tensor([[10, 11]])
        query["candidate_labels"] = torch.tensor([[0, 1]])
        query["candidate_scores"] = torch.tensor([[0.4, 0.6]])
        records = score_validation_queries(
            query, support, variants=("all", SELECTED), device="cpu",
            neighbor_k=1, progress_every=0,
        )
        self.assertEqual(len(records), 1)
        for variant in ("all", SELECTED):
            self.assertIsNotNone(records[0]["variants"][variant]["best_intact"])
            self.assertIsNotNone(records[0]["variants"][variant]["best_damaged"])


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required for synthetic rendering")
class SyntheticRenderingTests(unittest.TestCase):
    def test_three_row_case_sheet_is_rendered(self):
        query = _case_cache(["query"], [1], [torch.eye(4, 3)])
        support = _case_cache(
            ["intact", "damaged"], [0, 1],
            [torch.eye(4, 3), torch.flip(torch.eye(4, 3), dims=[0])],
        )
        query_region = _region_cache(1)
        support_region = _region_cache(2)
        record = _record(0, 1, -0.1, 0.4, 0.4, 0.7, "event-a")
        raw_cache = {
            image_id: (
                np.full((32, 32, 3), 0.25 + index * 0.15, dtype=np.float32),
                np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32),
            )
            for index, image_id in enumerate(("query", "intact", "damaged"))
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case"
            _render_case_sheet(
                record, "rescue_target", query, support,
                query_region, support_region, {0: 0}, {0: 0, 1: 1},
                directory, SELECTED, 1.0, output, raw_cache, save_pdf=False,
            )
            png = output.with_suffix(".png")
            self.assertTrue(png.exists())
            self.assertGreater(png.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
