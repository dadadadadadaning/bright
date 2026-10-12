import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bda_benchmark.script.standard_ML.audit_H9_building_oracle import (
    BuildingInstance,
    aggregate_building_probability,
    build_decision,
    complementarity_counts,
    confusion3,
    greedy_match_masks,
    index_coco_annotations,
    load_official_instance_records,
    oracle_severity_labels,
    rasterize_instances,
    sample_id_from_file_name,
    severity_metrics,
)


class H9BuildingOracleTests(unittest.TestCase):
    def test_sample_id_contract_matches_bright_names(self):
        self.assertEqual(
            sample_id_from_file_name("libya-flood_001_post_disaster.tif"),
            "libya-flood_001",
        )
        self.assertEqual(
            sample_id_from_file_name("libya-flood_001_instance_damage.json"),
            "libya-flood_001",
        )

    def test_coco_index_preserves_official_instances(self):
        payload = {
            "images": [
                {
                    "id": 7,
                    "file_name": "libya-flood_001_post_disaster.tif",
                    "height": 4,
                    "width": 5,
                }
            ],
            "annotations": [
                {
                    "id": 11,
                    "image_id": 7,
                    "category_id": 2,
                    "segmentation": [[0, 0, 1, 0, 1, 1]],
                },
                {
                    "id": 12,
                    "image_id": 7,
                    "category_id": 3,
                    "segmentation": [[2, 2, 3, 2, 3, 3]],
                },
            ],
            "categories": [
                {"id": 1, "name": "intact"},
                {"id": 2, "name": "damaged"},
                {"id": 3, "name": "destroyed"},
            ],
        }
        index = index_coco_annotations(payload)
        self.assertEqual(list(index), ["libya-flood_001"])
        self.assertEqual(len(index["libya-flood_001"]["annotations"]), 2)

    def test_per_image_loader_obeys_requested_evaluation_split(self):
        payload = {
            "images": [
                {
                    "id": 1,
                    "file_name": "generic_post_disaster.tif",
                    "height": 4,
                    "width": 5,
                }
            ],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "intact"},
                {"id": 2, "name": "damaged"},
                {"id": 3, "name": "destroyed"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "libya-flood_001_instance_damage.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            records, manifest = load_official_instance_records(
                ["libya-flood_001"], per_image_json_dir=str(directory)
            )
        self.assertEqual(list(records), ["libya-flood_001"])
        self.assertEqual(manifest["requested_samples"], 1)
        self.assertEqual(manifest["source"], "official_per_image_coco_jsons")

    def test_building_aggregation_ignores_background_after_geometry_is_known(self):
        probability = torch.tensor(
            [
                [[0.90, 0.90], [0.90, 0.90]],
                [[0.02, 0.02], [0.02, 0.02]],
                [[0.06, 0.06], [0.06, 0.06]],
                [[0.02, 0.02], [0.02, 0.02]],
            ]
        )
        distribution = aggregate_building_probability(
            probability, np.ones((2, 2), dtype=np.bool_)
        )
        self.assertAlmostEqual(float(distribution.sum()), 1.0)
        self.assertEqual(int(distribution.argmax()), 1)

    def test_oracle_keeps_p3_unless_a2_alone_is_correct(self):
        p3 = [1, 2, 1, 2]
        a2 = [2, 3, 3, 1]
        target = [1, 3, 2, 3]
        oracle = oracle_severity_labels(p3, a2, target)
        self.assertEqual(oracle, [1, 3, 1, 2])
        for p3_label, a2_label, target_label, oracle_label in zip(
            p3, a2, target, oracle
        ):
            if p3_label == target_label or a2_label == target_label:
                self.assertEqual(oracle_label, target_label)

    def test_complementarity_partition_is_complete(self):
        counts = complementarity_counts(
            [1, 2, 1, 2],
            [1, 3, 3, 1],
            [1, 2, 3, 3],
        )
        self.assertEqual(counts["valid"], 4)
        self.assertEqual(
            counts["valid"],
            counts["both_correct"]
            + counts["p3_only_correct"]
            + counts["a2_only_correct"]
            + counts["neither_correct"],
        )

    def test_severity_metrics_match_identity_confusion(self):
        metrics = severity_metrics(torch.diag(torch.tensor([4, 3, 2])))
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1_all3"], 1.0)
        self.assertAlmostEqual(metrics["per_class"]["damaged"]["f1"], 1.0)

    def test_greedy_matching_is_one_to_one(self):
        mask_a = np.zeros((4, 4), dtype=np.bool_)
        mask_a[:2, :2] = True
        mask_b = np.zeros((4, 4), dtype=np.bool_)
        mask_b[2:, 2:] = True
        predicted = [
            BuildingInstance(mask_a, None, 1),
            BuildingInstance(mask_b, None, 2),
        ]
        target = [
            BuildingInstance(mask_a, 1, 10),
            BuildingInstance(mask_b, 3, 11),
        ]
        matches = greedy_match_masks(predicted, target, iou_threshold=0.5)
        self.assertEqual(len(matches), 2)
        self.assertEqual({row[0] for row in matches}, {0, 1})
        self.assertEqual({row[1] for row in matches}, {0, 1})

    def test_rasterization_preserves_higher_score_on_overlap(self):
        large = np.ones((3, 3), dtype=np.bool_)
        small = np.zeros((3, 3), dtype=np.bool_)
        small[1, 1] = True
        instances = [
            BuildingInstance(large, None, 1, score=0.9),
            BuildingInstance(small, None, 2, score=0.2),
        ]
        raster, overlap = rasterize_instances(
            instances,
            [1, 3],
            height=3,
            width=3,
            preserve_high_score=True,
        )
        self.assertEqual(int(raster[1, 1]), 1)
        self.assertEqual(overlap, 1)

    def test_decision_requires_gt_event_and_deploy_gates(self):
        def report(macro_p3, macro_oracle, dmg_p3, dmg_oracle, valid_p3, valid_oracle):
            def expert(macro, damaged):
                return {
                    "macro_f1_present_gt": macro,
                    "per_class": {
                        "intact": {"f1": macro},
                        "damaged": {"f1": damaged},
                        "destroyed": {"f1": macro},
                    },
                }

            return {
                "building_metrics": {
                    "p3": expert(macro_p3, dmg_p3),
                    "a2": expert(macro_p3, dmg_p3),
                    "oracle": expert(macro_oracle, dmg_oracle),
                },
                "semantic_metrics": {
                    "p3": {"valid_miou": valid_p3, "miou4": valid_p3},
                    "a2": {"valid_miou": valid_p3, "miou4": valid_p3},
                    "oracle": {"valid_miou": valid_oracle, "miou4": valid_oracle},
                },
            }

        gt = report(0.60, 0.62, 0.40, 0.42, 0.60, 0.62)
        deploy = report(0.55, 0.56, 0.35, 0.36, 0.55, 0.56)
        events = [
            {"gt_oracle_gains": {"building_macro_f1_pp": 0.1}}
            for _ in range(10)
        ]
        decision = build_decision(
            gt,
            deploy,
            events,
            deploy_is_proxy=False,
            minimum_macro_f1_gain_pp=1.0,
            minimum_damaged_f1_gain_pp=1.0,
            minimum_positive_events=10,
            minimum_deploy_valid_gain_pp=0.5,
        )
        self.assertTrue(decision["passed"])
        self.assertEqual(
            decision["recommendation"], "IMPLEMENT_H9_B1_BUILDING_SELECTOR"
        )


if __name__ == "__main__":
    unittest.main()
