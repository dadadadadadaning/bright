import unittest

import torch

from model.cmr_c0.case_card import (
    CRITICAL_VARIANTS,
    critical_count,
    extract_fused_case,
    incident_to_disaster_type,
    select_critical_indices,
    validate_case_cache,
)


class CriticalSelectionTests(unittest.TestCase):
    def test_critical_count_is_bounded(self):
        self.assertEqual(critical_count(3, 0.1), 3)
        self.assertEqual(critical_count(20, 0.1), 4)
        self.assertEqual(critical_count(200, 0.3), 32)

    def test_margin_order_and_spatial_nms(self):
        margin = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0])
        xy = torch.tensor([[0, 0], [0, 1], [0, 2], [5, 5], [9, 9]])
        plain, plain_backfill = select_critical_indices(
            margin, xy, ratio=0.4, minimum=2, maximum=2
        )
        diverse, diverse_backfill = select_critical_indices(
            margin, xy, ratio=0.4, minimum=2, maximum=2, nms_radius=1.5
        )
        self.assertEqual(plain.tolist(), [0, 1])
        self.assertEqual(plain_backfill, 0)
        self.assertEqual(diverse.tolist(), [0, 2])
        self.assertEqual(diverse_backfill, 0)

    def test_extracts_fused_roi_and_same_position_margin(self):
        fused = torch.zeros(1, 256, 4, 4)
        fused[0, 0] = 1.0
        fused[0, 1] = torch.arange(16).reshape(4, 4)
        p_building = torch.full((1, 1, 4, 4), 0.75)
        logits = torch.zeros(1, 4, 4, 4)
        logits[0, 1] = 0.25  # Intact
        logits[0, 2] = torch.arange(16).reshape(4, 4)  # Damaged
        logits[0, 3] = -0.5
        component = torch.ones(8, 8)
        case = extract_fused_case(
            fused, p_building, logits, component,
            minimum=4, maximum=32, nms_radius=1.5,
        )
        self.assertEqual(tuple(case["features"].shape), (16, 256))
        self.assertTrue(torch.allclose(
            case["di_margin"], torch.arange(16).float() - 0.25
        ))
        self.assertEqual(case["critical_indices"]["di_top20"].tolist(), [15, 14, 13, 12])
        self.assertEqual(int(case["is_boundary"].sum()), 12)
        self.assertTrue(torch.allclose(case["building_confidence"], torch.full((16,), 0.75)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_mask_and_morphology_kernel_share_device(self):
        fused = torch.randn(1, 256, 4, 4, device="cuda")
        p_building = torch.full((1, 1, 4, 4), 0.75, device="cuda")
        logits = torch.randn(1, 4, 4, 4, device="cuda")
        case = extract_fused_case(
            fused, p_building, logits, torch.ones(8, 8),
            minimum=4, maximum=32,
        )
        self.assertEqual(tuple(case["features"].shape), (16, 256))
        # Published caches are deliberately moved back to CPU before saving.
        self.assertEqual(case["features"].device.type, "cpu")

    def test_incident_is_metadata_not_label(self):
        self.assertEqual(incident_to_disaster_type("libya-flood"), "flood")
        self.assertEqual(incident_to_disaster_type("turkey-earthquake"), "earthquake")


class CaseCacheContractTests(unittest.TestCase):
    @staticmethod
    def payload():
        count = 5
        index = {
            variant: [torch.arange(count, dtype=torch.int16)]
            for variant in CRITICAL_VARIANTS
        }
        return {
            "protocol": "CMR_C0_ISOLATED_fused_roi_critical_case_cache_v1",
            "role": "support",
            "source_region_indices": torch.tensor([7]),
            "features": [torch.randn(count, 256).half()],
            "xy": [torch.zeros(count, 2, dtype=torch.int16)],
            "normalized_xy": [torch.zeros(count, 2).half()],
            "damage_logits": [torch.zeros(count, 3).half()],
            "di_margin": [torch.zeros(count).half()],
            "building_confidence": [torch.ones(count).half()],
            "component_weight": [torch.ones(count).half()],
            "is_boundary": [torch.zeros(count, dtype=torch.bool)],
            "critical_indices": index,
            "image_ids": ["train_0"],
            "events": ["libya-flood"],
            "disaster_types": ["flood"],
            "region_fingerprints": ["abc"],
            "test_accessed": False,
        }

    def test_valid_payload_passes(self):
        validate_case_cache(self.payload(), "support")

    def test_test_access_is_rejected(self):
        payload = self.payload()
        payload["test_accessed"] = True
        with self.assertRaises(RuntimeError):
            validate_case_cache(payload, "support")

    def test_out_of_range_critical_index_is_rejected(self):
        payload = self.payload()
        payload["critical_indices"]["di_top20"][0] = torch.tensor([5])
        with self.assertRaises(RuntimeError):
            validate_case_cache(payload, "support")


if __name__ == "__main__":
    unittest.main()
