"""CPU tests for address-range alias attribution without tensor-value reads."""
from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import patch

import torch

from storage_audit import _alias, _metadata, audit_storage


class Head(torch.nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.weight = torch.nn.Parameter(parameter)


class StorageAuditTests(unittest.TestCase):
    def audit(self, watched, head, optimizer=None):
        return audit_storage(
            torch, watched, {"e": head},
            {"e": optimizer} if optimizer is not None else {},
        )

    def test_overlapping_views_with_different_data_pointers(self):
        backing = torch.zeros(16)
        head = Head(backing[4:12])
        watched = backing[2:8]
        self.assertNotEqual(watched.data_ptr(), head.weight.data_ptr())
        report = self.audit({"q": watched}, head)
        self.assertEqual(len(report["unsafe_aliases"]), 1)
        alias = report["unsafe_aliases"][0]
        self.assertEqual(alias["owner"]["owner_type"], "parameter")
        self.assertEqual(alias["owner"]["arm"], "e")
        self.assertEqual(alias["owner"]["name"], "weight")
        self.assertTrue(alias["same_storage_start"])
        self.assertTrue(alias["byte_spans_overlap"])
        self.assertTrue(alias["tensor_overlap_confirmed"])
        self.assertEqual(alias["byte_overlap_end"] - alias["byte_overlap_start"], 4 * 4)

    def test_disjoint_views_in_same_allocation_remain_conservatively_flagged(self):
        backing = torch.zeros(16)
        report = self.audit({"q": backing[:4]}, Head(backing[8:]))
        self.assertEqual(len(report["unsafe_aliases"]), 1)
        alias = report["unsafe_aliases"][0]
        self.assertTrue(alias["storage_ranges_overlap"])
        self.assertFalse(alias["byte_spans_overlap"])
        self.assertFalse(alias["tensor_overlap_confirmed"])
        self.assertEqual(alias["relation"], "disjoint_tensor_byte_spans_in_shared_storage")
        self.assertIsNone(alias["byte_overlap_start"])

    def test_independent_tensors_have_no_aliases(self):
        head = Head(torch.zeros(8))
        report = self.audit({"q": torch.zeros(8), "ids": torch.zeros(8, dtype=torch.long)}, head)
        self.assertEqual(report["unsafe_aliases"], [])
        self.assertEqual(report["watch_aliases"], [])
        self.assertEqual(report["owned_tensor_counts"], {"parameter": 1})
        json.dumps(report, allow_nan=False)

    def test_gradient_moment_and_scalar_step_aliases_are_found(self):
        backing = torch.zeros(32)
        head = Head(torch.zeros(8))
        head.weight.grad = backing[:8]
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
        optimizer.state[head.weight] = dict(
            step=backing[16], exp_avg=backing[8:16],
            exp_avg_sq=torch.zeros(8),
        )
        report = self.audit({"q": backing[3:20]}, head, optimizer)
        owners = report["unsafe_aliases"]
        self.assertEqual(len(owners), 3)
        self.assertEqual(
            {(item["owner"]["owner_type"], item["owner"]["name"]) for item in owners},
            {("gradient", "weight"), ("optimizer_state", "weight.exp_avg"),
             ("optimizer_state", "weight.step")},
        )
        self.assertTrue(all(item["tensor_overlap_confirmed"] for item in owners))
        self.assertEqual(report["owned_tensor_count"], 5)

    def test_frozen_watch_aliases_are_reported_separately(self):
        q = torch.zeros(2, 9, 5)
        report = audit_storage(torch, {"q": q, "q_detached": q.detach()}, {}, {})
        self.assertEqual(report["unsafe_aliases"], [])
        self.assertEqual(len(report["watch_aliases"]), 1)
        self.assertTrue(report["watch_aliases"][0]["tensor_overlap_confirmed"])

    def test_strided_holes_are_reported_as_conservative_spans(self):
        backing = torch.zeros(16)
        report = self.audit({"q": backing[::2]}, Head(backing[1::2]))
        alias = report["unsafe_aliases"][0]
        self.assertTrue(alias["byte_spans_overlap"])
        self.assertIsNone(alias["tensor_overlap_confirmed"])
        self.assertFalse(alias["watched"]["metadata"]["byte_span_is_dense"])

    def test_dense_transpose_has_exact_byte_span(self):
        backing = torch.zeros(3, 4)
        metadata = _metadata(torch, backing.t())
        self.assertTrue(metadata["byte_span_is_dense"])
        self.assertEqual(metadata["byte_span_end"] - metadata["byte_span_start"], 12 * 4)

    def test_distinct_storage_wrappers_over_external_memory_are_detected(self):
        import numpy as np
        backing = np.zeros(16, dtype=np.float32)
        left = torch.from_numpy(backing[1:9])
        right = torch.from_numpy(backing[5:13])
        report = self.audit({"q": left}, Head(right))
        alias = report["unsafe_aliases"][0]
        self.assertFalse(alias["same_storage_start"])
        self.assertTrue(alias["tensor_overlap_confirmed"])
        self.assertEqual(alias["byte_overlap_end"] - alias["byte_overlap_start"], 4 * 4)

    def test_device_identity_prevents_address_only_collisions(self):
        left = _metadata(torch, torch.zeros(8))
        for device_key in (["cuda", 0], ["cuda", 1], ["mps", None]):
            with self.subTest(device_key=device_key):
                right = copy.deepcopy(left)
                right["device_key"] = device_key
                self.assertIsNone(_alias(left, right))
        left["device_key"] = ["cuda", 0]
        right = copy.deepcopy(left)
        right["device_key"] = ["cuda", 1]
        self.assertIsNone(_alias(left, right))

    def test_empty_allocations_do_not_alias_but_empty_views_share_storage(self):
        report = audit_storage(torch, {"a": torch.empty(0), "b": torch.empty(0)}, {}, {})
        self.assertEqual(report["watch_aliases"], [])
        backing = torch.zeros(8)
        report = self.audit({"q": backing[:0]}, Head(backing))
        self.assertEqual(len(report["unsafe_aliases"]), 1)
        self.assertFalse(report["unsafe_aliases"][0]["byte_spans_overlap"])

    def test_metadata_audit_reads_no_tensor_values_and_changes_no_state_or_rng(self):
        backing = torch.arange(16, dtype=torch.float32)
        head = Head(backing[4:12])
        head.weight.grad = torch.ones_like(head.weight)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
        optimizer.state[head.weight] = dict(
            step=torch.tensor(1000.), exp_avg=torch.zeros(8), exp_avg_sq=torch.zeros(8),
        )
        before_rng = torch.get_rng_state().clone()
        before_value = backing.clone()
        before_version = backing._version
        before_keys = tuple(optimizer.state[head.weight])
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("cpu read")), \
                patch.object(torch.Tensor, "numpy", side_effect=AssertionError("numpy read")), \
                patch.object(torch.Tensor, "item", side_effect=AssertionError("item read")), \
                patch.object(torch.Tensor, "tolist", side_effect=AssertionError("list read")):
            report = self.audit({"q": backing[:8]}, head, optimizer)
        self.assertTrue(report["metadata_only"])
        self.assertFalse(report["values_read"])
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assertTrue(torch.equal(backing, before_value))
        self.assertEqual(backing._version, before_version)
        self.assertEqual(tuple(optimizer.state[head.weight]), before_keys)


if __name__ == "__main__":
    unittest.main()
