"""CPU tests; BRIGHT_ABLATION_REFERENCE points to the verified New package."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

import diagnostics


def _original_slots():
    roots = [
        Path(os.environ.get("BRIGHT_ABLATION_REFERENCE", ".")) / "deps" / "original",
        Path(os.environ.get("SLOT_AUDIT_REFERENCE_ROOT", ".")),
        Path(__file__).parent / "deps" / "original",
        Path(__file__).parent / "deps" / "new" / "deps" / "original",
    ]
    for root in roots:
        path = root / "slots.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("_audit_test_original_slots", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Set BRIGHT_ABLATION_REFERENCE to the verified New package")


def _fixture(building=None, *, memory=False):
    ops = _original_slots()
    q = torch.zeros(1, 9, 4)
    q[:, 0] = 1
    ids = torch.zeros(9, 4, dtype=torch.long)
    feature = torch.tensor([[[[1., 1.], [0., 0.]], [[0., 0.], [1., 1.]]]])
    if building is None:
        building = torch.tensor([[[[0., 0.], [1., 1.]]]])
    slot = ops.prepare_slots(feature, q, ids, 1, (2, 2), building, "soft")
    target = torch.tensor([[[1, 1], [2, 2]]])
    baseline = torch.zeros(1, 4, 2, 2)
    baseline[:, 1] = 1
    common = dict(pixel_features=feature, region_f=torch.tensor([[[.5, .5]]]),
                  region_t=torch.tensor([[[1., 0.]]]), q=q, ids=ids,
                  base_probability_full=baseline)
    example = SimpleNamespace(image="val_image", event="query_event",
                              target=target, common=common, diagnostics={})
    runtime = SimpleNamespace(
        last_extract=dict(name=example.image, result=dict(q=q, ids=ids, count=1, shape=(2, 2)),
                          building=building, slots={"soft": slot},
                          pair={"active": torch.tensor([True])}),
        diagnostics_slot_ops=ops, _train_ids={}, _val_set={example.image},
        args=SimpleNamespace(key_name="t128"),
    )
    # A forward/retrieval from diagnostics must fail the test immediately.
    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostics attempted a model/extraction/retrieval call")
    runtime.extract = runtime.prepare = runtime.capture = forbidden
    supports = {}
    if memory:
        runtime.bank = dict(images=torch.tensor([10, 10, 11]), events=torch.tensor([0, 0, 1]),
                            t128=torch.tensor([[1., 0.], [1., 0.], [0., 1.]]),
                            f256=torch.tensor([[1., 0.], [1., 0.], [0., 1.]]))
        labels = torch.tensor([[[0., .8, .2, 0.], [0., .6, .4, 0.]],
                               [[0., .9, .1, 0.], [0., .9, .1, 0.]],
                               [[0., .3, .7, 0.], [0., .3, .7, 0.]]])
        supports["d"] = dict(support_ids=torch.tensor([[0, 1, 2]]),
                             support_valid=torch.ones(3, 2, dtype=torch.bool),
                             support_fraction=torch.full((3, 2), .5),
                             support_labels=labels, support_t=runtime.bank["t128"])
    return example, runtime, supports


class EffectiveCountsTests(unittest.TestCase):
    def test_kish_count_merges_duplicate_routes_per_pixel(self):
        # Two candidates address the SAME two pixels. Four independent samples
        # would be wrong: pixel masses are [1,1], hence n_eff=2.
        q = torch.tensor([[[.2, .7], [.8, .3]]])
        ids = torch.zeros(2, 2, dtype=torch.long)
        mass, n_eff = diagnostics.effective_route_counts(q, ids, 1)
        self.assertAlmostEqual(float(mass[0, 0]), 2)
        self.assertAlmostEqual(float(n_eff[0, 0]), 2)

    def test_grouped_counts_distinguish_candidates_from_sources(self):
        ids = torch.tensor([[4, 4, 9, -1], [1, 2, 3, -1]])
        weights = torch.tensor([[1., 1., 1., 0.], [1., .5, .5, 0.]])
        unique, effective = diagnostics.grouped_effective_count(ids, weights)
        self.assertEqual(unique.tolist(), [2, 3])
        self.assertAlmostEqual(float(effective[0]), 9 / 5, places=6)
        self.assertAlmostEqual(float(effective[1]), 4 / 1.5, places=6)


class AuditTests(unittest.TestCase):
    def test_positive_mixture_separates_but_constant_b_collapses_raw_slots(self):
        example, runtime, support = _fixture()
        separated = diagnostics.audit_example(example, runtime, support)
        example, runtime, support = _fixture(torch.full((1, 1, 2, 2), .5))
        constant = diagnostics.audit_example(example, runtime, support)
        first = separated["slots_by_rule"]["soft"]["all_active"]
        second = constant["slots_by_rule"]["soft"]["all_active"]
        self.assertAlmostEqual(first["raw_cosine"]["mean"], 0)
        self.assertAlmostEqual(first["normalized_l2"]["mean"], 2 ** .5, places=6)
        self.assertAlmostEqual(first["gt_JS"]["mean"], 1)
        self.assertAlmostEqual(first["slot0_n_eff"]["mean"], 2)
        self.assertAlmostEqual(second["raw_cosine"]["mean"], 1, places=6)
        self.assertAlmostEqual(second["gt_JS"]["mean"], 0)
        self.assertAlmostEqual(second["slot0_n_eff"]["mean"], 4)
        self.assertEqual(separated["strata_counts"]["mixed"], 1)
        self.assertEqual(separated["strata_counts"]["border"], 1)
        self.assertEqual(separated["query_counts"]["GT_argmax"]["intact"], 1)
        self.assertEqual(separated["query_counts"]["GT_argmax"]["damaged"], 0)

    def test_memory_soft_mass_and_event_repetition_are_measured(self):
        example, runtime, support = _fixture(memory=True)
        result = diagnostics.audit_example(example, runtime, support)
        stats = result["memory_by_arm"]["d"]["strata"]["all_active"]
        self.assertEqual(stats["unique_parent_count"]["mean"], 3)
        self.assertEqual(stats["unique_image_count"]["mean"], 2)
        self.assertEqual(stats["unique_event_count"]["mean"], 2)
        self.assertAlmostEqual(stats["effective_parent_k"]["mean"], 3)
        self.assertAlmostEqual(stats["effective_event_k"]["mean"], 1.8, places=6)
        self.assertAlmostEqual(stats["softlabel_damaged_mass"]["mean"], 1.1 / 3, places=6)
        self.assertAlmostEqual(stats["intact_dominant_parent_rate"]["mean"], 2 / 3, places=6)
        self.assertAlmostEqual(stats["intact_hard_negative_rate"]["mean"], 2 / 3, places=6)
        self.assertFalse(result["memory_by_arm"]["d"]["metadata"]["learned_attention_measured"])

    def test_memory_chunk_boundary_and_original_parent_mapping(self):
        example, runtime, supports = _fixture(memory=True)
        rows = 257
        example.common["region_t"] = example.common["region_t"].expand(1, rows, -1)
        example.common["region_f"] = example.common["region_f"].expand(1, rows, -1)
        supports["d"]["support_ids"] = torch.tensor([[0, 1, 2]]).expand(rows, -1)
        # Two distinct prepared table rows map to the same original parent.
        supports["d"]["support_parent_ids"] = torch.tensor([0, 0, 2])
        metrics, _, metadata, parent_ids, _ = diagnostics._memory_metrics(example, runtime, supports["d"])
        self.assertEqual(metrics["effective_parent_k"].shape, (rows,))
        self.assertEqual(parent_ids[-1].tolist(), [0, 0, 2])
        self.assertTrue(torch.allclose(metrics["effective_parent_k"], torch.full((rows,), 1.8)))
        self.assertTrue(torch.equal(metrics["unique_parent_count"], torch.full((rows,), 2.)))
        self.assertTrue(metadata["identity_available"])

    def test_padding_and_zero_valid_labels_do_not_enter_denominators(self):
        example, runtime, support = _fixture(memory=True)
        support["d"]["support_ids"] = torch.tensor([[0, -1, -1]])
        example.target.fill_(255)
        result = diagnostics.audit_example(example, runtime, support)
        stats = result["memory_by_arm"]["d"]["strata"]["all_active"]
        self.assertEqual(stats["unique_parent_count"]["mean"], 1)
        self.assertEqual(stats["unique_event_count"]["mean"], 1)
        self.assertEqual(result["query_counts"]["GT_valid_active"], 0)
        self.assertEqual(result["slots_by_rule"]["soft"]["all_active"]["gt_JS"]["n"], 0)
        self.assertIsNone(result["selected_cases"][0]["gt_argmax"])
        json.dumps(result, allow_nan=False)

    def test_read_only_no_forward_and_json_only(self):
        example, runtime, support = _fixture(memory=True)
        def tensor_copies(value):
            if torch.is_tensor(value):
                return value.clone()
            if isinstance(value, dict):
                return {key: tensor_copies(item) for key, item in value.items()}
            return value
        def assert_equal(before, after):
            if torch.is_tensor(before):
                self.assertTrue(torch.equal(before, after))
                self.assertEqual(before.requires_grad, after.requires_grad)
            elif isinstance(before, dict):
                self.assertEqual(set(before), set(after))
                for key in before:
                    assert_equal(before[key], after[key])
        before = tensor_copies(dict(data=runtime.last_extract, common=example.common,
                                    bank=runtime.bank, supports=support, target=example.target))
        result = diagnostics.audit_example(example, runtime, support, sample_limit=0)
        assert_equal(before, dict(data=runtime.last_extract, common=example.common,
                                 bank=runtime.bank, supports=support, target=example.target))
        self.assertEqual(result["selected_cases"], [])
        self.assertFalse(result["query_GT_used_for_supports"])
        self.assertEqual(result["source_forward_count_added"], 0)
        json.dumps(result, allow_nan=False)

    def test_pooled_aggregation_uses_counts_and_rejects_duplicate_images(self):
        example, runtime, support = _fixture()
        first = diagnostics.audit_example(example, runtime, support)
        second = copy.deepcopy(first)
        second["image"] = "second_image"
        # An explicit unequal-denominator sufficient statistic catches a mean
        # of image means: pooled mean is 3/4, not 1/2.
        leaf = second["slots_by_rule"]["soft"]["all_active"]["raw_cosine"]
        leaf.update(n=3, sum=3., sum_sq=3., mean=1., std=0., min=1., max=1.)
        total = diagnostics.summarize_audits(iter([first, second]))
        result = total["slots_by_rule"]["soft"]["all_active"]["raw_cosine"]
        self.assertEqual(total["image_count"], 2)
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["mean"], .75)
        self.assertAlmostEqual(result["std"], (3 / 16) ** .5)
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            diagnostics.summarize_audits([first, first])
        json.dumps(total, allow_nan=False)

    def test_stale_extract_rejected(self):
        example, runtime, support = _fixture()
        runtime.last_extract["name"] = "another_image"
        with self.assertRaisesRegex(ValueError, "this image"):
            diagnostics.audit_example(example, runtime, support)


if __name__ == "__main__":
    unittest.main()
