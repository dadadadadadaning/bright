"""CPU contracts for observation-only pooling diagnostics; not BRIGHT evidence."""
from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from .feature_core import (
    damaged_margin_sums, feature_branches, mean_pool, resize_probability,
    unpool, validate_routes,
)
from .shared import tensor_hash, transition_counts, transition_summary


def example(dtype=torch.float32):
    q = torch.tensor([[[.8, .2, .7, .1], [.2, .8, .3, .9]]], dtype=dtype)
    ids = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.long)
    feature = torch.tensor([[[[1., 3.], [5., -2.]], [[-2., 4.], [1., 3.]]]], dtype=dtype)
    return feature, q, ids


class FeatureContracts(unittest.TestCase):
    def test_sparse_mean_matches_dense_operator_with_unused_region(self):
        feature, q, ids = example(torch.float64)
        token, mass = mean_pool(feature, q, ids, 3)
        dense = torch.zeros(1, 4, 3, dtype=torch.float64)
        for candidate in range(q.shape[1]):
            dense.scatter_add_(2, ids[candidate][None, :, None], q[:, candidate, :, None])
        expected_mass = dense.sum(1)
        expected = dense.transpose(1, 2) @ feature.flatten(2).transpose(1, 2)
        expected /= expected_mass.clamp_min(1)[..., None]
        torch.testing.assert_close(mass, expected_mass)
        torch.testing.assert_close(token, expected)
        torch.testing.assert_close(unpool(token, q, ids, (2, 2)),
                                   (dense @ expected).transpose(1, 2).reshape_as(feature))
        self.assertEqual(float(mass[0, 2]), 0.)
        self.assertTrue(bool((token[0, 2] == 0).all()))

    def test_duplicate_ids_accumulate_into_one_region(self):
        feature, q, ids = example()
        ids.zero_()
        token, mass = mean_pool(feature, q, ids, 2)
        torch.testing.assert_close(mass, torch.tensor([[4., 0.]]))
        torch.testing.assert_close(token[0, 0], feature.mean((0, 2, 3)))
        rebuilt = unpool(token, q, ids, (2, 2))
        torch.testing.assert_close(rebuilt, feature.mean((2, 3), keepdim=True).expand_as(feature))

    def test_tiny_positive_mass_uses_exact_mean_not_parent_epsilon(self):
        feature, _, ids = example()
        tiny = torch.full((1, 4), 1e-9)
        q = torch.stack((1-tiny, tiny), 1)
        token, mass = mean_pool(feature, q, ids, 3)
        self.assertGreater(float(mass[0, 1]), 0.)
        self.assertLess(float(mass[0, 1]), 1e-6)
        torch.testing.assert_close(token[0, 1], feature.mean((0, 2, 3)))
        self.assertTrue(bool(torch.isfinite(token).all()))

    def test_constant_preserved_with_empty_regions(self):
        feature, q, ids = example()
        feature.fill_(7.)
        token, mass = mean_pool(feature, q, ids, 5)
        torch.testing.assert_close(unpool(token, q, ids, (2, 2)), feature)
        self.assertEqual(int((mass == 0).sum()), 3)

    def test_bad_routes_fail_instead_of_being_repaired(self):
        _, q, ids = example()
        mutations = [
            (q*.5, ids, 2, (2, 2)),
            (q, ids-1, 2, (2, 2)),
            (q, ids+2, 2, (2, 2)),
            (q, ids.float(), 2, (2, 2)),
            (q, ids, 2, (1, 2)),
            (q, ids, 0, (2, 2)),
        ]
        for values in mutations:
            with self.subTest(values=[getattr(x, 'shape', x) for x in values]):
                with self.assertRaises(ValueError):
                    validate_routes(*values)
        for bad in (float('nan'), float('inf'), -.1):
            edited = q.clone()
            edited[0, 0, 0] = bad
            with self.assertRaises(ValueError):
                validate_routes(edited, ids, 2, (2, 2))

    def test_nonfinite_feature_fails(self):
        feature, q, ids = example()
        feature[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            mean_pool(feature, q, ids, 2)

    def test_affine_commutation_and_readonly_inputs(self):
        feature, q, ids = example()
        head = nn.Conv2d(2, 4, 1)
        before = [tensor_hash(x) for x in (feature, q, ids, head.weight, head.bias)]
        probs, checks, _ = feature_branches(feature, q, ids, 3, head, (7, 9))
        for key in ('affine_commutation', 'token_affine_commutation',
                    'constant_reconstruction', 'q_normalization'):
            self.assertTrue(checks[key]['passed'], (key, checks[key]))
        self.assertEqual(checks['empty_regions'], 1)
        self.assertEqual(before, [tensor_hash(x) for x in (feature, q, ids, head.weight, head.bias)])
        for probability in probs.values():
            self.assertEqual(tuple(probability.shape), (1, 4, 7, 9))
            self.assertFalse(probability.requires_grad)
            self.assertTrue(bool(torch.isfinite(probability).all()))
            self.assertTrue(bool((probability >= 0).all()))
            torch.testing.assert_close(probability.sum(1), torch.ones(1, 7, 9))

    def test_softmax_before_after_unpool_is_not_assumed_equal(self):
        feature, q, ids = example()
        head = nn.Conv2d(2, 4, 1)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[[[2.]], [[0.]]], [[[0.]], [[2.]]],
                                             [[[-1.]], [[0.]]], [[[0.]], [[-1.]]]]))
            head.bias.zero_()
        probs, _, _ = feature_branches(feature, q, ids, 2, head, (2, 2))
        self.assertGreater(float((probs['feature_mean']-probs['region_mean']).abs().max()), 1e-3)
        self.assertGreater(float((probs['pixel_grid']-probs['grid_probq']).abs().max()), 1e-3)

    def test_native_a1_logit_resize_and_grid_probability_resize_are_distinct(self):
        logits = torch.tensor([[[[5., -1.]], [[0., 0.]], [[-2., 3.]], [[-3., -2.]]]])
        native = F.interpolate(logits, (1, 9), mode='bilinear', align_corners=False).softmax(1)
        grid = resize_probability(logits.softmax(1), (1, 9))
        self.assertGreater(float((native-grid).abs().max()), .05)
        torch.testing.assert_close(native.sum(1), torch.ones(1, 1, 9))
        torch.testing.assert_close(grid.sum(1), torch.ones(1, 1, 9))

    def test_incompatible_classifier_rejected(self):
        feature, q, ids = example()
        for head in (nn.Conv2d(3, 4, 1), nn.Conv2d(2, 3, 1), nn.Conv2d(2, 4, 3),
                     nn.Conv2d(2, 4, 1, stride=2), nn.Linear(2, 4)):
            with self.subTest(head=type(head).__name__):
                with self.assertRaises(ValueError):
                    feature_branches(feature, q, ids, 2, head, (2, 2))

    def test_margin_uses_fractional_gt_area_and_ignores_255(self):
        before = torch.zeros(1, 4, 1, 1)
        after = before.clone()
        before[:, 2] = 2.
        after[:, 2] = -3.
        gt = torch.tensor([[[2, 255], [1, 0]]])
        result = damaged_margin_sums(before, after, gt)
        self.assertEqual(result['damaged_area_feature_cells'], .25)
        self.assertEqual(result['before_margin_sum'], .5)
        self.assertEqual(result['after_margin_sum'], -.75)
        self.assertEqual(result['positive_to_nonpositive_mass'], .25)

    def test_full_transitions_include_fp_and_ignore_labels(self):
        gt = torch.tensor([[[2, 2, 2, 1, 0, 255]]])
        old = torch.tensor([[[2, 1, 3, 1, 2, 99]]])
        new = torch.tensor([[[1, 2, 0, 2, 0, -4]]])
        joint = transition_counts(gt, old, new)
        self.assertEqual(int(joint.sum()), 5)
        self.assertEqual(int(joint[2, 3, 0]), 1)  # wrong -> different wrong is retained
        result = transition_summary(joint)['per_class']['Damaged']
        self.assertEqual(result['harm'], 1)
        self.assertEqual(result['rescue'], 1)
        self.assertEqual(result['net_tp'], result['candidate_tp']-result['reference_tp'])
        self.assertEqual(result['reference_fp'], 1)
        self.assertEqual(result['candidate_fp'], 1)
        gt_d = gt == 2
        self.assertEqual(int(transition_counts(gt, old, new, gt_d).sum()), 3)

    def test_predictions_on_valid_gt_must_be_legal(self):
        gt = torch.tensor([[[2]]])
        with self.assertRaises(ValueError):
            transition_counts(gt, gt, torch.tensor([[[9]]]))
        with self.assertRaises(ValueError):
            transition_counts(gt, gt, gt, torch.ones_like(gt))


class EmbeddingHookContracts(unittest.TestCase):
    def test_optional_embedding_interventions_restore_hooks_and_weights(self):
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_audit import embedding_probability
        from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import SuperpixelEmbedding
        model = nn.Module()
        model.embedding = SuperpixelEmbedding(content_dim=2, adapted_dim=2, dim=4).eval()
        model.region_classifier = nn.Linear(4, 4).eval()
        feature, q, ids = example()
        # Existing parent pool requires exactly nine supports. Padded zero routes
        # remain inert and preserve the two nonzero candidates in this fixture.
        q = torch.cat((q, torch.zeros(1, 7, 4)), 1)
        ids = torch.cat((ids, torch.zeros(7, 4, dtype=torch.long)), 0)
        native = dict(q=q, ids=ids, count=3, shape=(2, 2),
                      adapted_content=feature+.2, content_only=feature.clone())
        original = {k: tensor_hash(v) for k, v in model.state_dict().items()}
        before_hooks = {name: len(m._forward_hooks) for name, m in model.named_modules()}
        baseline = embedding_probability(model, native, feature, (4, 4))
        for mode in ('no_structure', 'uniform_attention', 'no_stats', 'no_geometry'):
            p = embedding_probability(model, native, feature, (4, 4), mode)
            self.assertTrue(bool(torch.isfinite(p).all()))
            torch.testing.assert_close(p.sum(1), torch.ones(1, 4, 4))
            self.assertEqual(before_hooks, {name: len(m._forward_hooks) for name, m in model.named_modules()})
        torch.testing.assert_close(embedding_probability(model, native, feature, (4, 4)), baseline)
        self.assertEqual(original, {k: tensor_hash(v) for k, v in model.state_dict().items()})
        with self.assertRaises(ValueError):
            embedding_probability(model, native, feature, (4, 4), 'invalid')
        self.assertEqual(before_hooks, {name: len(m._forward_hooks) for name, m in model.named_modules()})


if __name__ == '__main__':
    unittest.main()
