"""Synthetic CPU contracts; no benchmark data, GPU, or source mutations."""
import copy
import inspect
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from .core import (ResidualSupportHead, _shuffled_allowed_labels, add_statistics,
                   classify_regions, device_bank, finish_statistics, make_heads, retrieve)


def tiny_bank(dim=8):
    return device_bank(dict(tokens=torch.arange(6 * dim).reshape(6, dim).float() / 20.,
        distribution=F.one_hot(torch.arange(6) % 4, 4).float(),
        images=torch.arange(6), events=torch.tensor([0, 0, 1, 1, 2, 2]),
        image_names=['train_a', 'train_b', 'train_c', 'train_d', 'train_e', 'train_f'],
        event_names=['event_a', 'event_b', 'event_c']), 'cpu')


def activate(head):
    # Tests beyond the identity contract must not pass trivially because the
    # output projection starts at zero.
    with torch.no_grad():
        head.ffn[-1].weight.normal_(0, .1)
        head.ffn[-1].bias.normal_(0, .1)
    return head


class ResidualMemoryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)

    def test_exact_initial_source_identity_in_both_modes(self):
        query, base = torch.randn(5, 8), torch.randn(5, 4)
        support, fractions, valid = retrieve(query, tiny_bank(), -1, -1, topk=4)
        for mode, head in make_heads(8, 'cpu').items():
            logits = head(query, base, support, fractions, valid, mode=mode)
            torch.testing.assert_close(logits, base, rtol=0, atol=0)
            torch.testing.assert_close(head.correction(query, support, fractions, valid, mode),
                                       torch.zeros_like(base), rtol=0, atol=0)

    def test_equal_parameter_initialization_independent_storage(self):
        heads = make_heads(8, 'cpu')
        left, right = heads['nomemory'], heads['memory']
        self.assertEqual(sum(p.numel() for p in left.parameters()),
                         sum(p.numel() for p in right.parameters()))
        for a, b in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        before = copy.deepcopy(left.state_dict())
        with torch.no_grad():
            right.labels.weight.add_(1)
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_zero_initialization_gradient_staging(self):
        head = ResidualSupportHead(8)
        query, base = torch.randn(7, 8), torch.randn(7, 4)
        support, fractions, valid = retrieve(query, tiny_bank(), -1, -1, 4)
        opt = torch.optim.SGD(head.parameters(), lr=.1)
        F.cross_entropy(head(query, base, support, fractions, valid), torch.arange(7) % 4).backward()
        self.assertGreater(float(head.ffn[-1].weight.grad.abs().sum()), 0.)
        self.assertEqual(float(head.attention.in_proj_weight.grad.abs().sum()), 0.)
        self.assertEqual(float(head.labels.weight.grad.abs().sum()), 0.)
        opt.step()
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(head(query, base, support, fractions, valid), torch.arange(7) % 4).backward()
        self.assertGreater(float(head.attention.in_proj_weight.grad.abs().sum()), 0.)
        self.assertGreater(float(head.labels.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_frozen_inputs_and_classifier_are_unchanged_after_backward(self):
        source = torch.nn.Linear(8, 4).eval().requires_grad_(False)
        feature = torch.randn(1, 5, 8, requires_grad=True)
        snapshot = copy.deepcopy(source.state_dict())
        with torch.no_grad():
            tokens = feature.detach().clone()
            base = source(tokens)
        head = activate(ResidualSupportHead(8))
        logits, _ = classify_regions(head, tokens, torch.ones(1, 5), base,
            tiny_bank(), ['query'], ['event_a'], 'memory')
        F.cross_entropy(logits.flatten(0, 1), torch.arange(5) % 4).backward()
        self.assertIsNone(feature.grad)
        self.assertTrue(all(p.grad is None for p in source.parameters()))
        self.assertGreater(float(head.attention.in_proj_weight.grad.abs().sum()), 0.)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)

    def test_nomemory_is_bank_independent_and_uses_no_retrieval(self):
        head = activate(ResidualSupportHead(8)).eval()
        tokens, base, mass = torch.randn(1, 7, 8), torch.randn(1, 7, 4), torch.ones(1, 7)
        with mock.patch('bda_benchmark.experiments.damageformer_region_memory_v2.core.retrieve',
                        side_effect=AssertionError('No-memory must not access the bank')):
            a, _ = classify_regions(head, tokens, mass, base, None, ['q'], ['e'], 'nomemory')
            b, _ = classify_regions(head, tokens, mass, base, {'invalid': True}, ['q'], ['e'], 'nomemory')
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_empty_support_fallback_and_masked_undefined_padding_are_finite(self):
        head = activate(ResidualSupportHead(8))
        query, base = torch.randn(3, 8), torch.randn(3, 4)
        for count in (0, 4):
            support = torch.full((3, count, 8), float('nan'))
            labels = torch.full((3, count, 4), float('nan'))
            valid = torch.zeros((3, count), dtype=torch.bool)
            logits = head(query, base, support, labels, valid)
            self.assertTrue(torch.isfinite(logits).all())
            F.cross_entropy(logits, torch.arange(3)).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_chunk_consistency_and_empty_slot_preservation(self):
        head = activate(ResidualSupportHead(8)).eval()
        tokens, base = torch.randn(2, 7, 8), torch.randn(2, 7, 4)
        mass = torch.tensor([[1., 0., 2., 1., 0., 1., 2.], [0., 1., 1., 1., 1., 0., 1.]])
        for mode in ('memory', 'nomemory'):
            a, x = classify_regions(head, tokens, mass, base, tiny_bank(),
                ['query_a', 'query_b'], ['event_a', 'event_b'], mode, query_chunk=1)
            b, y = classify_regions(head, tokens, mass, base, tiny_bank(),
                ['query_a', 'query_b'], ['event_a', 'event_b'], mode, query_chunk=100)
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(a[mass == 0], base[mass == 0], atol=0, rtol=0)
            for key in ('queries', 'supported', 'valid_neighbors', 'changed_regions', 'scalar_values'):
                self.assertEqual(x[key], y[key])
            self.assertEqual(x['queries'], 10)
        empty, stats = classify_regions(head, tokens, torch.zeros_like(mass), base,
            None, ['q_a', 'q_b'], ['e_a', 'e_b'], 'nomemory')
        torch.testing.assert_close(empty, base, atol=0, rtol=0)
        self.assertEqual(stats['queries'], 0)

    def test_shuffle_labels_fixed_per_image_chunk_invariance(self):
        head = activate(ResidualSupportHead(8)).eval()
        tokens, base, mass = torch.randn(2, 9, 8), torch.randn(2, 9, 4), torch.ones(2, 9)
        results = []
        for chunk in (1, 100):
            logits, _ = classify_regions(head, tokens, mass, base, tiny_bank(),
                ['train_a', 'query_b'], ['event_a', 'event_b'], 'memory',
                query_chunk=chunk, intervention='shuffle_labels',
                generator=torch.Generator().manual_seed(9))
            results.append(logits)
        torch.testing.assert_close(*results, atol=2e-6, rtol=2e-5)
        original = tiny_bank()
        shuffled = _shuffled_allowed_labels(original, 0, 0, True,
                                            torch.Generator().manual_seed(9))
        torch.testing.assert_close(shuffled['distribution'][:2], original['distribution'][:2])
        self.assertIs(shuffled['tokens'], original['tokens'])
        # It is a permutation of allowed labels, not newly sampled labels.
        torch.testing.assert_close(shuffled['distribution'][2:].sum(0),
                                   original['distribution'][2:].sum(0))

    def test_retrieval_excludes_same_image_and_optional_same_event(self):
        bank = tiny_bank()
        query = bank['tokens'][:1]
        support, _, valid = retrieve(query, bank, 0, 0, topk=6)
        self.assertEqual(int(valid.sum()), 4)
        for token in support[valid]:
            self.assertTrue(any(torch.equal(token, row) for row in bank['tokens'][2:]))
        _, _, valid = retrieve(query, bank, 0, 0, topk=6, exclude_same_event=False)
        self.assertEqual(int(valid.sum()), 5)

    def test_query_gt_never_ranks_neighbors(self):
        for function in (ResidualSupportHead.forward, ResidualSupportHead.correction, classify_regions, retrieve):
            self.assertFalse({'gt', 'query_gt', 'query_labels', 'target'} & inspect.signature(function).parameters.keys())
        bank = tiny_bank()
        query = torch.randn(5, 8)
        first = retrieve(query, bank, -1, -1, 3)
        second = retrieve(query, dict(bank, distribution=bank['distribution'].roll(1, 0)), -1, -1, 3)
        torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
        torch.testing.assert_close(first[2], second[2], atol=0, rtol=0)
        self.assertFalse(torch.equal(first[1], second[1]))

    def test_off_intervention_coverage_zero(self):
        head = activate(ResidualSupportHead(8))
        tokens, base = torch.randn(1, 4, 8), torch.randn(1, 4, 4)
        logits, stats = classify_regions(head, tokens, torch.ones(1, 4), base,
            tiny_bank(), ['q'], ['event_a'], 'memory', intervention='off')
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(stats['supported'], 0)
        self.assertEqual(stats['valid_neighbors'], 0)
        self.assertEqual(finish_statistics(stats)['support_coverage'], 0.)

    def test_statistics_keep_fractional_sums(self):
        total = {}
        add_statistics(total, dict(queries=2, scalar_values=8, delta_abs_sum=.125, base_abs_sum=.5))
        add_statistics(total, dict(queries=2, scalar_values=8, delta_abs_sum=.125, base_abs_sum=.5))
        report = finish_statistics(total)
        self.assertEqual(report['delta_abs_sum'], .25)
        self.assertEqual(report['correction_to_base_abs_ratio'], .25)
        self.assertEqual(report['correction_abs_mean'], .25 / 16)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in finish_statistics({}).values()))

    def test_invalid_shape_mode_and_mask_fail(self):
        for dim in (0, 3, 6):
            with self.assertRaises(ValueError):
                ResidualSupportHead(dim)
        head = ResidualSupportHead(8)
        q, base = torch.randn(3, 8), torch.randn(3, 4)
        support, distribution, valid = retrieve(q, tiny_bank(), -1, -1)
        cases = [dict(query=q[:, :7], base_logits=base, mode='nomemory'),
                 dict(query=q, base_logits=base[:, :3], mode='nomemory'),
                 dict(query=q, base_logits=base, mode='bad'),
                 dict(query=q, base_logits=base, support=support,
                      distribution=distribution, valid=valid.float())]
        for kwargs in cases:
            with self.assertRaises(ValueError):
                head(**kwargs)
        with self.assertRaises(ValueError):
            classify_regions(head, q[None], torch.ones(1, 3), base[None], None,
                             ['q'], ['e'], 'memory')

    def test_initial_logits_preserve_full_region_projection(self):
        from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import (
            decode_regions, encode_regions, route_from_q,
        )
        from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
        source = Readout(dim=8).eval().requires_grad_(False)
        q = torch.full((1, 9, 4), 1. / 9)
        ids = torch.tensor([[0, 0, 1, 1]]).expand(9, -1).clone()
        building = torch.tensor([[[[0., 1.], [1., 0.]]]])
        route = route_from_q(q, ids, 2, (2, 2), building)
        with torch.no_grad():
            encoded = encode_regions(source, torch.randn(1, 256, 2, 2), route)
            base = source.classifier(encoded['tokens'])
            expected = decode_regions(base, route, encoded['mass'], (4, 4))
        for mode, head in make_heads(8, 'cpu').items():
            logits, _ = classify_regions(head, encoded['tokens'], encoded['mass'], base,
                tiny_bank(), ['query'], ['event_a'], mode, query_chunk=1)
            actual = decode_regions(logits, route, encoded['mass'], (4, 4))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            F.nll_loss(actual.clamp_min(1e-8).log(), (torch.arange(16).reshape(1, 4, 4) % 4)).backward()
            self.assertTrue(all(p.grad is None for p in source.parameters()))
            self.assertGreater(float(head.ffn[-1].weight.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
