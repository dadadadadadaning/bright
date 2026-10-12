"""Synthetic CPU contracts only; these do not establish CUDA/checkpoint parity."""
import copy
import inspect
import unittest

import torch
from torch.nn import functional as F

from .core import hard_counts, read_tail_fraction, split_corrections, trace_reader
from bda_benchmark.experiments.damageformer_region_memory_v2.core import ResidualSupportHead
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import pool, unpool
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import valid_read_route


def frozen_head(active=True):
    head = ResidualSupportHead(8)
    if active:
        with torch.no_grad():
            head.ffn[-1].weight.normal_(0., .2)
            head.ffn[-1].bias.normal_(0., .05)
    return head.eval().requires_grad_(False)


def reader_inputs():
    query, base = torch.randn(3, 8), torch.randn(3, 4)
    support = torch.randn(3, 4, 8)
    labels = F.one_hot(torch.arange(12).reshape(3, 4) % 4, 4).float()
    valid = torch.tensor([[True, True, True, True],
                          [True, False, True, False],
                          [False, False, False, False]])
    return query, base, support, labels, valid


def tiny_route():
    # Slot 1 in image 0 has positive gather mass but wins no native pixel.
    # Slots 2 and 3 are allocated, strictly empty slots.
    q = torch.tensor([[[.6, .7, .8, .9], [.4, .3, .2, .1], [0., 0., 0., 0.]],
                      [[0., .5, .25, .4], [1., .5, .75, .6], [0., 0., 0., 0.]]])
    ids = torch.arange(3)[None, :, None].expand(2, 3, 4).clone()
    return dict(q=q, ids=ids, k=4, shape=(2, 2))


class ReadoutAuditCPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(73)

    def assertTensorExact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    def test_trace_keeps_real_forward_and_observes_per_head_weights(self):
        head = frozen_head()
        query, base, support, labels, valid = reader_inputs()
        calls, original_outputs = [], []
        before = head.attention.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append(dict(kwargs)), with_kwargs=True)
        after = head.register_forward_hook(
            lambda module, args, output: original_outputs.append(output))
        try:
            trace = trace_reader(head, query, base, support, labels, valid)
        finally:
            before.remove()
            after.remove()
        self.assertEqual(len(original_outputs), 1)
        self.assertIs(trace['logits'], original_outputs[0])
        self.assertEqual([call['need_weights'] for call in calls], [False, True])
        self.assertFalse(calls[1]['average_attn_weights'])
        self.assertEqual(tuple(trace['weights'].shape), (3, head.attention.num_heads, 5))
        with torch.no_grad():
            actual = head(query, base, support, labels, valid, mode='memory')
        self.assertTensorExact(trace['logits'], actual)
        self.assertLessEqual(trace['context_error'], 2e-5)
        self.assertLessEqual(trace['logit_error'], 2e-5)
        self.assertEqual(trace['argmax_disagreements'], 0)

    def test_weights_normalize_over_support_plus_query_and_mask_to_zero(self):
        head = frozen_head()
        query, base, support, labels, valid = reader_inputs()
        # Invalid padding may be undefined without contaminating the reader.
        support[~valid] = float('nan')
        labels[~valid] = float('nan')
        trace = trace_reader(head, query, base, support, labels, valid)
        weights = trace['weights']
        torch.testing.assert_close(weights.sum(-1), torch.ones(3, 4))
        mask = (~valid)[:, None].expand(-1, 4, -1)
        self.assertTensorExact(weights[..., :-1][mask], torch.zeros(int(mask.sum())))
        self.assertTensorExact(weights[2, :, -1], torch.ones(4))
        self.assertTrue(torch.isfinite(trace['logits']).all())
        self.assertTrue((weights[:2, :, -1] > 0).all())

    def test_off_and_empty_support_use_query_fallback_exactly(self):
        head = frozen_head()
        query, base = torch.randn(3, 8), torch.randn(3, 4)
        results = []
        for slots in (0, 4):
            support = torch.full((3, slots, 8), float('nan'))
            labels = torch.full((3, slots, 4), float('nan'))
            valid = torch.zeros(3, slots, dtype=torch.bool)
            trace = trace_reader(head, query, base, support, labels, valid)
            self.assertTensorExact(trace['weights'][..., -1], torch.ones(3, 4))
            self.assertTensorExact(trace['weights'][..., :-1], torch.zeros(3, 4, slots))
            results.append(trace['logits'])
        torch.testing.assert_close(*results, atol=2e-6, rtol=2e-5)
        # Off removes external support, not the learned query-only residual.
        self.assertGreater(float((results[0] - base).abs().max()), 1e-5)

    def test_zero_final_layer_has_attention_but_no_effective_residual(self):
        head = frozen_head(active=False)
        query, base, support, labels, valid = reader_inputs()
        normal = trace_reader(head, query, base, support, labels, valid)
        shuffled = trace_reader(head, query, base, support, labels.roll(1, 1), valid)
        self.assertGreater(float(normal['weights'][0, :, :-1].sum()), 0.)
        self.assertTensorExact(normal['logits'], base)
        self.assertTensorExact(shuffled['logits'], base)
        self.assertTensorExact(normal['weights'], shuffled['weights'])

    def test_nonzero_final_layer_reveals_label_shuffle_effect(self):
        head = frozen_head()
        query, base, support, labels, valid = reader_inputs()
        normal = trace_reader(head, query, base, support, labels, valid)
        shuffled = trace_reader(head, query, base, support, labels.roll(1, 1), valid)
        # Keys, queries and masks are fixed; only support-label values change.
        self.assertTensorExact(normal['weights'], shuffled['weights'])
        self.assertGreater(float((normal['logits'][:2] - shuffled['logits'][:2]).abs().max()), 1e-5)
        self.assertTensorExact(normal['logits'][2], shuffled['logits'][2])

    def test_trace_preserves_inputs_parameters_buffers_rng_and_existing_hooks(self):
        head = frozen_head()
        head.register_buffer('audit_test_sentinel', torch.tensor([2., 7.]))
        inputs = reader_inputs()
        snapshots = [item.clone() for item in inputs]
        state = copy.deepcopy(head.state_dict())
        pointers = {name: value.data_ptr() for name, value in head.state_dict().items()}
        hook_counts = (len(head.attention._forward_pre_hooks), len(head.attention._forward_hooks))
        rng = torch.random.get_rng_state().clone()
        trace = trace_reader(head, *inputs)
        self.assertTensorExact(torch.random.get_rng_state(), rng)
        for actual, expected in zip(inputs, snapshots):
            self.assertTensorExact(actual, expected)
        for name, value in head.state_dict().items():
            self.assertTensorExact(value, state[name])
            self.assertEqual(value.data_ptr(), pointers[name])
        self.assertFalse(head.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in head.parameters()))
        self.assertFalse(trace['logits'].requires_grad)
        self.assertFalse(trace['weights'].requires_grad)
        self.assertEqual(hook_counts, (len(head.attention._forward_pre_hooks), len(head.attention._forward_hooks)))

    def test_trace_rejects_nonfrozen_or_training_head_and_cleans_failed_hooks(self):
        for head in (ResidualSupportHead(8).eval(), frozen_head().train()):
            with self.assertRaises(ValueError):
                trace_reader(head, *reader_inputs())
        head = frozen_head()
        head.ffn[-1].weight.grad = torch.zeros_like(head.ffn[-1].weight)
        with self.assertRaises(ValueError):
            trace_reader(head, *reader_inputs())
        head = frozen_head()
        query, base, support, labels, valid = reader_inputs()
        with self.assertRaises(ValueError):
            trace_reader(head, query, base[:, :3], support, labels, valid)
        self.assertEqual(len(head.attention._forward_pre_hooks), 0)
        self.assertEqual(len(head.attention._forward_hooks), 0)

    def test_hard_counts_cover_shared_and_batched_ids_and_positive_soft_tail(self):
        route = tiny_route()
        before = copy.deepcopy(route)
        counts = hard_counts(route)
        self.assertEqual(counts.dtype, torch.long)
        self.assertTensorExact(counts, torch.tensor([[4, 0, 0, 0], [1, 3, 0, 0]]))
        self.assertTensorExact(counts.sum(-1), torch.full((2,), 4, dtype=torch.long))
        shared = dict(route, ids=route['ids'][0])
        self.assertTensorExact(hard_counts(shared), counts)
        _, mass = pool(torch.ones(2, 4, 1), route['q'], route['ids'], route['k'])
        self.assertGreater(float(mass[0, 1]), 0.)
        self.assertEqual(int(counts[0, 1]), 0)
        for key in ('q', 'ids'):
            self.assertTensorExact(route[key], before[key])

    def test_hard_count_is_candidate_argmax_not_aggregated_slot_argmax(self):
        route = dict(q=torch.tensor([[[.35], [.34], [.31]]]),
                     ids=torch.tensor([[[0], [1], [1]]]), k=2, shape=(1, 1))
        self.assertTensorExact(hard_counts(route), torch.tensor([[1, 0]]))
        _, mass = pool(torch.ones(1, 1, 1), route['q'], route['ids'], 2)
        self.assertGreater(float(mass[0, 1]), float(mass[0, 0]))

    def test_split_changes_only_selected_residual_and_preserves_empty_slots(self):
        route = tiny_route()
        counts = hard_counts(route)
        _, mass = pool(torch.ones(2, 4, 1), route['q'], route['ids'], route['k'])
        base = torch.randn(2, 4, 4)
        # Production normal logits already preserve strictly empty slots.
        normal = base + torch.randn_like(base) * (mass > 0)[..., None]
        snapshots = [tensor.clone() for tensor in (base, normal, counts, route['q'], route['ids'])]
        split = split_corrections(base, normal, counts)
        hard, tail = counts > 0, counts == 0
        self.assertTensorExact(split['hard_only'][hard], normal[hard])
        self.assertTensorExact(split['hard_only'][tail], base[tail])
        self.assertTensorExact(split['tail_only'][hard], base[hard])
        self.assertTensorExact(split['tail_only'][tail], normal[tail])
        for result in split.values():
            self.assertTensorExact(result[mass == 0], base[mass == 0])
        torch.testing.assert_close((split['hard_only'] - base) + (split['tail_only'] - base),
                                   normal - base, atol=0, rtol=0)
        for actual, expected in zip((base, normal, counts, route['q'], route['ids']), snapshots):
            self.assertTensorExact(actual, expected)

    def test_tail_fraction_uses_original_unpool_and_keeps_all_positive_mass_slots(self):
        route = tiny_route()
        counts = hard_counts(route)
        _, mass = pool(torch.ones(2, 4, 1), route['q'], route['ids'], route['k'])
        snapshots = [tensor.clone() for tensor in (route['q'], route['ids'], mass, counts)]
        native = read_tail_fraction(route, mass, counts, (2, 2))
        self.assertTensorExact(native[0], torch.tensor([[.4, .3], [.2, .1]]))
        self.assertTensorExact(native[1], torch.zeros(2, 2))
        valid_route = valid_read_route(route, mass)
        indicator = ((counts == 0) & (mass > 0)).float()
        probabilities = torch.stack((1 - indicator, indicator), -1)
        expected = unpool(probabilities, valid_route['q'], valid_route['ids'], route['shape'], (5, 7))
        actual = read_tail_fraction(route, mass, counts, (5, 7))
        self.assertTensorExact(actual, expected[:, 1])
        torch.testing.assert_close(expected.sum(1), torch.ones(2, 5, 7))
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())
        for current, old in zip((route['q'], route['ids'], mass, counts), snapshots):
            self.assertTensorExact(current, old)

    def test_tail_fraction_renormalizes_after_excluding_strictly_empty_slots(self):
        route = dict(q=torch.tensor([[[.2, .1], [.3, .6], [.5, .3]]]),
                     ids=torch.tensor([[[0, 0], [1, 1], [2, 2]]]), k=3, shape=(1, 2))
        mass = torch.tensor([[1., 1e-12, 0.]])
        counts = torch.tensor([[2, 0, 0]])
        fraction = read_tail_fraction(route, mass, counts, (1, 2))
        torch.testing.assert_close(fraction, torch.tensor([[[.6, 6 / 7]]]))
        self.assertGreater(float(fraction.min()), .5)

    def test_tail_fraction_zero_one_boundaries_and_no_reachable_mass_error(self):
        mass = torch.tensor([[2., .2]])
        counts = torch.tensor([[4, 0]])
        ids = torch.tensor([[[0, 0], [1, 1]]])
        for selected, expected in ((0, 0.), (1, 1.)):
            q = torch.zeros(1, 2, 2)
            q[:, selected] = 1.
            route = dict(q=q, ids=ids, k=2, shape=(1, 2))
            actual = read_tail_fraction(route, mass, counts, (3, 5))
            self.assertTensorExact(actual, torch.full((1, 3, 5), expected))
        with self.assertRaises(ValueError):
            read_tail_fraction(route, torch.zeros_like(mass), counts, (1, 2))

    def test_prediction_and_intervention_interfaces_have_no_query_gt(self):
        banned = {'gt', 'query_gt', 'query_labels', 'target', 'targets',
                  'target_distribution', 'labelled_mass', 'ground_truth'}
        for function in (trace_reader, hard_counts, split_corrections, read_tail_fraction,
                         ResidualSupportHead.forward, ResidualSupportHead.correction):
            with self.subTest(function=function.__name__):
                self.assertFalse(banned & inspect.signature(function).parameters.keys())


if __name__ == '__main__':
    unittest.main()
