"""Synthetic CPU contracts; real checkpoint/CUDA parity is a separate gate."""
import copy
import inspect
import unittest

import torch
from torch.nn import functional as F

from .core import CASES, TEMPERATURE, redistribute_external_weights, trace_weight_cases
from bda_benchmark.experiments.damageformer_region_memory_v2.core import ResidualSupportHead


def frozen_head():
    head = ResidualSupportHead(8)
    with torch.no_grad():
        head.ffn[-1].weight.normal_(0., .3)
        head.ffn[-1].bias.normal_(0., .1)
        # Nonzero projection biases catch accidental omission in manual replay.
        head.attention.in_proj_bias.normal_(0., .2)
        head.attention.out_proj.bias.normal_(0., .2)
    return head.eval().requires_grad_(False)


def inputs():
    query, base = torch.randn(3, 8), torch.randn(3, 4)
    support = torch.randn(3, 4, 8)
    labels = F.one_hot(torch.arange(12).reshape(3, 4) % 4, 4).float()
    valid = torch.tensor([[True, True, True, True],
                          [True, False, True, False],
                          [False, False, False, False]])
    cosine = F.cosine_similarity(query[:, None], support, dim=-1)
    return query, base, support, labels, valid, cosine


class WeightAuditContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(109)

    def exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)

    def test_production_forward_retained_and_identity_matches_real_mha(self):
        head, args = frozen_head(), inputs()
        calls, outputs = [], []
        pre = head.attention.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append(dict(kwargs)), with_kwargs=True)
        post = head.register_forward_hook(lambda module, args, output: outputs.append(output))
        try:
            result = trace_weight_cases(head, *args)
        finally:
            pre.remove()
            post.remove()
        self.assertEqual(list(result['traces']), list(CASES))
        self.assertEqual([call['need_weights'] for call in calls], [False, True])
        self.assertFalse(calls[1]['average_attn_weights'])
        self.assertEqual(len(outputs), 1)
        self.assertIs(result['traces']['normal']['logits'], outputs[0])
        self.assertEqual(result['traces']['normal']['weights'].shape, (3, 4, 5))
        self.assertLess(result['replay']['max_abs_context'], 2e-5)
        self.assertLess(result['replay']['max_abs_logits'], 2e-5)
        self.assertEqual(result['replay']['identity_region_argmax_disagreements'], 0)
        self.assertGreater(float((outputs[0] - args[1]).abs().max()), 1e-3)
        self.assertGreater(float((result['traces']['uniform']['logits'] - outputs[0]).abs().max()), 1e-5)

    def test_masked_nan_padding_and_per_head_mass_preserved(self):
        head = frozen_head()
        query, base, support, labels, valid, cosine = inputs()
        support[~valid] = labels[~valid] = cosine[~valid] = float('nan')
        result = trace_weight_cases(head, query, base, support, labels, valid, cosine)
        normal = result['traces']['normal']['weights']
        self.assertGreater(float(normal[:2, :, -1].std(1).max()), 1e-4)
        for case, trace in result['traces'].items():
            with self.subTest(case=case):
                weights = trace['weights']
                self.exact(weights[..., -1], normal[..., -1])
                torch.testing.assert_close(weights[..., :-1].sum(-1), normal[..., :-1].sum(-1))
                torch.testing.assert_close(weights.sum(-1), torch.ones(3, 4))
                self.exact(weights[..., :-1].masked_select(~valid[:, None]), torch.zeros(24))
                self.exact(weights[2, :, -1], torch.ones(4))
                self.assertTrue(torch.isfinite(trace['logits']).all())
        self.assertEqual(result['replay']['max_self_weight_error'], 0)
        self.assertEqual(result['replay']['max_invalid_weight'], 0)

    def test_explicit_per_head_allocations_and_fixed_cosine_temperature(self):
        # Heads intentionally differ: preserving only a head average is wrong.
        weights = torch.tensor([[[.1, .2, 0., .7], [.4, .4, 0., .2]]])
        valid = torch.tensor([[True, True, False]])
        cosine = torch.tensor([[.0, .07, float('nan')]])
        uniform = redistribute_external_weights(weights, valid, cosine, 'uniform')
        expected = torch.tensor([[[.15, .15, 0., .7], [.4, .4, 0., .2]]])
        torch.testing.assert_close(uniform, expected)
        cosine_weights = redistribute_external_weights(weights, valid, cosine, 'cosine')
        relative = torch.tensor([0., 1.]).softmax(-1)
        torch.testing.assert_close(cosine_weights[..., :2], torch.tensor([[[.3], [.8]]]) * relative)
        self.exact(cosine_weights[..., -1], weights[..., -1])
        self.assertEqual(TEMPERATURE, .07)
        self.assertNotIn('temperature', inspect.signature(trace_weight_cases).parameters)

    def test_empty_and_fully_masked_neighbors_keep_query_fallback(self):
        head = frozen_head()
        query, base = torch.randn(3, 8), torch.randn(3, 4)
        normal_outputs = []
        for slots in (0, 4):
            support = torch.full((3, slots, 8), float('nan'))
            labels = torch.full((3, slots, 4), float('nan'))
            valid = torch.zeros(3, slots, dtype=torch.bool)
            cosine = torch.full((3, slots), float('nan'))
            result = trace_weight_cases(head, query, base, support, labels, valid, cosine)
            normal_outputs.append(result['traces']['normal']['logits'])
            for case in CASES:
                trace = result['traces'][case]
                self.exact(trace['weights'][..., -1], torch.ones(3, 4))
                self.exact(trace['weights'][..., :-1], torch.zeros(3, 4, slots))
                torch.testing.assert_close(trace['logits'], normal_outputs[-1])
        torch.testing.assert_close(*normal_outputs)

    def test_one_valid_neighbor_means_no_reallocation(self):
        head = frozen_head()
        query, base, support, labels, valid, cosine = inputs()
        valid[:] = False
        valid[:, 2] = True
        result = trace_weight_cases(head, query, base, support, labels, valid, cosine)
        for case in ('uniform', 'cosine'):
            self.exact(result['traces'][case]['weights'], result['traces']['identity']['weights'])
            self.exact(result['traces'][case]['logits'], result['traces']['identity']['logits'])

    def test_empty_query_batch_returns_empty_traces_without_calling_mha(self):
        head = frozen_head()
        args = [value[:0] for value in inputs()]
        calls = []
        hook = head.attention.register_forward_hook(lambda module, args, output: calls.append(1))
        try:
            result = trace_weight_cases(head, *args)
        finally:
            hook.remove()
        self.assertEqual(calls, [])
        for trace in result['traces'].values():
            self.assertEqual(trace['logits'].shape, (0, 4))
            self.assertEqual(trace['weights'].shape, (0, 4, 5))
        self.assertTrue(all(value == 0 for value in result['replay'].values()))

    def test_support_labels_change_values_but_never_weights(self):
        head = frozen_head()
        query, base, support, labels, valid, cosine = inputs()
        original = trace_weight_cases(head, query, base, support, labels, valid, cosine)
        shuffled = trace_weight_cases(head, query, base, support, labels.roll(1, 1), valid, cosine)
        for case in CASES:
            self.exact(original['traces'][case]['weights'], shuffled['traces'][case]['weights'])
            self.assertGreater(float((original['traces'][case]['logits'][:2] - shuffled['traces'][case]['logits'][:2]).abs().max()), 1e-5)
            self.exact(original['traces'][case]['logits'][2], shuffled['traces'][case]['logits'][2])

    def test_manual_multihead_value_sum_includes_both_biases(self):
        head = frozen_head()
        query, base, support, labels, valid, cosine = inputs()
        result = trace_weight_cases(head, query, base, support, labels, valid, cosine)
        values = torch.where(valid[..., None], support, 0.) + torch.where(valid[..., None], labels, 0.) @ head.labels.weight
        values = torch.cat((values, query[:, None]), dim=1)
        # Independent explicit per-query/per-head reference, including channel
        # concatenation order. The nonzero biases prevent a false zero-bias pass.
        for case in ('identity', 'uniform', 'cosine'):
            contexts = []
            for row in range(3):
                projected = F.linear(values[row], head.attention.in_proj_weight[16:], head.attention.in_proj_bias[16:])
                chunks = []
                for attn_head in range(4):
                    chunks.append((result['traces'][case]['weights'][row, attn_head, :, None] * projected[:, 2 * attn_head:2 * attn_head + 2]).sum(0))
                contexts.append(F.linear(torch.cat(chunks), head.attention.out_proj.weight, head.attention.out_proj.bias))
            expected = base + head.ffn(head.norm(query + torch.stack(contexts)))
            torch.testing.assert_close(result['traces'][case]['logits'], expected)

    def test_inputs_head_rng_and_existing_hooks_are_preserved(self):
        head, args = frozen_head(), inputs()
        snapshots = [value.clone() for value in args]
        state = copy.deepcopy(head.state_dict())
        pointers = {key: value.data_ptr() for key, value in head.state_dict().items()}
        hook = head.attention.register_forward_hook(lambda module, args, output: None)
        counts = (len(head.attention._forward_pre_hooks), len(head.attention._forward_hooks))
        rng = torch.random.get_rng_state().clone()
        try:
            result = trace_weight_cases(head, *args)
            self.assertEqual(counts, (len(head.attention._forward_pre_hooks), len(head.attention._forward_hooks)))
        finally:
            hook.remove()
        self.exact(torch.random.get_rng_state(), rng)
        for current, before in zip(args, snapshots):
            self.exact(current, before)
        for key, value in head.state_dict().items():
            self.exact(value, state[key])
            self.assertEqual(value.data_ptr(), pointers[key])
        for trace in result['traces'].values():
            self.assertFalse(trace['logits'].requires_grad)
            self.assertFalse(trace['weights'].requires_grad)

    def test_identity_also_supports_bias_free_projection(self):
        head = frozen_head()
        head.attention.in_proj_bias = None
        head.attention.out_proj.bias = None
        result = trace_weight_cases(head, *inputs())
        torch.testing.assert_close(result['traces']['identity']['logits'], result['traces']['normal']['logits'])

    def test_reject_train_grad_and_unsupported_mha(self):
        heads = [frozen_head().train(), ResidualSupportHead(8).eval()]
        head = frozen_head()
        head.ffn[-1].weight.grad = torch.zeros_like(head.ffn[-1].weight)
        heads.append(head)
        for attribute, value in (('dropout', .1), ('batch_first', False), ('add_zero_attn', True), ('bias_k', torch.ones(1, 1, 8))):
            head = frozen_head()
            setattr(head.attention, attribute, value)
            heads.append(head)
        for head in heads:
            with self.assertRaises(ValueError):
                trace_weight_cases(head, *inputs())
        head, args = frozen_head(), list(inputs())
        args[0].requires_grad_(True)
        with self.assertRaises(ValueError):
            trace_weight_cases(head, *args)

    def test_reject_nonfinite_valid_inputs_and_clean_hooks_on_failure(self):
        for index in (0, 1, 2, 3, 5):
            args = list(inputs())
            args[index].flatten()[0] = float('nan')
            with self.subTest(index=index), self.assertRaises(ValueError):
                trace_weight_cases(frozen_head(), *args)
        head = frozen_head()
        # Force production to fail after the audit hooks have been registered.
        fail_hook = head.attention.register_forward_pre_hook(
            lambda module, args: (_ for _ in ()).throw(RuntimeError('test failure')))
        counts = (len(head.attention._forward_pre_hooks), len(head.attention._forward_hooks))
        try:
            with self.assertRaises(RuntimeError):
                trace_weight_cases(head, *inputs())
            self.assertEqual(counts, (len(head.attention._forward_pre_hooks), len(head.attention._forward_hooks)))
        finally:
            fail_hook.remove()

    def test_no_query_gt_in_forward_or_weight_interfaces(self):
        banned = {'gt', 'query_gt', 'query_labels', 'target', 'targets', 'ground_truth'}
        for function in (trace_weight_cases, redistribute_external_weights):
            self.assertFalse(banned & inspect.signature(function).parameters.keys())


if __name__ == '__main__':
    unittest.main()
