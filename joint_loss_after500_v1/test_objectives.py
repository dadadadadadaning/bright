"""CPU contracts for loss semantics; no BRIGHT data or CUDA claims."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from objectives import objective_terms, serialized_terms


def fixture_original(probability, target):
    """Standalone reference semantics, independent of objective_terms helpers."""
    flat = probability.permute(0, 2, 3, 1).reshape(-1, 4)
    labels = target.long().reshape(-1)
    mask = labels != 255
    if not bool(mask.any()):
        return probability.sum() * 0
    flat, labels = flat[mask], labels[mask]
    nll = F.nll_loss(flat.float().clamp_min(1e-8).log(), labels)
    present = []
    for cls in range(4):
        fg = (labels == cls).float()
        if not bool(fg.sum()):
            continue
        errors, ordering = (fg - flat[:, cls]).abs().sort(descending=True)
        fg = fg[ordering]
        total = fg.sum()
        jaccard = 1 - (total - fg.cumsum(0)) / (total + (1 - fg).cumsum(0)).clamp_min(1e-6)
        gradient = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
        present.append((errors * gradient).sum())
    return nll + torch.stack(present).mean()


def source_original_if_available():
    """Read actual functions via AST without importing training dependencies.

    Delivery can run entirely with the standalone fixture. When BRIGHT_REPO (or
    /workspace/bright) is available, additionally exercise the exact source.
    """
    repo = Path(os.environ.get('BRIGHT_REPO', '/workspace/bright'))
    source = repo / 'bda_benchmark/script/standard_ML/train_S5_S8_GPU.py'
    losses = repo / 'bda_benchmark/experiments/damageformer_srac_fourstage_v2/losses.py'
    if not source.is_file() or not losses.is_file():
        return None
    functions = []
    for path, names in [(source, {'lovasz_grad', 'flatten_probas', 'lovasz_softmax_flat', 'lovasz_softmax'}),
                        (losses, {'pixel_table', 'region_nll'})]:
        tree = ast.parse(path.read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in selected} != names:
            raise AssertionError(f'Reference functions missing from {path}')
        functions.extend(selected)
    from typing import List, Tuple
    scope = dict(torch=torch, F=F, List=List, Tuple=Tuple)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<BRIGHT source loss functions>', 'exec'), scope)
    return lambda p, t: scope['region_nll'](p, t) + scope['lovasz_softmax'](p, t, classes='present', ignore=255)


class ObjectiveContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(391)

    def random_case(self):
        logits = torch.randn(2, 4, 3, 5, requires_grad=True)
        labels = torch.arange(30).reshape(2, 3, 5) % 4
        labels[0, 1, 1] = 255
        return logits, labels

    def assert_reference_parity(self, reference):
        logits, labels = self.random_case()
        probability = logits.softmax(1)
        terms = objective_terms(probability, labels, reference_loss=reference, verify_reference=True)
        expected = reference(probability, labels)
        torch.testing.assert_close(terms['loss'], expected, rtol=1e-6, atol=1e-7)
        actual_grad = torch.autograd.grad(terms['loss'], logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)

    def test_standalone_original_value_and_gradient_parity(self):
        self.assert_reference_parity(fixture_original)

    def test_exact_source_original_value_and_gradient_parity_when_available(self):
        reference = source_original_if_available()
        if reference is None:
            self.skipTest('Optional source checkout unavailable; standalone reference is tested')
        self.assert_reference_parity(reference)

    def test_contributions_sum_and_json_are_finite(self):
        logits, labels = self.random_case()
        terms = objective_terms(logits.softmax(1), labels)
        torch.testing.assert_close(terms['class_nll_contribution'].sum(), terms['nll'])
        torch.testing.assert_close(terms['class_lovasz_contribution'].sum(), terms['lovasz'])
        self.assertEqual(terms['class_valid_counts'].sum().item(), terms['valid_count'])
        self.assertEqual(terms['valid_count'], 29)
        json.dumps(serialized_terms(terms), allow_nan=False)

    def test_candidate_removes_only_background_gradient(self):
        logits, labels = self.random_case()
        terms = objective_terms(logits.softmax(1), labels)
        torch.testing.assert_close(terms['loss'] - terms['candidate_loss'], terms['background_lovasz_contribution'])
        candidate_grad = torch.autograd.grad(terms['candidate_loss'], logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(terms['loss'] - terms['background_lovasz_contribution'], logits)[0]
        torch.testing.assert_close(candidate_grad, expected_grad, rtol=2e-6, atol=1e-7)

    def test_same_original_denominator_with_two_present_classes(self):
        probability = torch.tensor([[[[.65, .10]], [[.10, .10]], [[.20, .70]], [[.05, .10]]]], requires_grad=True)
        labels = torch.tensor([[[0, 2]]])
        terms = objective_terms(probability, labels)
        self.assertEqual(terms['present_count'], 2)
        torch.testing.assert_close(terms['candidate_lovasz'], terms['class_lovasz_contribution'][2])
        # Damaged raw Lovasz is .3, so retained contribution is .3/2, not .3.
        self.assertAlmostEqual(terms['candidate_lovasz'].item(), .15, places=6)
        self.assertEqual(terms['class_lovasz_contribution'][1].item(), 0.)
        self.assertEqual(terms['class_lovasz_contribution'][3].item(), 0.)

    def test_no_background_means_objectives_equal(self):
        logits = torch.randn(1, 4, 2, 3, requires_grad=True)
        labels = torch.tensor([[[1, 2, 3], [2, 1, 3]]])
        terms = objective_terms(logits.softmax(1), labels)
        torch.testing.assert_close(terms['loss'], terms['candidate_loss'])
        self.assertEqual(terms['background_lovasz_contribution'].item(), 0.)

    def test_all_background_candidate_still_has_nll_supervision(self):
        probability = torch.tensor([.5, .2, .2, .1]).reshape(1, 4, 1, 1).requires_grad_()
        labels = torch.zeros(1, 1, 1, dtype=torch.long)
        terms = objective_terms(probability, labels)
        self.assertEqual(terms['candidate_lovasz'].item(), 0.)
        self.assertAlmostEqual(terms['nll'].item(), .69314718, places=6)
        self.assertAlmostEqual(terms['lovasz'].item(), .5, places=6)
        terms['candidate_loss'].backward()
        self.assertLess(probability.grad[0, 0, 0, 0].item(), 0.)

    def test_background_pixels_remain_foreground_lovasz_negatives(self):
        probability = torch.tensor([[[[.10, .10]], [[.05, .10]], [[.80, .70]], [[.05, .10]]]], requires_grad=True)
        terms = objective_terms(probability, torch.tensor([[[0, 2]]]))
        grad = torch.autograd.grad(terms['candidate_lovasz'], probability)[0]
        self.assertGreater(grad[0, 2, 0, 0].item(), 0., 'Damaged false positive on background must be penalized')

    def test_ignore_pixels_have_zero_gradients_for_both_objectives(self):
        probability = torch.randn(1, 4, 1, 3).softmax(1).detach().requires_grad_()
        terms = objective_terms(probability, torch.tensor([[[0, 255, 2]]]))
        for name in ['loss', 'candidate_loss']:
            grad = torch.autograd.grad(terms[name], probability, retain_graph=True)[0]
            self.assertTrue(torch.equal(grad[:, :, :, 1], torch.zeros_like(grad[:, :, :, 1])))

    def test_all_ignore_is_finite_differentiable_zero(self):
        probability = torch.randn(1, 4, 2, 2).softmax(1).detach().requires_grad_()
        terms = objective_terms(probability, torch.full((1, 2, 2), 255))
        self.assertEqual(terms['valid_count'], 0)
        self.assertEqual(terms['present_count'], 0)
        for name in ['loss', 'candidate_loss']:
            self.assertEqual(terms[name].item(), 0.)
            grad = torch.autograd.grad(terms[name], probability, retain_graph=True)[0]
            self.assertTrue(torch.equal(grad, torch.zeros_like(grad)))
        json.dumps(serialized_terms(terms), allow_nan=False)

    def test_invalid_labels_rejected_before_nll(self):
        probability = torch.full((1, 4, 1, 1), .25)
        for label in [-1, 4, 254]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                objective_terms(probability, torch.tensor([[[label]]]))
        with self.assertRaises(ValueError):
            objective_terms(probability, torch.tensor([[[1.5]]]))

    def test_nll_clamp_and_fp32_match_original(self):
        probability = torch.tensor([[[[0., .2]], [[.4, .3]], [[.3, .3]], [[.3, .2]]]], dtype=torch.float16, requires_grad=True)
        labels = torch.tensor([[[0, 2]]])
        terms = objective_terms(probability, labels)
        self.assertEqual(terms['nll'].dtype, torch.float32)
        torch.testing.assert_close(terms['loss'], fixture_original(probability, labels))

    def test_scalar_reference_preserves_original_control_graph(self):
        logits, labels = self.random_case()
        probability = logits.softmax(1)
        reference = fixture_original(probability, labels)
        terms = objective_terms(probability, labels, reference_loss=reference, verify_reference=True)
        self.assertIs(terms['loss'], reference)
        self.assertIsNot(terms['candidate_loss'], reference)

    def test_callable_reference_is_no_grad_and_only_runs_when_requested(self):
        calls = []
        def reference(p, target):
            calls.append(torch.is_grad_enabled())
            return fixture_original(p, target)
        logits, labels = self.random_case()
        probability = logits.softmax(1)
        objective_terms(probability, labels, reference_loss=reference)
        self.assertEqual(calls, [])
        objective_terms(probability, labels, reference_loss=reference, verify_reference=True)
        self.assertEqual(calls, [False])

    def test_reference_mismatch_fails_closed(self):
        logits, labels = self.random_case()
        with self.assertRaises(RuntimeError):
            objective_terms(logits.softmax(1), labels,
                            reference_loss=lambda p, t: fixture_original(p, t) + .01,
                            verify_reference=True)
        with self.assertRaises(ValueError):
            objective_terms(logits.softmax(1), labels, verify_reference=True)


if __name__ == '__main__':
    unittest.main()
