import copy
import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .audit import (
    CASES,
    evidence_row,
    finish_stats,
    retrieve_case,
)
from bda_benchmark.experiments.damageformer_roi_memory_selective_v1.memory import (
    retrieve,
)


def tiny_bank():
    tokens = F.normalize(torch.tensor([
        [1., 0., 0., 0.], [.9, .1, 0., 0.],
        [0., 1., 0., 0.], [0., .9, .1, 0.],
        [0., 0., 1., 0.], [0., 0., .9, .1],
        [0., 0., 0., 1.], [.1, 0., 0., .9],
    ]), dim=-1)
    distribution = torch.eye(4).repeat_interleave(2, 0)
    return dict(
        tokens=tokens, normalized_tokens=tokens,
        distribution=distribution,
        dominant_class=distribution.argmax(-1),
        quality_eligible=torch.ones(8, dtype=torch.bool),
        hard_count=torch.ones(8, dtype=torch.long),
        labelled_mass=torch.ones(8), purity=torch.ones(8),
        entropy=torch.zeros(8), events=torch.arange(8) % 2,
        images=torch.arange(8), regions=torch.arange(8),
        event_names=['e0', 'e1'], image_names=[f'i{x}' for x in range(8)])


class Contracts(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            topk=4, preselect=8, per_class_cap=2,
            min_cosine=-1., temperature=.07)
        self.query = F.normalize(torch.tensor([
            [1., .01, 0., 0.], [0., 1., .01, 0.]]), dim=-1)
        self.group = torch.tensor([0, 1])
        self.bank = tiny_bank()

    def test_unrestricted_wrapper_matches_parent(self):
        for case in CASES:
            if case['same_group']:
                continue
            left_gen = torch.Generator().manual_seed(9)
            right_gen = torch.Generator().manual_seed(9)
            left = retrieve_case(
                self.query, self.group, self.bank, -1, -1, case,
                self.args, left_gen)
            right = retrieve(
                self.query, self.bank, -1, -1, policy=case['policy'],
                topk=4, preselect=8, per_class_cap=2, min_cosine=-1.,
                exclude_same_event=case['exclude_same_event'],
                intervention=case['intervention'], generator=right_gen)
            for a, b in zip(left[:3], right[:3]):
                torch.testing.assert_close(a, b)
            for key in right[3]:
                torch.testing.assert_close(left[3][key], right[3][key])

    def test_same_group_returns_only_matching_slot_parity(self):
        case = next(row for row in CASES if row['same_group'])
        _, _, valid, metadata = retrieve_case(
            self.query, self.group, self.bank, -1, -1, case,
            self.args, torch.Generator().manual_seed(3))
        matched = metadata['regions'] % 2 == self.group[:, None]
        self.assertTrue(bool((matched | ~valid).all()))

    def test_gt_conditioned_metrics_include_required_transitions(self):
        target = torch.tensor([
            [0., 0., 1., 0.], [0., 0., 0., 1.]])
        labels = torch.tensor([
            [[0., 1., 0., 0.], [0., 0., 1., 0.]],
            [[0., 0., 1., 0.], [0., 1., 0., 0.]],
        ])
        valid = torch.ones((2, 2), dtype=torch.bool)
        metadata = dict(
            dominant_class=labels.argmax(-1),
            cosine=torch.tensor([[1., 0.], [1., 0.]]),
            regions=torch.tensor([[0, 2], [1, 3]]))
        row = evidence_row(
            target, torch.ones(2), torch.ones(2, dtype=torch.long),
            torch.tensor([0, 1]), labels, valid, metadata, .07)
        result = finish_stats(row)
        self.assertEqual(result['per_class']['Damaged']['queries'], 1)
        self.assertEqual(result['per_class']['Destroyed']['queries'], 1)
        self.assertAlmostEqual(
            result['per_class']['Damaged']['same_class_hit_at_k'], 1.)
        self.assertAlmostEqual(
            result['per_class']['Damaged']['posterior_to_class']['Intact'], 1.)
        self.assertAlmostEqual(
            result['per_class']['Destroyed']['posterior_to_class']['Damaged'], 1.)
        self.assertAlmostEqual(result['same_predicted_group_neighbor_fraction'], 1.)

    def test_evidence_does_not_mutate_inputs(self):
        bank = tiny_bank()
        before = copy.deepcopy(bank)
        case = next(row for row in CASES if row['same_group'])
        retrieve_case(self.query, self.group, bank, -1, -1, case,
                      self.args, torch.Generator().manual_seed(1))
        for key, value in before.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, bank[key])

    def test_retrieved_identity_and_injected_label_are_separate(self):
        target = torch.tensor([[0., 0., 1., 0.]])
        injected = torch.tensor([[[0., 1., 0., 0.]]])
        metadata = dict(
            dominant_class=torch.tensor([[2]]),
            cosine=torch.tensor([[1.]]),
            regions=torch.tensor([[1]]))
        row = evidence_row(
            target, torch.ones(1), torch.ones(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long), injected,
            torch.ones((1, 1), dtype=torch.bool), metadata, .07)
        result = finish_stats(row)
        self.assertEqual(result['support_class_matrix'][2][2], 1)
        self.assertEqual(result['injected_label_class_matrix'][2][1], 1)
        self.assertAlmostEqual(
            result['per_class']['Damaged']['same_class_hit_at_k'], 1.)
        self.assertAlmostEqual(
            result['per_class']['Damaged']['posterior_to_class']['Intact'], 1.)


if __name__ == '__main__':
    unittest.main()

