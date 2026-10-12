"""CPU contracts for dual banks, retrieval isolation and residual identity."""
import copy
import inspect
import unittest

import torch
from torch.nn import functional as F

from .memory import (BANK_FORMAT, MODES, DualQualityReservoir,
                     assert_global_matches_v1, classify_regions, device_bank,
                     finish_statistics, make_heads, retrieve,
                     target_distribution_and_quality, validate_bank)
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import route_from_q


def tiny_route():
    q = torch.zeros(1, 9, 4)
    q[:, 0] = 1
    ids = torch.tensor([[0, 0, 1, 1]]).expand(9, -1).clone()
    building = torch.tensor([[[[0., 1.], [1., 0.]]]])
    return route_from_q(q, ids, 2, (2, 2), building, epsilon=0.)


def tiny_bank(policy='all', dim=8):
    tokens = torch.arange(8 * dim, dtype=torch.float32).reshape(8, dim) / 50
    distribution = torch.eye(4).repeat(2, 1)
    hard_count = torch.ones(8, dtype=torch.long)
    if policy == 'all':
        hard_count[1::2] = 0
    result = dict(
        format=BANK_FORMAT, policy=policy, split='train', test_accessed=False,
        smoke=False, identity={'dim': dim, 'space': 'frozen'},
        image_names=[f'image_{i}' for i in range(8)],
        event_names=['event_a', 'event_b'],
        image_event_ids=[0] * 4 + [1] * 4, per_bucket=1,
        tokens=tokens, distribution=distribution,
        labelled_mass=torch.ones(8), hard_count=hard_count,
        purity=torch.ones(8), entropy=torch.zeros(8),
        images=torch.arange(8), events=torch.tensor([0] * 4 + [1] * 4),
        regions=torch.arange(8))
    return result


class CoreContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(9)

    def test_threshold_free_hard_count_matches_argmax_slot_scatter(self):
        route = tiny_route()
        gt = torch.tensor([[[0, 1], [2, 3]]])
        distribution, labelled, hard, purity, entropy = target_distribution_and_quality(gt, route)
        self.assertEqual(hard.dtype, torch.long)
        self.assertEqual(int(hard.sum()), 4)
        winner = route['q'].argmax(1)
        expected = torch.zeros_like(hard)
        expected.scatter_add_(1, route['ids'].gather(1, winner[:, None]).squeeze(1),
                              torch.ones_like(winner))
        torch.testing.assert_close(hard, expected)
        torch.testing.assert_close(distribution[labelled > 0].sum(-1),
                                   torch.ones(int((labelled > 0).sum())))
        torch.testing.assert_close(purity, distribution.max(-1).values)
        self.assertTrue((entropy >= 0).all())

    def test_dual_reservoir_caps_hard_directly_not_after_all_filter(self):
        reservoir = DualQualityReservoir(per_bucket=2, seed=3)
        tokens = torch.randn(24, 8)
        labels = F.one_hot(torch.arange(24) % 4, 4).float()
        mass = torch.ones(24)
        hard = torch.tensor(([0] * 8) + ([1] * 16))
        purity = torch.ones(24); entropy = torch.zeros(24)
        reservoir.add(tokens, labels, mass, hard, purity, entropy, 0, 0)
        all_bank = reservoir.finish('all', ['a'], ['event'], {'dim': 8}, False)
        hard_bank = reservoir.finish('hard', ['a'], ['event'], {'dim': 8}, False)
        self.assertEqual(len(all_bank['tokens']), 8)
        self.assertEqual(len(hard_bank['tokens']), 8)
        self.assertTrue((hard_bank['hard_count'] > 0).all())
        self.assertTrue(any(row['candidates'] > row['retained']
                            for row in reservoir.counts('hard')))

    def test_bank_validation_is_strict_and_quality_consistent(self):
        for policy in ('all', 'hard'):
            bank = tiny_bank(policy)
            if policy == 'hard':
                bank['hard_count'].fill_(1)
            validate_bank(bank, bank['identity'], False, policy, bank['image_names'])
            corrupt = copy.deepcopy(bank); corrupt['purity'][0] = .2
            with self.assertRaises(ValueError):
                validate_bank(corrupt, bank['identity'], False, policy)
        corrupt = tiny_bank('hard'); corrupt['hard_count'][0] = 0
        with self.assertRaises(ValueError):
            validate_bank(corrupt, corrupt['identity'], False, 'hard')

    def test_global_policy_exactly_matches_v1_and_excludes_sources(self):
        bank = device_bank(tiny_bank('all'), 'cpu')
        query = bank['tokens'][:2]
        assert_global_matches_v1(query, bank, 0, 0, topk=8)
        _, _, valid, meta = retrieve(query, bank, 0, 0, 8)
        self.assertEqual(int(valid.sum()), 8)  # four cross-event rows per query
        self.assertTrue((meta['events'][valid] == 1).all())
        _, _, allowed_same_event, meta = retrieve(
            query, bank, 0, 0, 8, exclude_same_event=False)
        self.assertFalse((meta['images'][allowed_same_event] == 0).any())

    def test_balanced_hard_has_per_class_quota_and_missing_slots_stay_invalid(self):
        bank = device_bank(tiny_bank('hard'), 'cpu')
        query = torch.randn(3, 8)
        _, _, valid, meta = retrieve(query, bank, 0, -1, 8, balanced=True)
        for cls in range(4):
            self.assertTrue(((meta['dominant_class'] == cls) & valid).sum().item() == 3)
        # Remove class 3 from the only allowed event; its two allocated slots
        # remain invalid and are not backfilled by frequent classes.
        bank['events'][bank['dominant_class'] == 3] = 0
        _, _, valid, meta = retrieve(query, bank, 0, -1, 8, balanced=True)
        self.assertFalse((valid & (meta['dominant_class'] == 3)).any())
        self.assertEqual(valid.shape[1], 8)

    def test_four_residual_heads_are_independent_and_exact_identity_at_step0(self):
        heads = make_heads(8, 'cpu')
        hashes = [list(head.state_dict().values()) for head in heads.values()]
        for values in hashes[1:]:
            for a, b in zip(hashes[0], values):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        pointers = [{p.data_ptr() for p in head.parameters()} for head in heads.values()]
        for i in range(len(pointers)):
            for j in range(i + 1, len(pointers)):
                self.assertFalse(pointers[i] & pointers[j])
        query, base = torch.randn(5, 8), torch.randn(5, 4)
        bank = device_bank(tiny_bank('hard'), 'cpu')
        support, labels, valid, _ = retrieve(query, bank, -1, -1, 8)
        for name, head in heads.items():
            logits = (head(query, base, mode='nomemory') if name == 'nomemory'
                      else head(query, base, support, labels, valid, mode='memory'))
            torch.testing.assert_close(logits, base, rtol=0, atol=0)

    def test_classification_is_chunk_invariant_and_reports_class_coverage(self):
        banks = {name: device_bank(tiny_bank(name), 'cpu')
                 for name in ('all', 'hard')}
        head = make_heads(8, 'cpu')['balanced_hard']
        torch.nn.init.normal_(head.ffn[-1].weight)
        tokens, base = torch.randn(1, 6, 8), torch.randn(1, 6, 4)
        mass = torch.ones(1, 6)
        outputs = []
        for chunk in (1, 20):
            logits, stats = classify_regions(
                head, tokens, mass, base, banks, ['query'], ['event_a'],
                'balanced_hard', 8, chunk, True, 'shuffle_labels',
                torch.Generator().manual_seed(22))
            outputs.append(logits)
            finished = finish_statistics(stats)
            self.assertEqual(set(finished['by_class']), set(('Background','Intact','Damaged','Destroyed')))
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)

    def test_off_is_finite_zero_coverage_and_query_gt_absent(self):
        banks = {name: device_bank(tiny_bank(name), 'cpu')
                 for name in ('all', 'hard')}
        head = make_heads(8, 'cpu')['balanced_hard']
        torch.nn.init.normal_(head.ffn[-1].weight)
        logits, stats = classify_regions(
            head, torch.randn(1, 4, 8), torch.ones(1, 4), torch.randn(1, 4, 4),
            banks, ['query'], ['event_a'], 'balanced_hard', 8, 2, True, 'off')
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(finish_statistics(stats)['support_coverage'], 0.)
        forbidden = {'gt', 'target', 'query_gt', 'query_label', 'query_labels'}
        for function in (retrieve, classify_regions):
            self.assertFalse(forbidden & set(inspect.signature(function).parameters))


if __name__ == '__main__':
    unittest.main()
