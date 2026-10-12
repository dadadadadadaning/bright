"""Read-only retrieval diagnostic contracts, synthetic CPU only."""
import copy
import json
import unittest
from unittest import mock

import torch

from .core import device_bank
from .diagnostics import add_quality, finish_quality, retrieval_quality
from bda_benchmark.experiments.damageformer_roi_followup_v1.memory import retrieve


def tiny_inputs():
    # Four one-pixel regions: retain nine-support routing interface, first
    # support alone nonzero. Region labels are 0,1,2,3, not building instances.
    q = torch.zeros(1, 9, 4)
    q[:, 0] = 1
    ids = torch.arange(4)[None, None].expand(1, 9, -1).clone()
    route = dict(q=q, ids=ids, k=4, shape=(2, 2))
    tokens = torch.eye(4)[None]
    bank = device_bank(dict(tokens=torch.eye(4), distribution=torch.eye(4),
        images=torch.arange(4), events=torch.tensor([0, 0, 1, 2]),
        image_names=['a', 'b', 'c', 'd'], event_names=['e0', 'e1', 'e2']), 'cpu')
    return tokens, torch.ones(1, 4), route, torch.arange(4).reshape(1, 2, 2), bank


class RetrievalDiagnosticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_feature_ranking_precedes_gt_and_does_not_depend_on_gt(self):
        inputs = tiny_inputs()
        neighbors = []
        from bda_benchmark.experiments.damageformer_region_memory_v2 import diagnostics
        original_targets = diagnostics.target_distribution
        for flip in (False, True):
            calls = []
            def ranked(*args, **kwargs):
                calls.append('retrieve')
                result = retrieve(*args, **kwargs)
                neighbors.append(result[0].clone())
                return result
            def labelled(*args, **kwargs):
                self.assertTrue(calls)
                calls.append('target')
                return original_targets(*args, **kwargs)
            tokens, mass, route, gt, bank = inputs
            with mock.patch.object(diagnostics, 'retrieve', ranked), mock.patch.object(diagnostics, 'target_distribution', labelled):
                retrieval_quality(tokens, mass, route, 3-gt if flip else gt, bank,
                                  ['unknown'], ['unknown'], topk=1, query_chunk=2)
            self.assertEqual(calls, ['retrieve', 'retrieve', 'target'])
        for a, b in zip(neighbors[:2], neighbors[2:]):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_exclusion_changes_eligible_prior_and_is_label_independent(self):
        tokens, mass, route, gt, bank = tiny_inputs()
        row = retrieval_quality(tokens, mass, route, gt, bank, ['a'], ['e0'], topk=4)
        report = finish_quality(row)
        self.assertEqual(report['all']['mean_valid_neighbors'], 2.)
        for name, expected in zip(('Background', 'Intact', 'Damaged', 'Destroyed'), (0., 0., .5, .5)):
            self.assertEqual(report['by_class'][name]['eligible_bank_query_class_prior'], expected)
        row = retrieval_quality(tokens, mass, route, gt, bank, ['a'], ['e0'], topk=4,
                                exclude_same_event=False)
        self.assertEqual(finish_quality(row)['all']['mean_valid_neighbors'], 3.)
        self.assertEqual(finish_quality(row)['by_class']['Background']['eligible_bank_query_class_prior'], 0.)

    def test_no_mutations_no_gradients_and_json_serializable(self):
        tokens, mass, route, gt, bank = tiny_inputs()
        tokens.requires_grad_(True)
        snapshot = copy.deepcopy((tokens.detach(), mass, route, gt, bank))
        raw = retrieval_quality(tokens, mass, route, gt, bank, ['q'], ['q'], topk=1)
        report = finish_quality(raw)
        self.assertEqual(report['all']['neighbor_vote_agreement'], 1.)
        self.assertEqual(report['all']['neighbor_query_class_mass'], 1.)
        self.assertEqual(report['all']['eligible_bank_query_class_prior'], .25)
        self.assertIsNone(tokens.grad)
        for value, expected in zip((tokens, mass, gt), (snapshot[0], snapshot[1], snapshot[3])):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        for name, value in route.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, snapshot[2][name], atol=0, rtol=0)
        for name, value in bank.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, snapshot[4][name], atol=0, rtol=0)
        json.dumps(raw, allow_nan=False)
        json.dumps(report, allow_nan=False)

    def test_none_valid_empty_mass_and_ignored_gt(self):
        tokens, mass, route, gt, bank = tiny_inputs()
        bank['events'] = torch.zeros_like(bank['events'])
        mass[:, 0] = 0
        gt[0, 0, 1] = 255
        report = finish_quality(retrieval_quality(tokens, mass, route, gt, bank,
                                                  ['q'], ['e0'], topk=4))
        self.assertEqual(report['all']['queries'], 2)
        self.assertEqual(report['all']['supported_queries'], 0)
        self.assertIsNone(report['all']['neighbor_query_class_mass'])
        self.assertIsNone(report['all']['eligible_bank_query_class_prior'])
        json.dumps(report, allow_nan=False)

    def test_additive_counts_chunk_independent(self):
        inputs = tiny_inputs()
        first = retrieval_quality(*inputs, ['q'], ['q'], topk=2, query_chunk=1)
        second = retrieval_quality(*inputs, ['q'], ['q'], topk=2, query_chunk=100)
        self.assertEqual(first, second)
        total = {}
        add_quality(total, first)
        add_quality(total, second)
        self.assertEqual(total['images'], 2)
        for name in total['classes']:
            for key in total['classes'][name]:
                self.assertEqual(total['classes'][name][key], 2*first['classes'][name][key])
        self.assertEqual(finish_quality(total)['all']['neighbor_query_class_mass'],
                         finish_quality(first)['all']['neighbor_query_class_mass'])
        json.dumps(finish_quality({}), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
