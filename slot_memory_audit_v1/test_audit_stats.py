"""Integration contracts for the audit's actual model/statistic field wiring."""
from types import SimpleNamespace
import json
import unittest

import numpy as np
import torch

from audit import analyze_example, selected_cases
from test_reader_pixel import TinyHead, sdpa_reference
from test_slot_stats import fixture


class CallableHead(TinyHead):
    def forward(self, **kwargs):
        if any(k in kwargs for k in ('target', 'query_labels', 'labels', 'GT')):
            raise AssertionError('GT must not enter the classifier')
        query = self._encode_slots(kwargs['slot_f'][0], kwargs['region_t'][0],
                                   kwargs['slot_fraction'][0], kwargs['slot_valid'][0])
        context, _ = sdpa_reference(self, query, kwargs)
        return {'probabilities': kwargs['base_probability_full'],
                'slot_encoding': query[None], 'memory_context': context[None]}


def example_fixture():
    torch.manual_seed(71)
    feature = torch.randn(1, 4, 2, 2)
    building = torch.tensor([[[[.2, .8], [.3, .7]]]])
    target = torch.tensor([[[1, 1], [2, 2]]])
    q = torch.zeros(1, 9, 4)
    q[:, 0] = 1
    ids = torch.tensor([[0, 0, 1, 1]]).expand(9, -1).clone()
    f, q, ids, b, y, slots = fixture(feature, building, target, q, ids, 2)
    common = {'pixel_features': f, 'region_f': torch.randn(1, 2, 4),
              'region_t': torch.randn(1, 2, 4), 'q': q, 'ids': ids,
              'base_probability_full': torch.randn(1, 4, 2, 2).softmax(1)}
    query = {key: slots[key] for key in ('slot_f', 'slot_fraction', 'slot_valid', 'route_q', 'route_ids')}
    local = dict(**query, support_f=slots['slot_f'][0], support_t=common['region_t'][0],
                 support_fraction=slots['slot_fraction'][0], support_valid=slots['slot_valid'][0],
                 support_ids=torch.tensor([[0, 1], [1, 0]]),
                 support_labels=torch.tensor([[[0., .8, .2, 0], [0, .8, .2, 0]],
                                              [[0., .4, .6, 0], [0, .4, .6, 0]]]))
    memory = dict(**query, support_f=torch.randn(3, 2, 4), support_t=torch.randn(3, 4),
                  support_fraction=torch.full((3, 2), .5), support_valid=torch.ones(3, 2, dtype=torch.bool),
                  support_ids=torch.tensor([[0, 1], [1, 2]]),
                  support_labels=torch.tensor([[[0., 1, 0, 0], [0, 1., 0, 0]],
                                               [[0., 0, 1, 0], [0, 0., 1, 0]],
                                               [[0., 0, 0, 1], [0, 0., 0, 1]]]))
    example = SimpleNamespace(image='query_001', event='query-event', target=y,
                              common=common, supports={'b': local, 'd': memory})
    data = dict(feature=f, building=b, result=dict(q=q, ids=ids), slots={'soft': slots})
    heads = {'b': CallableHead().eval(), 'd': CallableHead().eval()}
    return example, data, heads, ['image0', 'image1', 'image2'], ['eventA', 'eventA', 'eventB']


class AuditIntegrationTests(unittest.TestCase):
    def test_real_field_wiring_and_case_export(self):
        example, data, heads, images, events = example_fixture()
        result = analyze_example(example, data, heads, images, events, 1)
        self.assertEqual(set(result['predictions']), {'grid_probq', 'b', 'd'})
        self.assertEqual(result['rows']['memory_valid_parent_count'].tolist(), [2, 2])
        self.assertEqual(result['rows']['local_self_parent_occurrences'].tolist(), [1, 1])
        self.assertTrue(torch.isfinite(result['rows']['reader_d_damaged_label_mass']).all())
        cases = selected_cases(example, result, images, events)
        self.assertGreater(len(cases), 0)
        json.dumps(cases, allow_nan=False)
        self.assertTrue(all(c['actual_parent_budget'] == 2 for c in cases))
        self.assertTrue(all(c['displayed_candidates'] for c in cases))

    def test_changed_gt_changes_only_diagnostic_quantities(self):
        example, data, heads, images, events = example_fixture()
        before = analyze_example(example, data, heads, images, events, 1)
        example.target = torch.full_like(example.target, 3)
        after = analyze_example(example, data, heads, images, events, 1)
        for arm in ('grid_probq', 'b', 'd'):
            np.testing.assert_array_equal(before['predictions'][arm], after['predictions'][arm])
        for name in ('memory_cosine_vote_damaged', 'local_pred_cosine_vote_damaged',
                     'reader_b_damaged_label_mass', 'reader_d_damaged_label_mass'):
            torch.testing.assert_close(before['rows'][name], after['rows'][name], equal_nan=True)
        self.assertFalse(torch.equal(before['slots']['query_labels'], after['slots']['query_labels']))


if __name__ == '__main__':
    unittest.main()
