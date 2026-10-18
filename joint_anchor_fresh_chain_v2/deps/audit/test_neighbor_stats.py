import unittest

import torch

from neighbor_stats import inventory, neighbor_diagnostics


class NeighborStatsTests(unittest.TestCase):
    def inputs(self):
        # Four parents, with two from one source image; both roles are visible.
        f = torch.tensor([[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]], [[1., 1.], [1., 1.]]])
        gt = torch.zeros(3, 2, 4)
        gt[0, :, 1] = 1
        gt[1, :, 2] = 1
        gt[2, :, 3] = 1
        mf = torch.tensor([[[1., 0.], [1., 0.]], [[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]], [[1., 1.], [1., 1.]]])
        my = torch.zeros(4, 2, 4)
        my[0, :, 0] = 1
        my[1, :, 2] = 1
        my[2, :, 1] = 1
        my[3, :, 3] = 1
        return dict(query_f=f, query_labels=gt, query_valid=torch.ones(3, 2, dtype=torch.bool),
                    memory_ids=torch.tensor([[0, 1, 2], [0, 1, 2], [3, -1, -1]]),
                    memory_f=mf, memory_labels=my, memory_valid=torch.ones(4, 2, dtype=torch.bool),
                    memory_images=['one', 'one', 'two', 'three'], memory_events=['A', 'A', 'B', 'B'],
                    local_ids=torch.tensor([[0, 1, 2], [1, 0, 2], [2, -1, -1]]),
                    local_labels_pred=gt.clone(), local_valid=torch.ones(3, 2, dtype=torch.bool))

    def test_counts_dedup_sources_and_both_roles(self):
        result = neighbor_diagnostics(**self.inputs())
        rows = result['region_rows']
        self.assertEqual(rows['memory_valid_parent_count'].tolist(), [3, 3, 1])
        self.assertEqual(rows['memory_valid_child_slot_count'].tolist(), [6, 6, 2])
        self.assertEqual(rows['memory_unique_source_images'].tolist(), [2, 2, 1])
        self.assertEqual(rows['memory_unique_source_events'].tolist(), [2, 2, 1])
        self.assertAlmostEqual(rows['memory_duplicate_image_fraction'][0].item(), 1/3, places=6)
        self.assertEqual(result['inventory']['valid_child_slot_count'], 8)
        self.assertEqual(result['inventory']['unique_source_images'], 3)

    def test_no_support_is_nan_not_zero_accuracy(self):
        args = self.inputs()
        args['memory_ids'].fill_(-1)
        rows = neighbor_diagnostics(**args)['region_rows']
        self.assertTrue(torch.equal(rows['memory_valid_parent_count'], torch.zeros(3)))
        for key in ('memory_cosine_vote_damaged', 'memory_posthoc_vote_tv_compatibility', 'memory_candidate_max_damaged', 'memory_duplicate_image_fraction'):
            self.assertTrue(torch.isnan(rows[key]).all(), key)

    def test_empty_bank_and_zero_budget(self):
        args = self.inputs()
        args['memory_f'] = args['memory_f'][:0]
        args['memory_labels'] = args['memory_labels'][:0]
        args['memory_valid'] = args['memory_valid'][:0]
        args['memory_images'] = []
        args['memory_events'] = []
        args['memory_ids'] = torch.empty((3, 0), dtype=torch.long)
        args['local_ids'] = torch.empty((3, 0), dtype=torch.long)
        result = neighbor_diagnostics(**args)
        self.assertEqual(result['inventory']['valid_parent_count'], 0)
        self.assertTrue(torch.isnan(result['region_rows']['memory_posthoc_best_candidate_tv_compatibility']).all())

    def test_query_gt_cannot_change_real_votes_or_selection(self):
        args = self.inputs()
        before = neighbor_diagnostics(**args)['region_rows']
        args['query_labels'] = args['query_labels'].roll(1, dims=-1)
        after = neighbor_diagnostics(**args)['region_rows']
        for key in before:
            if 'posthoc' not in key and 'oracle' not in key:
                torch.testing.assert_close(before[key], after[key], equal_nan=True)
        self.assertFalse(torch.equal(before['memory_posthoc_vote_tv_compatibility'], after['memory_posthoc_vote_tv_compatibility']))

    def test_local_self_probe_explicit_without_refill(self):
        rows = neighbor_diagnostics(**self.inputs())['region_rows']
        self.assertEqual(rows['local_self_parent_occurrences'].tolist(), [1, 1, 1])
        self.assertEqual(rows['local_other_pred_valid_parent_count'].tolist(), [2, 2, 0])
        self.assertTrue(torch.isnan(rows['local_other_pred_cosine_vote_damaged'][2]))

    def test_zero_norm_is_not_structural_absence(self):
        args = self.inputs()
        args['query_f'][0] = 0
        rows = neighbor_diagnostics(**args)['region_rows']
        self.assertEqual(rows['memory_valid_child_slot_count'][0], 6)
        self.assertEqual(rows['memory_cosine_valid_slot_count'][0], 0)
        self.assertTrue(torch.isnan(rows['memory_cosine_vote_damaged'][0]))
        self.assertFalse(torch.isnan(rows['memory_posthoc_best_candidate_tv_compatibility'][0]))

    def test_missing_gt_does_not_remove_prediction(self):
        args = self.inputs()
        args['query_labels'][0] = 0
        rows = neighbor_diagnostics(**args)['region_rows']
        self.assertFalse(torch.isnan(rows['memory_cosine_vote_damaged'][0]))
        self.assertTrue(torch.isnan(rows['memory_posthoc_vote_tv_compatibility'][0]))

    def test_invalid_slot_is_ignored_even_with_nan_payload(self):
        args = self.inputs()
        args['memory_valid'][0, 0] = False
        args['memory_f'][0, 0] = torch.nan
        args['memory_labels'][0, 0] = torch.nan
        rows = neighbor_diagnostics(**args)['region_rows']
        self.assertEqual(rows['memory_valid_child_slot_count'][0], 5)
        self.assertTrue(torch.isfinite(rows['memory_cosine_vote_damaged']).all())

    def test_valid_distribution_validation(self):
        args = self.inputs()
        args['memory_labels'][0, 0] = torch.tensor([1.1, -.1, 0., 0.])
        with self.assertRaises(ValueError):
            neighbor_diagnostics(**args)
        args = self.inputs()
        args['local_labels_pred'][0, 0] *= .5
        with self.assertRaises(ValueError):
            neighbor_diagnostics(**args)

    def test_foreground_conditional_is_explicit_and_background_not_quality(self):
        args = self.inputs()
        args['memory_labels'][:] = torch.tensor([.9, 0, .1, 0])
        args['query_labels'][:] = torch.tensor([.1, 0, .9, 0])
        rows = neighbor_diagnostics(**args)['region_rows']
        torch.testing.assert_close(rows['memory_posthoc_foreground_conditional_vote_tv'], torch.ones(3))
        torch.testing.assert_close(rows['memory_posthoc_vote_tv_compatibility'], torch.full((3,), .2))
        torch.testing.assert_close(rows['memory_candidate_foreground_conditional_mean_damaged'], torch.ones(3))
        self.assertTrue((rows['memory_posthoc_vote_soft_dot_not_accuracy'] < 1).all())

    def test_padding_and_ids_bounds(self):
        args = self.inputs()
        args['memory_ids'][0, 0] = -2
        with self.assertRaises(ValueError):
            neighbor_diagnostics(**args)
        args = self.inputs()
        args['memory_ids'][0, 0] = 4
        with self.assertRaises(ValueError):
            neighbor_diagnostics(**args)
        args = self.inputs()
        args['local_ids'] = args['local_ids'][:, :1]
        with self.assertRaises(ValueError):
            neighbor_diagnostics(**args)

    def test_chunk_independence(self):
        args = self.inputs()
        a = neighbor_diagnostics(**args, chunk_size=1)['region_rows']
        b = neighbor_diagnostics(**args, chunk_size=32)['region_rows']
        for key in a:
            torch.testing.assert_close(a[key], b[key], equal_nan=True)

    def test_duplicate_parent_and_invalid_parent_counts(self):
        args = self.inputs()
        args['memory_ids'][0] = torch.tensor([0, 0, 3])
        args['memory_valid'][3] = False
        rows = neighbor_diagnostics(**args)['region_rows']
        self.assertEqual(rows['memory_valid_parent_count'][0], 2)
        self.assertEqual(rows['memory_unique_parent_count'][0], 1)
        self.assertEqual(rows['memory_duplicate_parent_fraction'][0], .5)
        self.assertEqual(rows['memory_duplicate_image_fraction'][0], 0)

    def test_similar_label_exposure_not_query_gt(self):
        rows = neighbor_diagnostics(**self.inputs())['region_rows']
        self.assertEqual(rows['memory_has_similar_damaged50'][0], 1)
        self.assertEqual(rows['memory_has_similar_intact50'][0], 0)
        self.assertEqual(rows['memory_has_similar_intact50'][1], 1)


if __name__ == '__main__':
    unittest.main()
