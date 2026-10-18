"""Reader reconstruction and paired error contracts, independently referenced."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pixel_stats import PATHS, PixelAccumulator, check_replay, metrics
from reader_stats import add_posthoc_label_scores, reader_diagnostics


class TinyHead(nn.Module):
    """Two nonidentical heads; same field interface as the actual trained head."""

    def __init__(self):
        super().__init__()
        self.hidden, self.heads = 4, 2
        self.query_projection = nn.Linear(4, 4)
        self.key_projection = nn.Linear(4, 4)
        self.value_projection = nn.Linear(4, 4)
        self.label_embedding = nn.Linear(4, 4, bias=False)
        self.context_projection = nn.Linear(4, 4, bias=False)
        self.register_buffer('roles', torch.tensor([[.1, -.2, .3, -.4], [-.2, .4, -.6, .8]]))

    def _encode_slots(self, f, parent, fraction, valid):
        f = torch.where(valid[..., None], f, 0.)
        parent = torch.where(valid.any(-1)[..., None], parent, 0.)
        encoded = f + .2 * parent[:, None] + self.roles + fraction[..., None]
        return torch.where(valid[..., None], encoded, 0.)


@torch.no_grad()
def sdpa_reference(head, query, support):
    """Use PyTorch's independent SDPA kernel, not audit softmax/einsum code."""
    r, _, hidden = query.shape
    heads, dim = head.heads, hidden // head.heads
    if support['support_valid'].shape[0] == 0:
        return query * 0., None
    table = head._encode_slots(support['support_f'], support['support_t'],
                               support['support_fraction'], support['support_valid'])
    table_k = head.key_projection(table)
    clean_labels = torch.where(support['support_valid'][..., None], support['support_labels'], 0.)
    table_v = head.value_projection(table + head.label_embedding(clean_labels))
    ids = support['support_ids']
    safe = ids.clamp_min(0)
    allowed = ((ids[..., None] >= 0) & support['support_valid'][safe]).flatten(1)
    tokens = allowed.shape[1]
    q = head.query_projection(query).reshape(r, 2, heads, dim).transpose(1, 2)
    k = table_k[safe].reshape(r, tokens, heads, dim).transpose(1, 2)
    v = table_v[safe].reshape(r, tokens, heads, dim).transpose(1, 2)
    # PyTorch releases differ for a row with no allowed keys: older SDPA
    # returns NaN, newer SDPA returns zero. The real Reader explicitly returns
    # zero for that row. Keep this independent oracle portable by never asking
    # SDPA to normalize an empty set, rather than sanitizing its result.
    supported = allowed.any(1)
    context = q.new_zeros((r, heads, 2, dim))
    attention = q.new_zeros((r, heads, 2, tokens))
    if bool(supported.any()):
        mask = allowed[supported, None, None]
        context[supported] = F.scaled_dot_product_attention(
            q[supported], k[supported], v[supported], attn_mask=mask, dropout_p=0.)
        # An identity value table independently obtains each head's weights.
        identity = torch.eye(tokens, device=q.device, dtype=q.dtype).reshape(
            1, 1, tokens, tokens).expand(int(supported.sum()), heads, -1, -1)
        attention[supported] = F.scaled_dot_product_attention(
            q[supported], k[supported], identity, attn_mask=mask, dropout_p=0.)
    context = head.context_projection(context.transpose(1, 2).reshape(r, 2, hidden))
    context = torch.where(support['slot_valid'][0, ..., None], context, 0.)
    return context, attention


def reader_fixture(empty=False):
    torch.manual_seed(9)
    head = TinyHead().eval()
    query = torch.randn(3, 2, 4)
    valid = torch.tensor([[True, True], [True, True], [True, False]])
    m = 0 if empty else 3
    support = {
        'slot_valid': valid[None],
        'support_ids': torch.full((3, 2), -1, dtype=torch.long) if empty else
                       torch.tensor([[0, 1], [-1, -1], [2, 0]]),
        'support_f': torch.randn(m, 2, 4),
        'support_t': torch.randn(m, 4),
        'support_fraction': torch.empty(0, 2) if empty else
                            torch.tensor([[.3, .7], [.005, .995], [0., 1.]]),
        'support_valid': torch.empty(0, 2, dtype=torch.bool) if empty else
                         torch.tensor([[True, True], [True, True], [False, True]]),
        'support_labels': torch.empty(0, 2, 4) if empty else
                          torch.tensor([[[1., 0, 0, 0], [0, 1., 0, 0]],
                                        [[0, 1., 0, 0], [0, 0, 1., 0]],
                                        [[0, 0, 0, 0], [0, 0, 0, 1.]]]),
    }
    context, attention = sdpa_reference(head, query, support)
    result = {'slot_encoding': query[None], 'memory_context': context[None]}
    return head, result, support, attention


class ReaderTests(unittest.TestCase):
    def test_sdpa_oracle_never_normalizes_an_empty_candidate_set(self):
        real_sdpa = F.scaled_dot_product_attention

        def checked_sdpa(query, key, value, *, attn_mask, dropout_p):
            self.assertTrue(bool(attn_mask.any(-1).all()))
            return real_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p)

        with patch.object(F, 'scaled_dot_product_attention', side_effect=checked_sdpa) as mocked:
            # A single fixture mixes valid, padding-only, and invalid-query rows.
            head, result, support, attention = reader_fixture()
            self.assertEqual(mocked.call_count, 2)
            self.assertTrue(torch.isfinite(result['memory_context']).all())
            self.assertTrue(torch.equal(attention[1], torch.zeros_like(attention[1])))
            rows = reader_diagnostics(head, result, support)
            self.assertEqual(rows['has_support'].tolist(), [1., 0., 0.])

            # Even with a nonempty table, all query selections can be padding.
            support['support_ids'].fill_(-1)
            context, attention = sdpa_reference(head, result['slot_encoding'][0], support)
            self.assertEqual(mocked.call_count, 2)
            self.assertTrue(torch.equal(context, torch.zeros_like(context)))
            self.assertTrue(torch.equal(attention, torch.zeros_like(attention)))
            result['memory_context'] = context[None]
            rows = reader_diagnostics(head, result, support)
            self.assertEqual(rows['has_support'].tolist(), [0., 0., 0.])

    def test_nonempty_sdpa_nonfinite_result_still_fails_reconstruction(self):
        real_sdpa = F.scaled_dot_product_attention

        def corrupt_supported_row(query, key, value, *, attn_mask, dropout_p):
            output = real_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p)
            output[0] = float('nan')
            return output

        with patch.object(F, 'scaled_dot_product_attention', side_effect=corrupt_supported_row):
            head, result, support, _ = reader_fixture()
        with self.assertRaisesRegex(ValueError, 'reconstruction differs'):
            reader_diagnostics(head, result, support)

    def test_reconstruction_and_exposure_match_independent_two_head_sdpa(self):
        head, result, support, attention = reader_fixture()
        rows = reader_diagnostics(head, result, support, chunk_size=1)
        # First query has four valid tokens. Two heads genuinely attend differently.
        self.assertFalse(torch.allclose(attention[0, 0, 1], attention[0, 1, 1]))
        att = attention[0, :, 1].mean(0)
        labels = support['support_labels'][support['support_ids'][0]].reshape(-1, 4)
        expected = att @ labels
        actual = torch.stack([rows[k + '_label_mass'][0] for k in
                              ('background', 'intact', 'damaged', 'destroyed')])
        torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(rows['background_role_attention'][0].item(), att[::2].sum().item(), places=6)
        self.assertAlmostEqual(rows['tiny_fraction_attention'][0].item(), att[2].item(), places=6)
        self.assertAlmostEqual(rows['effective_slots'][0].item(), (1 / att.square().sum()).item(), places=5)
        self.assertAlmostEqual(actual.sum().item(), 1., places=6)

    def test_padding_only_selection_and_invalid_query_have_no_exposure(self):
        head, result, support, _ = reader_fixture()
        rows = reader_diagnostics(head, result, support)
        self.assertEqual(rows['has_support'].tolist(), [1., 0., 0.])
        for name, values in rows.items():
            if name != 'has_support':
                self.assertTrue(torch.isnan(values[1:]).all())
        self.assertTrue(torch.equal(result['memory_context'][0, 1], torch.zeros(2, 4)))

    def test_genuinely_empty_support_table(self):
        head, result, support, _ = reader_fixture(empty=True)
        rows = reader_diagnostics(head, result, support)
        self.assertEqual(rows['has_support'].sum().item(), 0)
        self.assertTrue(torch.isnan(rows['effective_slots']).all())
        support['support_ids'][0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'empty support table'):
            reader_diagnostics(head, result, support)

    def test_chunk_size_does_not_change_statistics(self):
        head, result, support, _ = reader_fixture()
        one = reader_diagnostics(head, result, support, chunk_size=1)
        all_at_once = reader_diagnostics(head, result, support, chunk_size=128)
        for name in one:
            torch.testing.assert_close(one[name], all_at_once[name], equal_nan=True)

    def test_query_gt_only_changes_posthoc_label_distance(self):
        head, result, support, _ = reader_fixture()
        original = reader_diagnostics(head, result, support)
        baseline = {k: v.clone() for k, v in original.items()}
        labels1 = torch.zeros(3, 2, 4)
        labels1[:, :, 1] = 1
        labels2 = torch.zeros(3, 2, 4)
        labels2[:, :, 2] = 1
        first = add_posthoc_label_scores({k: v.clone() for k, v in original.items()}, labels1,
                                         torch.ones(3, 2, dtype=torch.bool))
        second = add_posthoc_label_scores({k: v.clone() for k, v in original.items()}, labels2,
                                          torch.ones(3, 2, dtype=torch.bool))
        for name, values in baseline.items():
            torch.testing.assert_close(first[name], values, equal_nan=True)
            torch.testing.assert_close(second[name], values, equal_nan=True)
        self.assertNotAlmostEqual(first['posthoc_label_TV'][0].item(), second['posthoc_label_TV'][0].item())
        self.assertTrue(torch.isnan(first['posthoc_label_TV'][1:]).all())

    def test_reconstruction_failure_is_not_silently_ignored(self):
        head, result, support, _ = reader_fixture()
        result['memory_context'] = result['memory_context'] + .05
        with self.assertRaisesRegex(ValueError, 'reconstruction differs'):
            reader_diagnostics(head, result, support)


def pixel_fixture():
    target = np.array([[1, 2, 2, 0], [3, 1, 255, 2]], dtype=np.int64)
    predictions = {
        'grid_probq': np.array([[1, 0, 2, 0], [1, 2, 3, 1]], dtype=np.int64),
        'b': np.array([[2, 2, 1, 0], [3, 1, 0, 2]], dtype=np.int64),
        'd': np.array([[1, 2, 2, 2], [3, 2, 0, 1]], dtype=np.int64),
    }
    masks = {'all': np.ones_like(target, dtype=bool), 'damaged': target == 2,
             'selected': np.array([[True, True, False, False], [False, False, True, False]])}
    return target, predictions, masks


class PixelTests(unittest.TestCase):
    def test_confusions_and_overlapping_strata_ignore_255(self):
        acc = PixelAccumulator()
        acc.add(*pixel_fixture())
        report = acc.report()
        expected_d = [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 0, 1]]
        self.assertEqual(report['all']['metrics']['d']['confusion'], expected_d)
        self.assertEqual(report['all']['metrics']['d']['valid_pixels'], 7)
        self.assertEqual(report['damaged']['metrics']['d']['valid_pixels'], 3)
        self.assertEqual(report['selected']['metrics']['d']['valid_pixels'], 2)
        self.assertAlmostEqual(report['all']['metrics']['d']['foreground_miou'], (1 / 3 + .4 + 1.) / 3)

    def test_paired_rescue_harm_are_same_pixel_transitions(self):
        acc = PixelAccumulator()
        acc.add(*pixel_fixture())
        paired = acc.report()['all']['paired']
        self.assertEqual(paired['b_vs_grid']['rescue_by_gt_class'], [0, 1, 2, 1])
        self.assertEqual(paired['b_vs_grid']['harm_by_gt_class'], [0, 1, 1, 0])
        self.assertEqual(paired['d_vs_b']['rescue_by_gt_class'], [0, 1, 1, 0])
        self.assertEqual(paired['d_vs_b']['harm_by_gt_class'], [1, 1, 1, 0])
        self.assertEqual(paired['d_vs_b']['intact_correct_to_damaged'], 1)
        self.assertEqual(paired['b_vs_grid']['damaged_background_to_correct'], 1)
        self.assertEqual(paired['b_vs_grid']['damaged_intact_to_correct'], 1)
        self.assertEqual(paired['b_vs_grid']['damaged_correct_to_intact'], 1)

    def test_split_aggregation_is_not_mean_of_image_ious(self):
        acc = PixelAccumulator()
        acc.add(np.array([[2]]), {k: np.array([[2]]) for k in PATHS}, {'all': np.array([[True]])})
        acc.add(np.full((1, 9), 2), {k: np.ones((1, 9), dtype=np.int64) for k in PATHS},
                {'all': np.ones((1, 9), dtype=bool)})
        m = acc.report()['all']['metrics']['d']
        self.assertAlmostEqual(m['iou'][2], .1)
        self.assertAlmostEqual(m['foreground_miou'], .1 / 3)
        self.assertIsNone(m['iou'][3])

    def test_replay_checks_percent_point_tolerance_and_confusion(self):
        acc = PixelAccumulator()
        acc.add(*pixel_fixture())
        actual = acc.report()['all']['metrics']
        reference = {'metrics': copy.deepcopy(actual)}
        exact = check_replay(actual, reference)
        self.assertTrue(exact['passed'])
        self.assertTrue(all(x['exact_confusion'] for x in exact['paths'].values()))
        reference['metrics']['d']['iou'][2] += .001  # 0.1 percentage point
        failed = check_replay(actual, reference, tolerance_pp=.02)
        self.assertFalse(failed['passed'])
        self.assertAlmostEqual(failed['paths']['d']['max_iou_delta_pp'], .1)

    def test_replay_rejects_class_mismatch_malformed_and_nonfinite(self):
        m = metrics(np.eye(4, dtype=np.int64))
        actual = {p: copy.deepcopy(m) for p in PATHS}
        for mutate, pattern in (
            (lambda x: x['iou'].pop(), 'exactly four'),
            (lambda x: x['iou'].__setitem__(2, float('nan')), 'finite'),
            (lambda x: x.__setitem__('foreground_miou', float('inf')), 'finite'),
            (lambda x: x['iou'].__setitem__(2, None), 'presence'),
            (lambda x: x.__setitem__('confusion', [[1, 0]]), 'confusion'),
        ):
            reference = {'metrics': copy.deepcopy(actual)}
            mutate(reference['metrics']['d'])
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                check_replay(actual, reference)
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            check_replay(actual, {'metrics': actual}, tolerance_pp=float('nan'))

    def test_invalid_gt_and_stratum_masks_are_rejected(self):
        target, preds, masks = pixel_fixture()
        masks['all'] = masks['all'].astype(np.float32)
        with self.assertRaisesRegex(ValueError, 'pixel mask'):
            PixelAccumulator().add(target, preds, masks)
        target[0, 0] = 7
        with self.assertRaisesRegex(ValueError, 'GT class'):
            PixelAccumulator().add(target, preds, {'all': np.ones_like(target, dtype=bool)})


if __name__ == '__main__':
    unittest.main()
