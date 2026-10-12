"""Synthetic contracts only; real BRIGHT/CUDA/full-Val are not replaced by these tests."""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .quota import partial_quotas, priority, ordered_rows, validate_audit_summary, evidence_gate


class QuotaContracts(unittest.TestCase):
    def test_fixed_budget_floor_capacity(self):
        quotas, floor = partial_quotas([0, 36, 49, 81, 16492, 17765], 400, 16)
        self.assertEqual(sum(quotas), 400)
        self.assertEqual(floor, 16)
        self.assertEqual(quotas[0], 0)
        for q, cap in zip(quotas, [0, 36, 49, 81, 16492, 17765]):
            self.assertLessEqual(q, cap)
            self.assertGreaterEqual(q, min(cap, floor))
        self.assertGreater(quotas[-1], 64)

    def test_floor_is_reduced_when_infeasible(self):
        quotas, floor = partial_quotas([1, 100, 100], 5, 16)
        self.assertEqual((quotas, floor), ([1, 2, 2], 2))

    def test_empty_and_capacity_boundaries(self):
        self.assertEqual(partial_quotas([0, 0], 0)[0], [0, 0])
        self.assertEqual(partial_quotas([2, 4, 6], 12)[0], [2, 4, 6])
        self.assertEqual(partial_quotas([], 0)[0], [])
        for cap, total in (([1], 2), ([1], -1), ([-1, 2], 0)):
            with self.assertRaises(ValueError):
                partial_quotas(cap, total)

    def test_deterministic_ties_and_all_small_budgets(self):
        self.assertEqual(partial_quotas([100, 100, 100], 5, 0)[0], [2, 2, 1])
        cap = [0, 1, 3, 7, 19]
        for total in range(sum(cap) + 1):
            quotas, _ = partial_quotas(cap, total, 4)
            self.assertEqual(sum(quotas), total)
            self.assertTrue(all(0 <= q <= n for q, n in zip(quotas, cap)))

    def test_shared_priorities_and_nested_selection(self):
        rows = [('image_' + str(i // 7), i % 7) for i in range(49)]
        order = ordered_rows(rows, 3407)
        self.assertEqual(order, ordered_rows(list(reversed(rows)), 3407))
        self.assertEqual(order[:8], ordered_rows(rows, 3407)[:8])
        self.assertTrue(set(order[:8]).issubset(order[:17]))
        self.assertNotEqual(order, ordered_rows(rows, 3408))
        self.assertEqual(priority(1, 'a', 0), priority(1, 'a', 0))

    def test_full_gate_does_not_accept_smoke_or_partial(self):
        good = dict(complete=True, gates_passed=True, full_validation_evidence=True,
                    smoke=False, test_accessed=False, image_count=450, reproduction=dict(passed=True),
                    input_files_unchanged=True, weights_buffers_unchanged=True,
                    numerical_execution_unchanged=True)
        validate_audit_summary(good)
        for key in ('complete', 'gates_passed', 'full_validation_evidence'):
            with self.assertRaises(ValueError):
                validate_audit_summary(dict(good, **{key: False}))
        for key in ('smoke', 'test_accessed', 'diagnostic_subset'):
            with self.assertRaises(ValueError):
                validate_audit_summary(dict(good, **{key: True}))
        for change in (dict(image_count=408), dict(reproduction=dict(passed=False)),
                       dict(failure={'message': 'failed'}), dict(input_files_unchanged=False),
                       dict(pool_guard={'restored': False})):
            with self.assertRaises(ValueError):
                validate_audit_summary(dict(good, **change))

    def test_explicit_hypothesis_required(self):
        with self.assertRaises(ValueError):
            evidence_gate(argparse.Namespace(smoke=False, confirm_quota_test=False, evidence_note=''))
        self.assertIsNone(evidence_gate(argparse.Namespace(smoke=True))['passed'])

    def test_payload_copies_soft_labels_and_row_identity(self):
        from .build_banks import selected_payload
        from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            d = torch.tensor([[.1, .1, .7, .1], [.2, .65, .1, .05]])
            shard = dict(distribution=d, event=0, tokens=torch.randn(2, 8).half(),
                         geometry=torch.rand(2, 6).half(), weights=torch.ones(2), mass=torch.ones(2))
            torch.save(shard, root / 'shard.pt')
            bundle = dict(cache_root=root, cache=dict(shards=[dict(path='shard.pt', sha256=digest(root/'shard.pt'))], min_purity=0),
                          bank_cpu=dict(event_names=['turkey-earthquake']), hashes=dict(front_sha256='f'))
            bank = selected_payload([(0, 2, 0, 0)], bundle)
            self.assertTrue(torch.equal(bank['distribution'], d[:1]))
            self.assertTrue(torch.equal(bank['tokens'], shard['tokens'][:1]))
            self.assertEqual(bank['images'].tolist(), [0])
            self.assertEqual(bank['regions'].tolist(), [0])
            self.assertNotEqual(bank['distribution'].tolist(), [[0., 0., 1., 0.]])

    def test_same_capacity_initialization_and_frozen_evidence_training(self):
        from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import JointContextDecoder
        from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import gather_support
        from .train_quota import decode
        torch.manual_seed(7)
        classifier = torch.nn.Conv2d(256, 4, 1).requires_grad_(False)
        template = JointContextDecoder(classifier, dim=128, num_events=2, slots=2)
        heads = {arm: copy.deepcopy(template) for arm in ('j1', 'm0', 'm1')}
        counts = {sum(p.numel() for p in head.parameters()) for head in heads.values()}
        self.assertEqual(len(counts), 1)
        feature = torch.randn(1, 256, 2, 2)
        result = dict(tokens=torch.randn(1, 2, 128), q=torch.tensor([[[.8,.2,.8,.2],[.2,.8,.2,.8]]]),
                      ids=torch.tensor([[0,0,0,0],[1,1,1,1]]), shape=(2,2))
        bank = dict(tokens=torch.randn(4, 128), distribution=torch.softmax(torch.randn(4,4), -1),
                    geometry=torch.rand(4,5), events=torch.tensor([0,0,1,1]), disasters=torch.zeros(4,dtype=torch.long))
        support = gather_support(torch.tensor([[2,3],[0,1]]), bank)
        fixed = {k: v.clone() for k,v in result.items() if torch.is_tensor(v)}
        a1 = classifier(feature).detach()
        for arm, head in heads.items():
            output = decode(head, arm, feature, result, None if arm == 'j1' else support)
            torch.testing.assert_close(output['logits'], a1, atol=0, rtol=0)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
            loss = torch.nn.functional.cross_entropy(output['logits'], torch.tensor([[[0,1],[2,3]]]))
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
            opt.step()
        self.assertTrue(all(p.grad is None for p in classifier.parameters()))
        self.assertTrue(all(torch.equal(result[k], v) for k,v in fixed.items()))

    def test_synthetic_training_driver_smoke_never_claims_full_validation(self):
        from . import train_quota as module
        from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import JointContextDecoder
        from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture

        class Front(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Module()
                self.base.clf_2 = torch.nn.Conv2d(256, 4, 1)
                self.register_buffer('tokens', torch.randn(1, 2, 128))
                self.register_buffer('q', torch.full((1, 9, 4), 1/9))
                self.register_buffer('ids', torch.arange(9).remainder(2)[:, None].expand(9, 4).clone())

            def forward(self, pre, post):
                logits = self.base.clf_2(pre)
                return dict(old_pixel=logits, prob=logits.softmax(1), tokens=self.tokens,
                            q=self.q, ids=self.ids, shape=(2,2))

        torch.manual_seed(19)
        front = Front().eval().requires_grad_(False)
        old = {arm: JointContextDecoder(front.base.clf_2, dim=128, num_events=2, slots=2).eval().requires_grad_(False)
               for arm in ('j1', 'j2', 'j3')}
        bank = dict(tokens=torch.randn(4,128), distribution=torch.softmax(torch.randn(4,4),-1),
                    geometry=torch.rand(4,5), events=torch.tensor([0,0,1,1]), images=torch.tensor([0,0,1,1]),
                    regions=torch.tensor([0,1,0,1]), disasters=torch.zeros(4,dtype=torch.long),
                    event_names=['turkey-earthquake', 'ukraine-conflict'])
        name = 'turkey-earthquake_00000001'
        batch = (torch.randn(1,256,2,2), torch.zeros(1,3,2,2), torch.tensor([[[0,1],[1,1]]]),
                 torch.tensor([[[0,1],[2,3]]]), [name])
        bundle = dict(front=front, cache=dict(config=dict(dim=128)), bank=bank, metric=torch.nn.Identity(),
                      decoders=old, topk=2, train_names=[name], val_names=[name], reference={}, hashes={},
                      head_hashes={}, reference_hashes={}, input_paths={}, inputs_before={}, code_before={},
                      source_manifest=dict(format='synthetic'))
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(smoke=True, steps=20, eval_every=20, workers=0, lr=1e-4,
                                      seed=3407, device='cpu', output=str(Path(directory)/'train'), dataset='synthetic',
                                      pool_mode='grouped')
            with patch.object(module, 'load_bundle', return_value=bundle), \
                    patch.object(module, 'read_banks', return_value=({'m0': copy.deepcopy(bank), 'm1': copy.deepcopy(bank)}, {}, {})), \
                    patch.object(module, 'full_image_loader', return_value=[batch]), \
                    patch.object(module, 'diagnostic_code_fingerprint', return_value={}):
                module.run(args)
            root = Path(args.output)
            summary = json.loads((root/'summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertFalse(summary['full_validation_evidence'])
            self.assertFalse(summary['run_complete'])
            self.assertTrue(summary['pool_guard']['restored'])
            self.assertEqual(summary['train_image_counts'], {'j1':20, 'm0':20, 'm1':20})
            self.assertEqual(summary['label_evidence_temperature'], .1)
            self.assertIn('NOT actual Reader attention', summary['label_evidence_definition'])
            self.assertEqual(summary['final_difficult_cohorts']['m0'], summary['final_difficult_cohorts']['m1'])
            self.assertEqual(summary['final_difficult_retrieval']['m0'], summary['final_difficult_retrieval']['m1'])
            cohort = summary['final_difficult_cohorts']['m0']['damaged_j1_wrong']
            self.assertEqual(cohort['count'], 1)
            self.assertLessEqual(cohort['evidence_damaged_and_final_damaged'], cohort['evidence_damaged'])
            self.assertLessEqual(cohort['final_damaged'], cohort['count'])
            row = json.loads((root/'validation.jsonl').read_text().splitlines()[-1])
            self.assertIn('turkey-earthquake', row['per_event_difficult_cohorts'])
            self.assertIn('damaged_hit_mass_sum', row['difficult_retrieval']['m0']['damaged_j1_wrong'])
            self.assertEqual(row['eligible_bank_by_query_event']['turkey-earthquake']['m0']['regions'], 2)
            state0 = torch.load(root/'m0_last.pth', weights_only=False)['model']
            state1 = torch.load(root/'m1_last.pth', weights_only=False)['model']
            self.assertTrue(all(torch.equal(state0[k], state1[k]) for k in state0))


if __name__ == '__main__':
    unittest.main()
