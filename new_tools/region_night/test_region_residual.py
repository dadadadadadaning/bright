"""Training and artifact contracts; synthetic data never counts as Val450."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import region_residual as driver
from residual_core import ARMS, FORMAT, make_heads, prepare


def fixtures():
    front = nn.Module()
    front.base = nn.Module()
    front.base.clf_2 = nn.Conv2d(256, 4, 1)
    front.requires_grad_(False).eval()
    feature = torch.randn(1, 256, 2, 2)
    q = torch.zeros(1, 9, 4)
    q[:, 4] = 1
    ids = torch.zeros(9, 4, dtype=torch.long)
    result = dict(q=q, ids=ids, count=1, shape=(2, 2))
    target = torch.tensor([[[0, 1], [2, 255]]])
    return front, feature, result, target


class TrainingContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.front, self.feature, self.result, self.target = fixtures()
        self.heads = make_heads(self.front)
        self.optimizers = {a: torch.optim.AdamW(self.heads[a].parameters(), lr=1e-3) for a in ARMS}
        self.inputs = prepare(self.feature, self.result, self.front.base.clf_2)

    def test_budget_cannot_promote_short_run(self):
        self.assertEqual(driver.training_budget(5000, 1e-4, False), 5000)
        self.assertEqual(driver.training_budget(20, 1e-4, True), 20)
        for steps, lr, smoke in ((20, 1e-4, False), (5000, 1e-4, True),
                                 (True, 1e-4, False), (5000, float('nan'), False)):
            with self.assertRaises(ValueError):
                driver.training_budget(steps, lr, smoke)

    def test_one_update_uses_both_arms_and_leaves_source_frozen(self):
        before = copy.deepcopy(self.front.state_dict())
        driver.validate_optimizers(self.heads, self.optimizers, [self.front])
        logs = driver.matched_update(self.heads, self.optimizers, self.inputs, self.target)
        self.assertEqual(set(logs), set(ARMS))
        self.assertTrue(all(bool((self.heads[a].output.weight != 0).any()) for a in ARMS))
        self.assertTrue(all(torch.equal(v, self.front.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(p.grad is None for p in self.front.parameters()))

    def test_later_nonfinite_loss_advances_neither_optimizer(self):
        before = copy.deepcopy(self.heads.state_dict())
        calls = 0
        def loss(p, y):
            nonlocal calls
            calls += 1
            return driver.supervised_loss(p, y) if calls == 1 else p.sum() * float('nan')
        with self.assertRaises(RuntimeError):
            driver.matched_update(self.heads, self.optimizers, self.inputs, self.target, loss_fn=loss)
        self.assertTrue(all(torch.equal(v, self.heads.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(not opt.state for opt in self.optimizers.values()))

    def test_rejects_optimizer_touching_other_head_and_grad_inputs(self):
        bad = dict(self.optimizers)
        bad[ARMS[1]] = self.optimizers[ARMS[0]]
        with self.assertRaises(ValueError):
            driver.validate_optimizers(self.heads, bad, [self.front])
        bad_inputs = dict(self.inputs, feature=self.inputs['feature'].clone().requires_grad_(True))
        with self.assertRaises(ValueError):
            driver.matched_update(self.heads, self.optimizers, bad_inputs, self.target)

    def test_failed_or_smoke_fit_never_loads_as_formal(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for summary in (dict(format=FORMAT, complete=False),
                            dict(format=FORMAT, complete=True, full_validation_evidence=False, smoke=True)):
                (p/'summary.json').write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    driver.load_fit(p, dict(protocol_sha256='fixture'))

    def test_complete_flag_cannot_replace_e0_q_and_provenance_checks(self):
        result = dict(format=driver.EVAL_FORMAT, complete=True, full_validation_evidence=True,
                      image_count=450, native_a1_replay_ok=True, immutable={'fixture':True},
                      reproduction={'passed':True}, e0_metric_comparison={'passed':True},
                      smoke=False, initial=False, test_accessed=False, q_checks_passed=True,
                      initial_checks_passed=True, protocol_sha256='p', source_sha256='s')
        driver.assert_evaluation(result, protocol_sha='p', code_sha='s', initial=False)
        for key, value in (('format','wrong'), ('q_checks_passed',False),
                           ('e0_metric_comparison',{'passed':False}), ('test_accessed',True),
                           ('protocol_sha256','other'), ('source_sha256','other')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.assert_evaluation(dict(result,**{key:value}), protocol_sha='p', code_sha='s')

    def test_smoke_runs_real_updates_and_keeps_nonformal_artifacts(self):
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1 import memory_audit
        from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1 import native_pool
        from bda_benchmark.experiments.damageformer_srac_fourstage_v2 import label_audit
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'fit'
            args = SimpleNamespace(dataset='synthetic', workers=0, seed=3407,
                                   pool_mode='native', train_list='train_set.txt', val_list='val_set.txt')
            bundle = dict(args=args, device=torch.device('cpu'), front=self.front,
                          metric=nn.Identity(), decoders={}, protocol_sha256='fixture',
                          hashes={}, head_hashes={}, runtime_settings={}, train_names=['a','b'])
            batches = [(self.feature, self.feature, None, self.target, [name]) for name in ['a','b']]
            class Capture:
                def __init__(s, front): pass
                def __call__(s, pre, post): return self.result, pre
                def close(s): pass
            class Guard:
                def __init__(s,*a,**k): s.failures=[]; s.restored=False
                def __enter__(s): return s
                def __exit__(s,*a): s.restored=True
                def set_image(s,name): pass
            def evaluation(bundle, heads, output, **kw):
                self.assertTrue(kw['smoke'])
                Path(output).mkdir()
                return dict(complete=True, full_validation_evidence=False, image_count=2,
                            format=driver.EVAL_FORMAT, smoke=True, initial=kw.get('initial',False),
                            test_accessed=False, q_checks_passed=True, initial_checks_passed=True,
                            initial_parameters={'passed':True},
                            initial_probability_checks={a:{'passed':True,'checked_images':2} for a in ARMS},
                            protocol_sha256='fixture', source_sha256='fixture_code',
                            native_a1_replay_ok=True, immutable={'fixture':True})
            cli = argparse.Namespace(protocol='fixture', output=str(out), steps=20, lr=1e-4, smoke=True,
                                     command='train')
            with patch.object(driver,'load_source',return_value=bundle), \
                 patch.object(driver,'evaluate',side_effect=evaluation), \
                 patch.object(driver,'source_hash',return_value='fixture_code'), \
                 patch.object(driver,'verify_unchanged',return_value={'fixture':True}), \
                 patch.object(memory_audit,'full_image_loader',return_value=batches), \
                 patch.object(memory_audit,'FrozenFrontCapture',Capture), \
                 patch.object(native_pool,'PoolGuard',Guard), \
                 patch.object(label_audit,'validate_batch_labels',return_value={'valid':True}), \
                 patch.object(label_audit,'transfer_checked_batch',side_effect=lambda b,d:b):
                summary = driver.train(cli)
            self.assertTrue(summary['complete'])
            self.assertFalse(summary['full_validation_evidence'])
            self.assertTrue(summary['smoke'])
            self.assertEqual(summary['updates_per_arm'], {a:20 for a in ARMS})
            self.assertEqual(len((out/'training.jsonl').read_text().splitlines()),20)
            payload = torch.load(out/'heads_last.pth', map_location='cpu', weights_only=False)
            self.assertTrue(payload['complete'])
            self.assertTrue(payload['smoke'])
            with self.assertRaises(ValueError):
                driver.load_fit(out,bundle)


if __name__ == '__main__':
    unittest.main()
