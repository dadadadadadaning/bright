"""CPU contracts for orchestration, real region pooling/loss and fail-closed gates."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from .common import OutputLock, initial_gate, require_hard
from .core import make_heads, classify_regions
from .train import schedule_factor
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import (
    route_from_q, encode_regions, decode_regions)
from bda_benchmark.experiments.damageformer_roi_followup_v1.test_memory import tiny_bank
from bda_benchmark.experiments.damageformer_roi_followup_v1.memory import device_bank
from bda_benchmark.experiments.damageformer_region_evidence_v1.train_heads import objective


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_pool_read_and_original_loss_can_train_both_residuals(self):
        torch.manual_seed(7)
        embedding = Readout(8).requires_grad_(False).eval()
        feature = torch.randn(1, 256, 2, 2)
        q = torch.full((1, 9, 4), 1/9.)
        ids = torch.tensor([[0, 0, 1, 1]]).expand(9, -1).clone()
        b = torch.tensor([[[[0., 1.], [1., 0.]]]])
        route = route_from_q(q, ids, 2, (2, 2), b, epsilon=0.)
        with torch.no_grad():
            encoded = encode_regions(embedding, feature, route)
            base = embedding.classifier(encoded['tokens'])
            expected = decode_regions(base, route, encoded['mass'], (4, 4))
        before = {k:v.clone() for k,v in embedding.state_dict().items()}
        bank = device_bank(tiny_bank(8), torch.device('cpu'))
        heads = make_heads(8, torch.device('cpu'))
        target = (torch.arange(16).reshape(1,4,4)%4).long()
        for name, head in heads.items():
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
            for step in range(3):
                logits, stats = classify_regions(head, encoded['tokens'], encoded['mass'], base,
                    bank, ['unseen'], ['unseen_event'], name, topk=2, query_chunk=1)
                prob = decode_regions(logits, route, encoded['mass'], (4,4))
                if step == 0:
                    torch.testing.assert_close(prob, expected, atol=0, rtol=0)
                loss = objective(prob, target)
                self.assertTrue(torch.isfinite(loss))
                opt.zero_grad(); loss.backward(); opt.step()
            self.assertGreater(stats['queries'], 0)
        for key, value in embedding.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertTrue(all(p.grad is None for p in embedding.parameters()))

    def test_output_lock_preserves_existing_result_and_other_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'result'
            with OutputLock(out):
                lock = out.with_name('result.lock')
                self.assertTrue(lock.exists())
                with self.assertRaises(FileExistsError):
                    with OutputLock(out):
                        pass
                self.assertTrue(lock.exists())
                out.mkdir()
            self.assertFalse(lock.exists())
            with self.assertRaises(FileExistsError):
                with OutputLock(out):
                    pass

    def test_no_exchange_allowed_and_end_lr_zero_is_intended(self):
        require_hard(SimpleNamespace(source_head='hard', pool_mode='hard', read_mode='hard'))
        with self.assertRaises(ValueError):
            require_hard(SimpleNamespace(source_head='hard', pool_mode='exchange', read_mode='hard'))
        self.assertEqual(schedule_factor(100,5000), 1.)
        self.assertEqual(schedule_factor(5000,5000), 0.)
        self.assertGreater(schedule_factor(2000,5000), schedule_factor(2500,5000))

    def test_gate_persists_failure_and_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = {'metrics': {}}
            report = dict(passed=False, checks=[], warnings=[])
            args = SimpleNamespace(smoke=False)
            with mock.patch('bda_benchmark.experiments.damageformer_region_memory_v2.common.repro.compare_baselines', return_value=report), \
                 mock.patch('bda_benchmark.experiments.damageformer_region_memory_v2.common.source.assert_source'):
                with self.assertRaises(ValueError):
                    initial_gate({'summary': {'final_metrics': {}}}, result, args, out, {})
            saved = json.loads((out/'summary.json').read_text())
            self.assertFalse(saved['complete'])
            self.assertEqual(saved['optimizer_steps'], 0)
            self.assertTrue((out/'initial.json').is_file())


if __name__ == '__main__':
    unittest.main()
