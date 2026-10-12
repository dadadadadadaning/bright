import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from .core import Pooler, SupportHead, retrieve, metrics


class Contracts(unittest.TestCase):
    def test_padding_cannot_change_pool(self):
        torch.manual_seed(1)
        m = Pooler(6, 8, True).eval()
        m.alpha.data.fill_(0.5)
        x = torch.randn(2, 4, 6)
        mask = torch.tensor([[True, True, False, False]] * 2)
        a = m(x, mask)[0]
        x[:, 2:] = 10000
        torch.testing.assert_close(a, m(x, mask)[0])

    def test_single_token_backward_finite(self):
        m = Pooler(6, 8, True)
        x = torch.randn(2, 1, 6)
        m(x, torch.ones(2, 1, dtype=torch.bool))[1].sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_excludes_same_event_and_image(self):
        q = torch.tensor([[1., 0.]])
        bank = torch.tensor([[1., 0.], [.9, .1], [.7, .3]])
        ix = retrieve(q, bank, ['a'], ['x'], ['a', 'b', 'c'], ['z', 'x', 'y'], 1)
        self.assertEqual(ix.item(), 2)
        with self.assertRaises(ValueError):
            retrieve(q, bank, ['a'], ['x'], ['a','b','c'], ['z','x','y'], 2)

    def test_e2_and_e3_all_parameters_participate(self):
        for memory in (False, True):
            m = SupportHead(8)
            q = torch.randn(3, 8)
            y = m(q, torch.randn(3, 4, 8), torch.tensor([[0,1,2,0]]*3)) if memory else m(q)
            y.square().sum().backward()
            self.assertTrue(all(p.grad is not None for p in m.parameters()))

    def test_metrics(self):
        m = metrics(torch.tensor([0,1,2]), torch.tensor([0,1,2]))
        self.assertEqual(m['macro_f1'], 1.)

    def test_end_to_end_train_and_interventions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'validation'):
                data = dict(split=split, test_accessed=False,
                            labels=torch.tensor([1,2,3] * 4),
                            tokens={'base_change': torch.randn(12, 3, 6)},
                            token_mask=torch.ones(12,3,dtype=torch.bool),
                            events=['a']*3 + ['b']*3 + ['c']*3 + ['d']*3,
                            images=[f'{split}_{i}' for i in range(12)], metadata={})
                torch.save(data, root / f'{split}.pt')
            cmd = [sys.executable, '-m', 'bda_benchmark.experiments.srac_support_ablation_v1.run',
                   '--train', str(root/'train.pt'), '--val', str(root/'validation.pt'),
                   '--output', str(root/'out'), '--sources', 'base_change', '--steps', '2',
                   '--eval_every', '2', '--dim', '8', '--batch_size', '4', '--topk', '2',
                   '--repeats', '2', '--device', 'cpu']
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((root/'out'/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(result['E2']['parameters'], result['E3']['parameters'])
            self.assertIn('E5_labels_1', result)
            self.assertEqual(len(list((root/'out').glob('*.pth'))), 8)


if __name__ == '__main__':
    unittest.main()
