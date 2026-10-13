import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import SAR_FREEZE_WITNESS_V1 as tool


def original(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        t = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        self.decoder = nn.Linear(3, 3)
        self.adapter = nn.Linear(3, 3)
        self.embedding = nn.Linear(3, 3)
        self.region_classifier = nn.Linear(3, 2)
        for m in (self.base, self.adapter, self.embedding, self.region_classifier):
            m.requires_grad_(False).eval()


class WitnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = Model()
        self.witness = tool.FrozenWitness(self.root, original, self.model)

    def tearDown(self):
        self.witness.close()
        self.tmp.cleanup()

    def test_hash_matches_original(self):
        self.assertEqual(tool.state_fingerprint(tool.clone_state(self.model.base)), original(self.model.base))

    def test_snapshot_does_not_alias_cpu(self):
        before = tool.clone_state(self.model.base)
        with torch.no_grad():
            self.model.base[0].weight.add_(1)
        self.assertNotEqual(tool.state_fingerprint(before), original(self.model.base))

    def test_unchanged_passes(self):
        value = self.witness.fingerprint(self.model.base)
        self.witness.check_all('test')
        self.assertEqual(value, self.witness.fingerprint(self.model.base))
        self.assertFalse((self.root / 'failure.json').exists())

    def test_parameter_change_captured_exactly(self):
        self.witness.fingerprint(self.model.base)
        before = self.model.base[0].weight.detach().clone()
        with torch.no_grad():
            self.model.base[0].weight[0, 0] += 0.1
        with self.assertRaisesRegex(RuntimeError, 'frozen_state_changed'):
            self.witness.check_all('after_optimizer_step')
        report = json.loads((self.root / 'failure.json').read_text())
        self.assertEqual([r['name'] for r in report['differences']], ['0.weight'])
        saved = torch.load(self.root / 'frozen_change.pth', weights_only=False)
        self.assertTrue(torch.equal(saved['before']['0.weight'], before))
        self.assertFalse(report['training_certified'])

    def test_buffer_change_detected(self):
        self.witness.fingerprint(self.model.base)
        self.model.base[1].num_batches_tracked.add_(1)
        with self.assertRaises(RuntimeError):
            self.witness.check_all('test')
        report = json.loads((self.root / 'failure.json').read_text())
        self.assertEqual(report['differences'][0]['name'], '1.num_batches_tracked')

    def test_mode_change_rejected_before_forward(self):
        self.witness.fingerprint(self.model.base)
        self.model.base.train()
        with self.assertRaisesRegex(RuntimeError, 'frozen_mode_or_gradient_contract'):
            self.model.base(torch.ones(2, 3))
        self.assertEqual(int(self.model.base[1].num_batches_tracked), 0)

    def test_requires_grad_rejected(self):
        self.witness.fingerprint(self.model.base)
        self.model.base[0].weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, 'frozen_mode_or_gradient_contract'):
            self.model.base(torch.ones(2, 3))

    def test_signed_zero_and_nan_use_bytes(self):
        self.assertEqual(len(tool.differences({'x': torch.tensor([0.0])}, {'x': torch.tensor([-0.0])})), 1)
        nan = torch.tensor([float('nan')])
        self.assertEqual(tool.differences({'x': nan}, {'x': nan.clone()}), [])

    def test_order_is_distinguished_from_mutation(self):
        a = torch.ones(1)
        self.assertEqual(tool.differences({'a': a, 'b': a}, {'b': a, 'a': a})[0]['reason'], 'state_dict_order_only')

    def test_added_deleted_and_shape(self):
        rows = tool.differences({'gone': torch.zeros(1), 'x': torch.zeros(1)},
                                {'added': torch.zeros(1), 'x': torch.zeros(2)})
        self.assertEqual(len(rows), 3)

    def test_unstable_reads_fail_not_saved_hash_returned(self):
        self.witness.original_fingerprint = lambda m: 'different'
        with self.assertRaisesRegex(RuntimeError, 'unstable_consecutive_state_reads'):
            self.witness.fingerprint(self.model.base)
        self.assertEqual(json.loads((self.root / 'failure.json').read_text())['second_sha256'], 'different')

    def test_instrumentation_preserves_gradients_and_update(self):
        import copy
        other = copy.deepcopy(self.model)
        for m in (self.model.base, self.model.adapter, self.model.embedding, self.model.region_classifier):
            self.witness.fingerprint(m)
        x = torch.randn(2, 3)
        for m in (self.model, other):
            opt = torch.optim.AdamW(m.decoder.parameters(), lr=1e-4)
            loss = m.region_classifier(m.embedding(m.adapter(m.decoder(m.base(x))))).square().mean()
            loss.backward()
            opt.step()
        self.witness.check_all('after_optimizer_step')
        for a, b in zip(self.model.parameters(), other.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_cli_passes_training_output_after_separator(self):
        with patch.object(tool, 'child') as mocked:
            tool.main(['child', '--evidence', str(self.root), '--', '--output', 'fit', '--arm', 'baseline'])
        self.assertEqual(mocked.call_args.args[1], ['--output', 'fit', '--arm', 'baseline'])

    def test_inspection_default_does_not_start_probe(self):
        with patch.object(tool, 'inspect') as inspected, patch.object(tool, 'probe') as probe:
            tool.main([])
        inspected.assert_called_once()
        probe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
