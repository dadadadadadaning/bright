"""Tiny CPU integration of the real loop; no real dataset/GPU training."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

import torch
from torch import nn
from . import train
from . import audit as audit_cli
from .report import render
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import (
    candidate_map, masked_local_softmax,
)


class DevicePreparationContract(unittest.TestCase):
    def test_indexed_cuda_initializes_then_selects_then_resets(self):
        cuda = Mock()
        cuda.is_available.return_value = True
        with patch.object(train.torch, 'cuda', cuda):
            device = train.prepare_device('cuda:0')
        self.assertEqual(device, torch.device('cuda:0'))
        self.assertEqual(cuda.mock_calls, [call.is_available(), call.init(),
                                         call.set_device(device), call.reset_peak_memory_stats(device)])

    def test_cpu_never_calls_cuda(self):
        cuda = Mock()
        with patch.object(train.torch, 'cuda', cuda):
            device = train.prepare_device('cpu')
        self.assertEqual(device, torch.device('cpu'))
        self.assertEqual(cuda.mock_calls, [])

    def test_unavailable_cuda_raises_without_cpu_fallback(self):
        cuda = Mock()
        cuda.is_available.return_value = False
        with patch.object(train.torch, 'cuda', cuda):
            with self.assertRaises(RuntimeError):
                train.prepare_device('cuda:0')
        self.assertEqual(cuda.mock_calls, [call.is_available()])

    def test_unindexed_cuda_resolves_current_index_after_initialization(self):
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.current_device.return_value = 2
        with patch.object(train.torch, 'cuda', cuda):
            device = train.prepare_device('cuda')
        self.assertEqual(device, torch.device('cuda:2'))
        self.assertEqual(cuda.mock_calls, [call.is_available(), call.init(), call.current_device(),
                                         call.set_device(device), call.reset_peak_memory_stats(device)])


class TrainingContract(unittest.TestCase):
    def test_invalid_epsilon_rejected_before_loading_data(self):
        common = ['train','--checkpoint','unused','--dataset','unused','--train_list','unused',
                  '--val_list','unused','--output','unused']
        for epsilon in ('-0.1','0.6','nan','inf'):
            with self.subTest(epsilon=epsilon), patch.object(sys,'argv',common+['--epsilon',epsilon]), \
                    patch.object(train,'lists') as lists, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    train.main()
                self.assertEqual(error.exception.code,2)
                lists.assert_not_called()

    def test_front_guard_detects_buffer_and_trainable_parameter_changes(self):
        front = nn.Linear(2,2).eval().requires_grad_(False)
        front.register_buffer('scalar',torch.tensor(3.))
        buffers = {k:v.clone() for k,v in front.named_buffers()}
        original = train.state_digest(front)
        train.assert_front(front,buffers)
        front.scalar.add_(1)
        self.assertNotEqual(original,train.state_digest(front))
        with self.assertRaisesRegex(RuntimeError,'buffer changed'):
            train.assert_front(front,buffers)
        front.scalar.copy_(buffers['scalar'])
        front.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError,'trainable'):
            train.assert_front(front,buffers)

    def test_twenty_step_synthetic_entrypoint_writes_complete_paired_artifacts(self):
        torch.set_num_threads(1)
        torch.manual_seed(10)
        front = nn.Linear(2, 2).eval().requires_grad_(False)
        front.register_buffer('unchanged', torch.tensor(7.))
        front.config = dict(dim=8, stride=4)
        ids, valid, gh, gw = candidate_map(3, 5, 4, torch.device('cpu'))
        reference = dict(q=masked_local_softmax(torch.randn(2, 9, 3, 5), valid), ids=ids,
                         shape=(3, 5), tokens=torch.zeros(2, gh*gw, 8),
                         building=torch.rand(2, 1, 3, 5),
                         features={'roi': torch.randn(2, 256, 3, 5)},
                         pixel=torch.randn(2, 4, 6, 10),
                         prob=torch.randn(2, 4, 6, 10).softmax(1), identity_max_abs=0.)
        gt = torch.randint(0, 4, (2, 6, 10))
        gt[:, 0, 0] = 255
        batch = (torch.randn(2, 3, 6, 10), torch.randn(2, 3, 6, 10), None, gt,
                 ['haiti-earthquake_1', 'haiti-earthquake_2'])
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            ck = base/'front.pth'; ck.write_bytes(b'synthetic fixture only')
            tl = base/'train_set.txt'; tl.write_text('fake train', encoding='utf-8')
            vl = base/'val_set.txt'; vl.write_text('fake val', encoding='utf-8')
            output = base/'new_run'
            argv = ['train', '--checkpoint', str(ck), '--dataset', temp,
                    '--train_list', str(tl), '--val_list', str(vl), '--output', str(output),
                    '--device', 'cpu', '--workers', '0', '--steps', '20', '--eval_every', '20', '--smoke']
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(sys, 'argv', argv))
                stack.enter_context(patch.object(train, 'lists', return_value=(['a'], ['b', 'c'])))
                stack.enter_context(patch.object(train, 'load_fixed', return_value=(
                    front, dict(step=3000, valid_miou=0.), train.digest(ck))))
                stack.enter_context(patch.object(train, 'loader', return_value=[batch]))
                stack.enter_context(patch.object(train, 'parse_batch', side_effect=lambda b, device: b))
                stack.enter_context(patch.object(train, 'capture', return_value=reference))
                stack.enter_context(patch.object(train, 'verify_original_readout', return_value=0.))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                train.main()
            summary = json.loads((output/'summary.json').read_text())
            manifest = json.loads((output/'manifest.json').read_text())
            audit = json.loads((output/'audit_final.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertTrue(summary['immutable_front_verified'])
            self.assertEqual(summary['final_step'], 20)
            self.assertTrue(summary['smoke'])
            self.assertEqual(audit['image_count'], 2)
            self.assertEqual(len(set(manifest['initial_head_sha256'].values())), 1)
            self.assertEqual(set(summary['best']), {'hard', 'exchange'})
            for mode in ('hard', 'exchange'):
                self.assertTrue((output/(mode+'_best.pth')).is_file())
                self.assertTrue((output/(mode+'_last.pth')).is_file())
            self.assertEqual(len(list(output.glob('*.pth'))), 4)
            self.assertTrue(all(p.grad is None for p in front.parameters()))
            self.assertEqual(float(front.unchanged), 7.)
            initial = json.loads((output/'initial.json').read_text())
            self.assertIn('hard_s2_parity', initial)
            self.assertEqual(manifest['effective_epsilon'], {'hard': 0., 'exchange': .05})
            self.assertEqual(manifest['arguments']['variants'], ['hard', 'exchange'])
            self.assertEqual(set(audit['groups']['all']['comparisons']),
                             {'hard__exchange','old_pixel__hard','old_pixel__exchange'})
            for roi in ('true_positive','false_negative','false_positive','true_negative'):
                counts = [audit['groups'][f'b_bin_{i}__roi_{roi}']['metrics']['hard']['count'] for i in range(10)]
                self.assertEqual(sum(counts),audit['groups']['roi_'+roi]['metrics']['hard']['count'])
            for mode in ('hard', 'exchange'):
                saved = torch.load(output/(mode+'_last.pth'), map_location='cpu', weights_only=False)
                self.assertEqual(set(saved), {'head', 'step', 'name', 'metrics', 'manifest'})
                self.assertEqual(saved['step'], 20)
                self.assertEqual(set(saved['head']), set(torch.load(
                    output/(mode+'_best.pth'), map_location='cpu', weights_only=False)['head']))
            log = [json.loads(line) for line in (output/'validation.jsonl').read_text().splitlines()]
            self.assertEqual(len(log), 1)
            self.assertEqual(log[0]['step'], 20)
            self.assertEqual(log[0]['diagnostics_file'], 'audit_final.json')
            self.assertNotIn('diagnostics', log[0])
            # Reload the REAL saved states through the separate frozen-audit CLI.
            # Only unavailable dataset/front inputs are replaced; provenance,
            # strict load_state_dict, routing and all diagnostics remain real.
            audit_output = base/'frozen_audit'
            audit_argv = ['audit','--checkpoint',str(ck),'--run_root',str(output),
                          '--output',str(audit_output),'--dataset',temp,
                          '--train_list',str(tl),'--val_list',str(vl),
                          '--names','hard','exchange','--device','cpu','--workers','0','--smoke']
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(sys,'argv',audit_argv))
                stack.enter_context(patch.object(audit_cli,'lists',return_value=(['a'],['b','c'])))
                stack.enter_context(patch.object(audit_cli,'load_fixed',return_value=(
                    front,dict(step=3000,valid_miou=0.),train.digest(ck))))
                stack.enter_context(patch.object(audit_cli,'loader',return_value=[batch]))
                stack.enter_context(patch.object(audit_cli,'parse_batch',side_effect=lambda b,device:b))
                stack.enter_context(patch.object(audit_cli.factorial,'capture',return_value=reference))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                audit_cli.main()
            audit_summary = json.loads((audit_output/'summary.json').read_text())
            audited = json.loads((audit_output/'audit_final.json').read_text())
            provenance = audited['provenance']
            self.assertTrue(audit_summary['complete'])
            self.assertEqual(audit_summary['final_step'],20)
            self.assertEqual(audit_summary['image_count'],2)
            self.assertEqual(audited['groups'],audit['groups'])
            self.assertEqual(audited['per_event'],audit['per_event'])
            self.assertEqual(provenance['before_state_sha256'],provenance['after_state_sha256'])
            for key in ('immutable_front_verified','immutable_readouts_verified',
                        'source_integrity_verified','same_step_verified'):
                self.assertTrue(provenance[key])
            self.assertFalse(provenance['weights_updated'])
            self.assertFalse(provenance['gt_used_for_routing'])
            self.assertFalse(provenance['final_metrics_reproduced'])  # smoke is not full-Val proof
            self.assertEqual(set(provenance['head_file_sha256']),{'hard','exchange'})
            self.assertEqual(provenance['source_run_manifest_sha256'],train.digest(output/'manifest.json'))
            self.assertEqual(audit_summary['audit_sha256'],train.digest(audit_output/'audit_final.json'))
            self.assertFalse(list(audit_output.glob('*.pth')))
            report = render(audit_output)
            self.assertIn('complete: True; smoke: True',report)
            self.assertIn('PRIMARY PAIR: exchange minus hard',report)
            self.assertIn('B confidence x ROI correctness',report)
            self.assertIn('Smoke validates execution only',report)


if __name__ == '__main__':
    unittest.main()


