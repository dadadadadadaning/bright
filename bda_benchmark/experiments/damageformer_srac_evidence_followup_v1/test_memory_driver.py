"""CPU synthetic driver integration; not BRIGHT/CUDA or an accuracy experiment.

Only artifact loading, image I/O, the frozen backbone and learned decoder are
replaced. Selection/exclusion, support interventions, native/grouped pooling,
GT region diagnostics, projected cohorts, transitions and JSON writing run.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1 import association
from . import memory_audit as driver
from .native_pool import _ORIGINAL_POOL


def logits_for(classes):
    return F.one_hot(classes.long(), 4).permute(0, 3, 1, 2).float()*8


class MemoryDriverContracts(unittest.TestCase):
    def setUp(self):
        self.settings = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        )

    def tearDown(self):
        enabled, warn, cudnn, benchmark, matmul, conv, workspace = self.settings
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
        torch.backends.cudnn.deterministic = cudnn
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = conv
        if workspace is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace

    def fixture(self, root, *, max_images=1, fail_on=None):
        names = ['fixture-event_00000001', 'fixture-event_00000002', 'fixture-event_00000003']
        train = root/'train.txt'
        val = root/'val.txt'
        train.write_text('other-event_00000001\n', encoding='utf-8')
        val.write_text('\n'.join(names)+'\n', encoding='utf-8')
        target = torch.tensor([[[2, 2], [1, 3]]])
        base_prediction = torch.tensor([[[1, 2], [1, 3]]])
        pixel_feature = torch.eye(4).T.reshape(1, 4, 2, 2)
        q = torch.zeros(1, 9, 4)
        q[:, 0] = 1
        ids = torch.arange(4).expand(9, 4).contiguous()
        tokens = torch.cat((torch.eye(4), -torch.ones(1, 4)), dim=0)
        distribution = F.one_hot(torch.tensor([2, 2, 1, 3, 0]), 4).float()
        bank = dict(tokens=tokens, distribution=distribution, geometry=torch.zeros(5, 8),
                    events=torch.ones(5, dtype=torch.long), images=torch.arange(5),
                    disasters=torch.zeros(5, dtype=torch.long),
                    event_names=['fixture-event', 'other-event'])
        ledger = {'train': train, 'val': val}
        code = {'memory_code_sha256': 'synthetic-test-code'}
        bundle = dict(
            val_names=names, front=object(), bank=bank, decoders={arm: arm for arm in driver.ARMS},
            metric=torch.nn.Identity(), topk=1, hashes={'front': 'synthetic-front'},
            head_hashes={arm: 'synthetic-'+arm for arm in driver.ARMS}, paths={},
            source_manifest={'format': 'SRAC_SNAPSHOT_JOINT123_V1', 'snapshot_protocol': True},
            front_payload={'step': 3000, 'run_complete': False}, joint_step=5000,
            reference_hashes={}, code_before=code, reference={}, input_paths=ledger,
            inputs_before=driver.input_fingerprints(ledger))
        capture_instances = []

        class Capture:
            def __init__(self, front):
                self.calls = 0
                self.closed = False
                capture_instances.append(self)

            def __call__(self, pre, post):
                self.calls += 1
                call_ids = ids.clone()
                if self.calls == fail_on:
                    call_ids[0, 0] = 4  # Pre-kernel contract failure, not simulated CUDA.
                pooled = association.soft_pool_unpool(pixel_feature, q, call_ids, 4)
                result = dict(q=q.clone(), ids=call_ids, tokens=pooled['tokens'], shape=(2, 2),
                              B=torch.ones(1, 1, 2, 2),
                              old_pixel=logits_for(base_prediction),
                              prob=logits_for(base_prediction).softmax(1))
                return result, pixel_feature

            def close(self):
                self.closed = True

        def data_loader(dataset, selected_names, shuffle, workers, seed):
            return [(torch.zeros(1, 3, 2, 2), torch.zeros(1, 1, 2, 2),
                     torch.zeros(1, 2, 2), target.clone(), [name]) for name in selected_names]

        def decode(decoder, arm, feature, result, support_j2, support_j3):
            if arm == 'j1':
                predicted = base_prediction
            else:
                support = support_j2 if arm == 'j2' else support_j3
                _, labels, valid, _ = support
                # Synthetic deterministic classifier depends on supplied labels
                # and falls back to J1 when supports are all invalid. No query GT.
                selected_class = labels[:, 0].argmax(-1)
                prediction = torch.where(valid.any(-1), selected_class, base_prediction.reshape(-1))
                predicted = prediction.reshape(1, 2, 2)
            return {'logits': logits_for(predicted)}

        args = SimpleNamespace(
            joint_run=str(root/'synthetic_joint'), a1_checkpoint=str(root/'synthetic_a1'),
            dataset='synthetic_memory_only', train_list=str(train), val_list=str(val),
            output=str(root/'audit'), workers=0, device='cpu', seed=3407,
            temperature=0.1, intervention_arms=['j2', 'j3'], inventory_shards=False,
            smoke=False, start_index=0, max_images=max_images, pool_mode='grouped',
            pool_capture='failure', pool_check_reference=True)
        replacements = dict(
            load_bundle=lambda _args, _device: bundle,
            fingerprint=lambda _bundle: {'unchanged': 'synthetic-model'},
            inventory=lambda _bundle, _scan: {'bank': driver.distribution_inventory(bank['distribution']),
                                             'train_shards_scanned': False},
            full_image_loader=data_loader, FrozenFrontCapture=Capture,
            parse_batch=lambda batch, _device: batch, decoder_forward=decode,
            code_hash=lambda: 'synthetic-followup-code',
            diagnostic_code_fingerprint=lambda _format: code,
        )
        return args, replacements, capture_instances

    def test_subset_executes_real_diagnostics_and_restores_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, replacements, captures = self.fixture(root)
            with ExitStack() as stack:
                for name, value in replacements.items():
                    stack.enter_context(patch.object(driver, name, value))
                driver.run(args)
            report = json.loads((root/'audit'/'audit.json').read_text(encoding='utf-8'))
            summary = json.loads((root/'audit'/'summary.json').read_text(encoding='utf-8'))
            row = json.loads((root/'audit'/'images.jsonl').read_text(encoding='utf-8'))
            self.assertTrue(summary['complete'])
            self.assertTrue(summary['gates_passed'])
            self.assertFalse(summary['full_validation_evidence'])
            self.assertTrue(summary['diagnostic_subset'])
            self.assertIsNone(summary['reproduction']['passed'])
            self.assertTrue(summary['pool_guard']['restored'])
            self.assertEqual(summary['pool_guard']['calls'], 2)
            self.assertEqual(summary['pool_guard']['reference_checks'], 2)
            self.assertTrue(captures[0].closed)
            self.assertIs(association.soft_pool_unpool, _ORIGINAL_POOL)
            for arm in ('j2', 'j3'):
                hard = report['difficult_retrieval'][arm]['damaged_j1_wrong']
                self.assertEqual(hard['count'], 1)
                self.assertEqual(hard['damaged_hit_mass_sum'], 1.0)
                self.assertEqual(hard['damaged_neighbor_fraction_mass_sum'], 1.0)
                self.assertEqual(row['difficult_retrieval'][arm], report['difficult_retrieval'][arm])
                cohort = report['difficult_cohorts'][arm]['damaged_j1_wrong']
                self.assertEqual(cohort['evidence_damaged'], 1)
                self.assertEqual(cohort['evidence_damaged_and_final_damaged'], 1)
                self.assertEqual(report['retrieval'][arm]['2']['queries'], 2)
                for control in ('random', 'shuffle_labels', 'off'):
                    self.assertIn(arm+'_'+control, report['metrics'])
                    self.assertIn(arm+'_to_'+arm+'_'+control, report['transitions'])
            self.assertEqual(report['metrics']['j1']['recall'][2], 0.5)
            self.assertEqual(report['metrics']['j2']['recall'][2], 1.0)
            self.assertEqual(report['metrics']['j2_off']['recall'][2], 0.5)
            self.assertIn('fixture-event', report['per_event_difficult_retrieval'])

    def test_failure_retains_completed_journal_and_exact_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, replacements, captures = self.fixture(root, max_images=2, fail_on=2)
            with ExitStack() as stack:
                for name, value in replacements.items():
                    stack.enter_context(patch.object(driver, name, value))
                with self.assertRaisesRegex(ValueError, 'Out-of-range'):
                    driver.run(args)
            summary = json.loads((root/'audit'/'summary.json').read_text(encoding='utf-8'))
            self.assertFalse(summary['complete'])
            self.assertFalse(summary['gates_passed'])
            self.assertFalse(summary['full_validation_evidence'])
            self.assertEqual(summary['failure']['image'], 'fixture-event_00000002')
            self.assertEqual(summary['failure']['phase'], 'native_capture')
            self.assertEqual(summary['failure']['images_completed'], 1)
            self.assertTrue(summary['pool_guard']['restored'])
            self.assertEqual(summary['pool_guard']['failures'][0]['image'], 'fixture-event_00000002')
            saved = torch.load(summary['pool_guard']['saved_inputs'][0], weights_only=True)
            self.assertEqual(saved['image'], 'fixture-event_00000002')
            journal = (root/'audit'/'images.jsonl').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(journal), 1)
            self.assertEqual(json.loads(journal[0])['image'], 'fixture-event_00000001')
            self.assertFalse((root/'audit'/'audit.json').exists())
            self.assertTrue(captures[0].closed)
            self.assertIs(association.soft_pool_unpool, _ORIGINAL_POOL)


if __name__ == '__main__':
    unittest.main()
