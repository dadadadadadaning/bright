"""CPU contracts; real captured inputs and server CUDA must still be checked."""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1 import association
from . import native_pool as pool
from .replay_pool import run as replay, restore_layout


def fixture():
    torch.manual_seed(7)
    # Transpose is deliberately noncontiguous; duplicate IDs/empty region 5.
    feature = torch.randn(2, 4, 3, 2).transpose(2, 3)
    ids = torch.randint(0, 5, (9, 6))
    weights = torch.rand(2, 9, 6)*0.17
    return feature, weights, ids, 6


class NativePoolContracts(unittest.TestCase):
    def test_grouped_matches_original_with_unnormalized_weights(self):
        args = fixture()
        original = pool._ORIGINAL_POOL(*args)
        actual = pool.grouped_pool(*args, max_work_elements=32)
        self.assertTrue(pool.compare_results(actual, original)['passed'])
        self.assertTrue(torch.equal(actual['tokens'][:, 5], torch.zeros_like(actual['tokens'][:, 5])))

    def test_tiny_mass_preserves_original_clamp(self):
        feature = torch.full((1, 1, 1, 1), 7.0)
        weights = torch.full((1, 9, 1), 1e-9)
        ids = torch.zeros(9, 1, dtype=torch.long)
        actual = pool.grouped_pool(feature, weights, ids, 2)
        self.assertAlmostEqual(actual['tokens'][0, 0, 0].item(), 0.063, places=6)
        self.assertTrue(pool.compare_results(actual, pool._ORIGINAL_POOL(feature, weights, ids, 2))['passed'])

    def test_zero_weights(self):
        ff, ww, ii, kk = fixture()
        result = pool.grouped_pool(ff, torch.zeros_like(ww), ii, kk)
        for tensor in result.values():
            self.assertTrue(torch.equal(tensor, torch.zeros_like(tensor)))

    def test_independent_dense_double_reference(self):
        ff, ww, ii, kk = fixture()
        ff, ww = ff.double(), ww.double()
        # Independent explicit K-by-N construction is tiny and test-only.
        route = torch.zeros(ff.shape[0], kk, ff.shape[-2]*ff.shape[-1], dtype=torch.float64)
        for batch in range(ff.shape[0]):
            for candidate in range(9):
                for pixel in range(ii.shape[1]):
                    route[batch, ii[candidate, pixel], pixel] += ww[batch, candidate, pixel]
        mass = route.sum(-1)
        numerator = torch.bmm(route, ff.flatten(2).transpose(1, 2))
        tokens = numerator / mass.clamp_min(1e-6)[..., None]
        reconstruction = torch.bmm(route.transpose(1, 2), tokens).transpose(1, 2).reshape_as(ff)
        actual = pool.grouped_pool(ff, ww, ii, kk, max_work_elements=64)
        for key, expected in (('tokens', tokens), ('mass', mass), ('reconstruction', reconstruction)):
            self.assertEqual(actual[key].dtype, torch.float64)
            torch.testing.assert_close(actual[key], expected, atol=1e-12, rtol=1e-12)

    def test_dtype_shape_and_negative_id_preconditions(self):
        ff, ww, ii, kk = fixture()
        with self.assertRaisesRegex(TypeError, 'torch.long'):
            pool.validate_metadata(ff, ww, ii.int(), kk)
        with self.assertRaisesRegex(TypeError, 'floating dtype'):
            pool.validate_metadata(ff, ww.double(), ii, kk)
        with self.assertRaisesRegex(ValueError, 'weights'):
            pool.validate_metadata(ff, ww[:, :8], ii, kk)
        snapshot = pool.snapshot_inputs(ff, ww, ii, kk)
        snapshot['ids'][0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'Out-of-range'):
            pool.validate_cpu_snapshot(snapshot)

    def test_cpu_snapshot_is_independent(self):
        ff, ww, ii, kk = fixture()
        saved = pool.snapshot_inputs(ff, ww, ii, kk)
        ff.add_(1)
        ww.zero_()
        ii.zero_()
        self.assertFalse(torch.equal(saved['content'], ff))
        self.assertFalse(torch.equal(saved['weights'], ww))
        self.assertFalse(torch.equal(saved['ids'], ii))

    def test_invalid_indices_rejected_before_kernel_and_saved(self):
        ff, ww, ii, kk = fixture()
        ii[0, 0] = kk
        with tempfile.TemporaryDirectory() as root:
            with pool.PoolGuard(Path(root)/'guard') as guard:
                guard.set_image('bad-image')
                with self.assertRaisesRegex(ValueError, 'Out-of-range'):
                    association.soft_pool_unpool(ff, ww, ii, kk)
            self.assertEqual(guard.report()['failures'][0]['phase'], 'input_validation')
            self.assertTrue(guard.report()['restored'])
            saved = torch.load(guard.saved_inputs[0], weights_only=True)
            self.assertEqual(saved['image'], 'bad-image')
            self.assertEqual(saved['ids'][0, 0], kk)

    def test_nonfinite_and_negative_inputs(self):
        for which in ('content', 'weights'):
            snapshot = pool.snapshot_inputs(*fixture())
            snapshot[which].reshape(-1)[0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                pool.validate_cpu_snapshot(snapshot)
        snapshot = pool.snapshot_inputs(*fixture())
        snapshot['weights'][0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'Negative'):
            pool.validate_cpu_snapshot(snapshot)

    def test_existing_and_late_import_aliases_restored(self):
        old_name = 'bda_benchmark._pool_guard_existing_contract'
        late_name = 'bda_benchmark._pool_guard_late_contract'
        old = types.ModuleType(old_name)
        old.alias = pool._ORIGINAL_POOL
        sys.modules[old_name] = old
        try:
            with tempfile.TemporaryDirectory() as root:
                with pool.PoolGuard(Path(root)/'guard', mode='grouped', check_reference=True) as guard:
                    self.assertIs(old.alias, guard.wrapper)
                    late = types.ModuleType(late_name)
                    late.alias = association.soft_pool_unpool
                    sys.modules[late_name] = late
                    actual = old.alias(*fixture())
                    self.assertEqual(guard.reference_checks, 1)
                    self.assertIn('tokens', actual)
                self.assertIs(old.alias, pool._ORIGINAL_POOL)
                self.assertIs(late.alias, pool._ORIGINAL_POOL)
                self.assertIs(association.soft_pool_unpool, pool._ORIGINAL_POOL)
        finally:
            sys.modules.pop(old_name, None)
            sys.modules.pop(late_name, None)

    def test_exception_restoration_and_capture_all(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, 'outer_failure'):
                with pool.PoolGuard(Path(root)/'guard', capture='all') as guard:
                    association.soft_pool_unpool(*fixture())
                    raise RuntimeError('outer_failure')
            self.assertIs(association.soft_pool_unpool, pool._ORIGINAL_POOL)
            self.assertEqual(len(guard.saved_inputs), 1)
            self.assertTrue((Path(root)/'guard'/'guard.json').is_file())

    def test_audit_guard_rejects_training(self):
        ff, ww, ii, kk = fixture()
        ff.requires_grad_()
        with tempfile.TemporaryDirectory() as root:
            with pool.PoolGuard(Path(root)/'guard'):
                with self.assertRaisesRegex(RuntimeError, 'audit-only'):
                    association.soft_pool_unpool(ff, ww, ii, kk)

    def test_native_kernel_failure_serializes_cpu_input(self):
        with tempfile.TemporaryDirectory() as root:
            with pool.PoolGuard(Path(root)/'guard') as guard:
                with patch.object(pool, '_ORIGINAL_POOL', side_effect=RuntimeError('CUDA error: simulated')):
                    with self.assertRaisesRegex(RuntimeError, 'simulated'):
                        association.soft_pool_unpool(*fixture())
            report = guard.report()
            self.assertEqual(len(report['saved_inputs']), 1)
            self.assertTrue(report['failures'][0]['restart_process_required'])
            snapshot = torch.load(report['saved_inputs'][0], weights_only=True)
            self.assertEqual(snapshot['content'].device.type, 'cpu')

    def test_replay_cpu_new_directory_only(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            torch.save(pool.snapshot_inputs(*fixture()), root/'input.pth')
            args = SimpleNamespace(input=str(root/'input.pth'), output=str(root/'out'),
                                   backend='cpu', device='cpu', repeats=2, layout='captured')
            self.assertTrue(replay(args)['gates_passed'])
            summary = json.loads((root/'out'/'summary.json').read_text())
            self.assertFalse(summary['full_validation_evidence'])
            with self.assertRaises(FileExistsError):
                replay(args)

    def test_replay_preserves_stride_offset_and_broadcast(self):
        source = torch.arange(80, dtype=torch.float32).reshape(8, 10)[1:5, 2:9:2]
        snapshot = source.contiguous().clone()
        restored = restore_layout(snapshot, pool.tensor_layout(source), torch.device('cpu'))
        self.assertEqual(restored.stride(), source.stride())
        self.assertEqual(restored.storage_offset(), source.storage_offset())
        self.assertTrue(torch.equal(source, restored))
        source = torch.arange(6.).reshape(1, 6).expand(4, 6)
        restored = restore_layout(source.clone(), pool.tensor_layout(source), torch.device('cpu'))
        self.assertEqual(restored.stride(), source.stride())
        self.assertTrue(torch.equal(source, restored))

    def test_guard_never_overwrites(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileExistsError):
                with pool.PoolGuard(root):
                    pass

    @unittest.skipUnless(torch.cuda.is_available(), 'Real server CUDA is required')
    def test_cuda_equivalence(self):
        ff, ww, ii, kk = fixture()
        ref = pool.cpu_reference(pool.snapshot_inputs(ff, ww, ii, kk))
        actual = pool.grouped_pool(ff.cuda(), ww.cuda(), ii.cuda(), kk)
        self.assertTrue(pool.compare_results(actual, ref)['passed'])


if __name__ == '__main__':
    unittest.main()
