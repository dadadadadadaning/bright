"""Subset, immutable replay, and wrapper-isolation contracts; no server needed."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .provenance import select_images, validate_failed_memory


class EntryContracts(unittest.TestCase):
    def test_full_smoke_and_prefix_are_distinct(self):
        val = ['im'+str(i) for i in range(450)]
        self.assertEqual(select_images(val), (val, False))
        self.assertEqual(select_images(val, True), (val[:2], True))
        self.assertEqual(select_images(val, False, 408, 1), (val[408:409], True))
        self.assertEqual(select_images(val, False, 0, 409), (val[:409], True))
        for start, count, smoke in ((-1, 1, False), (450, 1, False), (0, -1, False), (1, 1, True)):
            with self.assertRaises(ValueError):
                select_images(val, smoke, start, count)

    def _case(self, root):
        failed = root/'failed'
        failed.mkdir()
        manifest = dict(format='SRAC_STAGE_MEMORY_DIAGNOSTIC_V1',
            arguments=dict(train_list='train', val_list='val', output=str(failed),
                           workers=4, device='cuda:0', smoke=False),
            all_input_paths={'front': 'front_path'}, input_fingerprints_before={'front': 'hash'})
        (failed/'manifest.json').write_text(json.dumps(manifest))
        (failed/'summary.json').write_text(json.dumps(dict(complete=False, failure={'image': 'im408'})))
        args = argparse.Namespace(failed_run=str(failed), output=str(root/'new'), prefix=False,
                                  pool_mode='native', pool_capture='failure', pool_check_reference=False)
        return args, failed

    def test_recheck_exact_source_and_failed_image_not_filename_number(self):
        module = 'bda_benchmark.experiments.damageformer_srac_transformer_v1.common'
        with tempfile.TemporaryDirectory() as temp, patch(module+'.digest', return_value='hash'), \
                patch(module+'.lists', return_value=(['train'], ['im'+str(i) for i in range(450)])):
            args, failed = self._case(Path(temp))
            original = {p.name: p.read_bytes() for p in failed.iterdir()}
            result = validate_failed_memory(args)
            self.assertEqual((result['start_index'], result['max_images']), (408, 1))
            self.assertFalse(result['inventory_shards'])
            self.assertEqual(result['workers'], 4)
            args.prefix = True
            result = validate_failed_memory(args)
            self.assertEqual((result['start_index'], result['max_images']), (0, 409))
            self.assertEqual(original, {p.name: p.read_bytes() for p in failed.iterdir()})
            self.assertFalse(Path(args.output).exists())

    def test_recheck_hash_mismatch_and_overwrite_rejected(self):
        module = 'bda_benchmark.experiments.damageformer_srac_transformer_v1.common'
        with tempfile.TemporaryDirectory() as temp, patch(module+'.digest', return_value='changed'):
            args, failed = self._case(Path(temp))
            with self.assertRaisesRegex(ValueError, 'input changed'):
                validate_failed_memory(args)
            args.output = str(failed/'nested')
            with self.assertRaisesRegex(ValueError, 'preserved failed'):
                validate_failed_memory(args)

    def test_feature_wrapper_restores_functions_on_failure(self):
        from . import feature_audit as module
        class Guard:
            def __init__(self, output, **kwargs): self.output = output
            def __enter__(self): self.output.mkdir(); return self
            def __exit__(self, *args): pass
            def set_image(self, name): pass
            def report(self): return {'mode': 'native', 'restored': True}
        old_output, old_parse = module.parent.new_output, module.parent.parse_batch
        def failed_run(args):
            out = module.parent.new_output(args.output)
            (out/'summary.json').write_text(json.dumps({'complete': True, 'gates_passed': True}))
            raise RuntimeError('synthetic fault')
        with tempfile.TemporaryDirectory() as temp, patch.object(module, 'PoolGuard', Guard), \
                patch.object(module.parent, 'run', side_effect=failed_run):
            args = argparse.Namespace(output=str(Path(temp)/'new'), joint_run=None, checkpoint='fixed',
                                      pool_mode='native', pool_capture='failure', pool_check_reference=False,
                                      smoke=False, max_images=0)
            with self.assertRaisesRegex(RuntimeError, 'synthetic fault'):
                module.run(args)
            self.assertIs(module.parent.new_output, old_output)
            self.assertIs(module.parent.parse_batch, old_parse)
            summary = json.loads((Path(args.output)/'summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertFalse(summary['full_validation_evidence'])


if __name__ == '__main__':
    unittest.main()
