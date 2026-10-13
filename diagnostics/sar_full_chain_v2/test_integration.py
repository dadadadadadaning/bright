import argparse
from contextlib import redirect_stdout
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import chain
import test_chain


class IntegrationTests(unittest.TestCase):
    def fixture(self):
        pipeline = chain.load_q_pipeline()
        path = chain.HERE / 'sar_q_matched_v1/test_pipeline.py'
        spec = importlib.util.spec_from_file_location('_original_sar_fixture', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'pipeline': pipeline}):
            spec.loader.exec_module(module)
        fixture = module.PipelineContracts(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return pipeline, fixture

    def run_instrumented_original_pipeline(self, corrupt=False, run_now=False):
        pipeline, fixture = self.fixture()
        helper = SimpleNamespace(q_root=fixture.output,
            witness=chain.HERE / 'SAR_FREEZE_WITNESS_V1.py',
            train_source=fixture.package / 'train_q.py', protocol=fixture.protocol_path)
        def child(stage, repo, env, log):
            result = fixture.child(stage, repo, env, log)
            if stage['kind'] == 'train':
                folder, summary = test_chain.FullChainTests.make_witness(helper, stage)
                if corrupt and stage['name'] == 'formal_train_baseline':
                    summary['updates'] = 1999
                    (folder/'summary.json').write_text(json.dumps(summary))
            return result
        config = dict(q_root=str(fixture.output), repo=str(fixture.repo),
                      protocol=str(fixture.protocol_path), python=sys.executable, run_now=run_now)
        configure_original = chain.configure_gpu_queue
        def configure(directory, run_now):
            if run_now:
                module = configure_original(directory, True)
                module.resolve_gpu = lambda index: 'GPU-fixture'
                module.observation = lambda uuid: dict(uuid=uuid, compute_pids=[40001],
                    utilization_percent=70, free_memory_mib=12000, idle=False)
            else:
                module = fixture.queue_module()
            sys.modules['gpu_queue'] = module
            return module
        with patch.object(chain, 'load_q_pipeline', return_value=pipeline), \
             patch.dict(sys.modules, {'gpu_queue': fixture.queue_module()}), \
             patch.object(chain, 'configure_gpu_queue', side_effect=configure), \
             patch.object(pipeline, 'run_child', side_effect=child), redirect_stdout(io.StringIO()):
            if corrupt:
                with self.assertRaisesRegex(ValueError, 'exact training budget'):
                    chain.run_q(config)
            else:
                chain.run_q(config)
        return fixture, chain.read(fixture.output/'workflow_state.json'), chain.read(fixture.output/'manifest.json')

    def test_all_21_original_stages_complete_with_bound_instrumentation(self):
        fixture, state, manifest = self.run_instrumented_original_pipeline()
        self.assertTrue(state['complete'])
        self.assertTrue(state['immutable'])
        self.assertEqual(len(state['stages']), 21)
        self.assertEqual(manifest['source_bindings'][str(chain.HERE/'chain.py')], chain.file_sha(chain.HERE/'chain.py'))
        trains = [row for row in state['stages'] if '_train_' in row['name']]
        self.assertEqual(len(trains), 8)
        self.assertTrue(all(row['gate']['frozen_exact'] for row in trains))
        self.assertTrue(all('checkpoint_sha256' in row['gate'] for row in trains))
        self.assertEqual(manifest['arguments']['old_failed_checkpoint_reused'], False)

    def test_missing_witness_budget_stops_original_pipeline_before_second_arm(self):
        fixture, state, manifest = self.run_instrumented_original_pipeline(corrupt=True)
        self.assertFalse(state['complete'])
        self.assertEqual(fixture.called[-1], 'formal_train_baseline')
        self.assertNotIn('formal_train_rgb_capacity', fixture.called)

    def test_concurrent_original_q_pipeline_keeps_all_gates_and_records_busy_gpu(self):
        fixture, state, manifest = self.run_instrumented_original_pipeline(run_now=True)
        self.assertTrue(state['complete'])
        self.assertEqual(len(state['stages']), 21)
        self.assertFalse(state['gpu_queue']['idle'])
        self.assertEqual(state['gpu_queue']['compute_pids'], [40001])
        self.assertTrue(state['gpu_queue']['wait_skipped'])
        self.assertEqual(manifest['arguments']['execution_policy'], 'concurrent')
        self.assertFalse(manifest['arguments']['wait_gpu_idle'])
        self.assertTrue(all(row['gate']['passed'] for row in state['stages']))

    def test_background_lock_survives_parent_return_and_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            args = SimpleNamespace(output_parent=str(parent), target='all', sar_root=None)
            config = dict(format=chain.FORMAT, repo=tmp, python=sys.executable,
                          protocol=str(parent/'protocol.json'), target='all', bindings={}, probe_parent=tmp)
            real_popen = subprocess.Popen
            children = []
            def launch(command, **kwargs):
                fd = int(command[-1])
                script = 'import os,time; os.fstat(%d); time.sleep(15)' % fd
                process = real_popen([sys.executable, '-c', script], **kwargs)
                children.append(process)
                return process
            try:
                with patch.object(chain, 'checked_config', return_value=config), \
                     patch.object(chain.subprocess, 'Popen', side_effect=launch), redirect_stdout(io.StringIO()):
                    chain.start(args)
                    self.assertIsNone(children[0].poll())
                    with self.assertRaisesRegex(RuntimeError, 'already queued'):
                        chain.start(args)
                launch_record = chain.read(parent/'latest.json')
                self.assertEqual(launch_record['pid'], children[0].pid)
                self.assertEqual(len(children), 1)
            finally:
                for child in children:
                    child.terminate()
                    child.wait(timeout=5)
            with (parent/'.serial.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_memory_discovery_uses_script_path_and_skips_zombies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = [('11', 'S', ['/python', '/New/x/joint_live_recovery_v2/recovery.py']),
                       ('12', 'Z', ['/python', '/New/x/joint_live_recovery_v2/server_worker.py']),
                       ('13', 'S', ['/python', '/other/train.py']),
                       ('14', 'S', ['/python', '-c', 'print("/joint_live_recovery_v2/foo.py")'])]
            for pid, state, argv in entries:
                folder = root/pid
                folder.mkdir()
                fields = [state] + ['0'] * 18 + ['90001'] + ['0'] * 10
                (folder/'stat').write_text(pid+' (worker with spaces) '+' '.join(fields))
                (folder/'cmdline').write_bytes(b'\0'.join(s.encode() for s in argv)+b'\0')
            result = chain.discover_memory_processes(root)
            self.assertEqual([row['pid'] for row in result], [11])
            self.assertEqual(result[0]['start_ticks'], '90001')
            self.assertIsNone(chain.process_identity(404, root))


if __name__ == '__main__':
    unittest.main()
