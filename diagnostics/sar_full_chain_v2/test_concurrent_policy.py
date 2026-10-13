"""Independent CPU checks of explicit GPU sharing across Q and Soft drivers.

No NVIDIA commands, Torch imports, BRIGHT data, or training are required.
The fixture Soft driver exercises its actual delayed ``gpu_queue`` import.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import chain


GPU_SOURCE = '''
calls = []
def resolve_gpu(device_index):
    calls.append(('resolve', device_index))
    return 'GPU-fixture'
def observation(uuid):
    calls.append(('observe', uuid))
    return dict(uuid=uuid, compute_pids=[12345], utilization_percent=75,
                free_memory_mib=14000, idle=False)
def wait_for_idle(device_index, log=None, poll_seconds=30, consecutive=2):
    calls.append(('native_wait', device_index))
    result = dict(native_wait=True, device_index=device_index)
    if log:
        log('NATIVE_WAIT')
    return result
'''


SOFT_SOURCE = '''
import argparse
import hashlib
import json
from pathlib import Path
def source_files(repo):
    p = Path(repo) / 'original.py'
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()}
def parser():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('run',))
    p.add_argument('--repo-root')
    p.add_argument('--sar-root')
    p.add_argument('--output', '--root', dest='output')
    p.add_argument('--python')
    return p
def run(args):
    bindings = source_files(args.repo_root)
    from gpu_queue import wait_for_idle
    queued = wait_for_idle(0, log=lambda text: print(text, flush=True))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    result = dict(complete=True, full_validation_evidence=True,
                  test_accessed=False, gpu_queue=queued, arguments=vars(args))
    (root / 'manifest.json').write_text(json.dumps(dict(source_bindings=bindings)))
    (root / 'workflow_state.json').write_text(json.dumps(result))
    return result
'''


class ConcurrentPolicyTests(unittest.TestCase):
    def setUp(self):
        self.saved_gpu_module = sys.modules.get('gpu_queue')
        self.saved_sys_path = list(sys.path)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.q = self.root / 'sar_q_matched_v1'
        self.soft = self.root / 'sar_soft_cross_v1'
        for folder in (self.q, self.soft):
            folder.mkdir()
            (folder / 'gpu_queue.py').write_text(GPU_SOURCE)
        (self.soft / 'run.py').write_text(SOFT_SOURCE)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        (self.repo / 'original.py').write_text('# original source fixture\n')

    def tearDown(self):
        if self.saved_gpu_module is None:
            sys.modules.pop('gpu_queue', None)
        else:
            sys.modules['gpu_queue'] = self.saved_gpu_module
        sys.path[:] = self.saved_sys_path
        self.temp.cleanup()

    def fake_queue(self, *, idle=False):
        sample = dict(uuid='GPU-real-sample', compute_pids=[] if idle else [777],
                      utilization_percent=0 if idle else 80,
                      free_memory_mib=14000, idle=idle)
        queue = SimpleNamespace(resolve_gpu=Mock(return_value='GPU-real-sample'),
                                observation=Mock(return_value=sample))
        return queue, sample

    def test_concurrent_launch_reports_busy_gpu_without_faking_idle(self):
        queue, sample = self.fake_queue()
        messages = []
        result = chain.concurrent_queue(queue, 'user explicitly chose sharing')(
            0, log=messages.append)
        self.assertEqual(result['compute_pids'], [777])
        self.assertEqual(result['utilization_percent'], 80)
        self.assertEqual(result['free_memory_mib'], 14000)
        self.assertIs(result['idle'], False)
        self.assertIs(result['wait_skipped'], True)
        self.assertEqual(result['execution_policy'], 'concurrent')
        self.assertIs(result['exclusive_reservation'], False)
        self.assertIs(result['cuda_context_created'], False)
        queue.resolve_gpu.assert_called_once_with(0)
        queue.observation.assert_called_once_with('GPU-real-sample')
        self.assertTrue(messages)
        self.assertNotIn('wait_skipped', sample)

    def test_concurrent_launch_keeps_actual_idle_when_gpu_is_idle(self):
        queue, _sample = self.fake_queue(idle=True)
        result = chain.concurrent_queue(queue, 'explicit concurrent policy')(2, log=lambda _: None)
        self.assertIs(result['idle'], True)
        self.assertEqual(result['compute_pids'], [])
        self.assertIs(result['wait_skipped'], True)
        self.assertEqual(result['execution_policy'], 'concurrent')

    def test_observation_failure_propagates_instead_of_inventing_availability(self):
        queue, _sample = self.fake_queue()
        queue.observation.side_effect = RuntimeError('nvidia observation unavailable')
        with self.assertRaisesRegex(RuntimeError, 'observation unavailable'):
            chain.concurrent_queue(queue, 'explicit sharing')(0, log=lambda _: None)

    def test_device_mapping_failure_propagates(self):
        queue, _sample = self.fake_queue()
        queue.resolve_gpu.side_effect = ValueError('ambiguous GPU mapping')
        with self.assertRaisesRegex(ValueError, 'ambiguous GPU mapping'):
            chain.concurrent_queue(queue, 'explicit sharing')(0, log=lambda _: None)
        queue.observation.assert_not_called()

    def test_default_configuration_keeps_native_wait(self):
        with patch.object(chain, 'HERE', self.root):
            module = chain.configure_gpu_queue(self.q, False)
        self.assertEqual(Path(module.__file__).resolve(), (self.q / 'gpu_queue.py').resolve())
        self.assertIs(sys.modules['gpu_queue'], module)
        result = module.wait_for_idle(0)
        self.assertTrue(result['native_wait'])
        self.assertNotIn('wait_skipped', result)
        self.assertEqual(module.calls, [('native_wait', 0)])

    def test_switch_from_q_to_soft_loads_exact_queue_and_retains_sharing(self):
        with patch.object(chain, 'HERE', self.root):
            q = chain.configure_gpu_queue(self.q, True)
            soft = chain.configure_gpu_queue(self.soft, True)
        self.assertIsNot(q, soft)
        self.assertEqual(Path(soft.__file__).resolve(), (self.soft / 'gpu_queue.py').resolve())
        self.assertIs(sys.modules['gpu_queue'], soft)
        result = soft.wait_for_idle(0, log=lambda _: None)
        self.assertIs(result['idle'], False)
        self.assertIs(result['wait_skipped'], True)
        self.assertEqual(result['execution_policy'], 'concurrent')
        self.assertNotIn(('native_wait', 0), soft.calls)

    def config(self, run_now):
        static_file = self.soft / 'run.py'
        return dict(repo=str(self.repo), python=sys.executable,
                    q_root=str(self.root / 'q-result'),
                    soft_root=str(self.root / ('soft-sharing' if run_now else 'soft-waiting')),
                    run_now=run_now, execution_policy='concurrent' if run_now else 'wait_idle',
                    bindings={str(static_file): chain.file_sha(static_file)})

    def test_soft_driver_delayed_import_obeys_explicit_concurrent_policy(self):
        config = self.config(True)
        with patch.object(chain, 'HERE', self.root), contextlib.redirect_stdout(io.StringIO()):
            result = chain.run_soft(config)
        self.assertIs(result['complete'], True)
        self.assertIs(result['gpu_queue']['idle'], False)
        self.assertIs(result['gpu_queue']['wait_skipped'], True)
        self.assertEqual(result['gpu_queue']['execution_policy'], 'concurrent')
        self.assertEqual(result['arguments']['sar_root'], config['q_root'])
        self.assertEqual(result['arguments']['repo_root'], config['repo'])
        self.assertEqual(result['arguments']['python'], config['python'])
        self.assertEqual(result['arguments']['output'], config['soft_root'])
        self.assertEqual(result['arguments']['execution_policy'], 'concurrent')
        self.assertIs(result['arguments']['wait_gpu_idle'], False)
        manifest = json.loads((Path(config['soft_root']) / 'manifest.json').read_text())
        bindings = manifest['source_bindings']
        self.assertIn(str(self.repo / 'original.py'), bindings)
        self.assertIn(str(self.soft / 'run.py'), bindings)
        self.assertIn(str(self.q / 'gpu_queue.py'), bindings)

    def test_soft_driver_default_still_uses_native_wait(self):
        config = self.config(False)
        with patch.object(chain, 'HERE', self.root), contextlib.redirect_stdout(io.StringIO()):
            result = chain.run_soft(config)
        self.assertTrue(result['gpu_queue']['native_wait'])
        self.assertNotIn('wait_skipped', result['gpu_queue'])
        self.assertEqual(result['arguments']['execution_policy'], 'wait_idle')
        self.assertIs(result['arguments']['wait_gpu_idle'], True)

    def test_run_now_skips_memory_and_probe_waits_without_signalling_processes(self):
        proc = dict(pid=456, start_ticks='789', argv=['python', 'memory.py'])
        record = {}
        config = dict(run_now=True, probe_parent=str(self.root / 'unused-probe'))
        with patch.object(chain, 'discover_memory_processes', return_value=[proc]), \
                patch.object(chain, 'process_identity', side_effect=AssertionError('must not wait')), \
                patch.object(chain.time, 'sleep', side_effect=AssertionError('must not sleep')), \
                patch.object(chain.fcntl, 'flock', side_effect=AssertionError('must not lock probe')), \
                patch.object(chain.os, 'kill', side_effect=AssertionError('must not signal')), \
                contextlib.redirect_stdout(io.StringIO()):
            with chain.resource_policy(config, record, self.root):
                self.assertEqual(record['phase'], 'concurrent_start')
        self.assertEqual(record['declared_memory_processes'], [proc])
        self.assertEqual(record['execution_policy'], 'concurrent')
        self.assertFalse(Path(config['probe_parent']).exists())

    def test_default_resource_policy_waits_for_declared_memory_and_locks_probe(self):
        proc = dict(pid=456, start_ticks='789', argv=['python', 'memory.py'])
        record = {}
        config = dict(run_now=False, probe_parent=str(self.root / 'default-probe'))
        with patch.object(chain, 'discover_memory_processes', return_value=[proc]), \
                patch.object(chain, 'process_identity', side_effect=[proc, None]), \
                patch.object(chain.time, 'sleep') as sleep, \
                patch.object(chain.fcntl, 'flock') as flock, \
                patch.object(chain.os, 'kill', side_effect=AssertionError('must not signal')), \
                contextlib.redirect_stdout(io.StringIO()):
            with chain.resource_policy(config, record, self.root):
                self.assertEqual(record['phase'], 'waiting_existing_probe')
        sleep.assert_called_once_with(30)
        flock.assert_called_once()
        self.assertEqual(record['execution_policy'], 'wait_idle')
        self.assertTrue((Path(config['probe_parent']) / '.probe.lock').is_file())


if __name__ == '__main__':
    unittest.main()
