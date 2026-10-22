import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import probe_core as core
import probe_cli as cli
from build import build


class ProbeContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        torch.set_num_threads(1)

    def check(self,a,b,expected,**kwargs):
        return core.check_chunk(torch,a,b,expected,offset=399780,phase='after_compute',
                                folder=self.root,position={'pattern':'weight_bits'},**kwargs)

    def test_observed_exponent_bit_and_exact_halving(self):
        before=np.array([0x3dd5a746],dtype=np.uint32)
        after=np.array([0x3d55a746],dtype=np.uint32)
        self.assertEqual(float(after.view(np.float32)[0]),float(before.view(np.float32)[0])/2)
        row=core.differences(before.view(np.int32),after.view(np.int32),399783)['samples'][0]
        self.assertEqual((row['xor_hex'],row['changed_bits']),('0x800000',1))
        self.assertEqual(row['word_index'],399783)

    def test_chunked_patterns_match_whole_reference(self):
        for kind in core.PATTERNS:
            whole=core.pattern(kind,0,257,33)
            split=np.concatenate([core.pattern(kind,0,71,33),core.pattern(kind,71,186,33)])
            np.testing.assert_array_equal(whole,split)
            self.assertEqual(whole.dtype,np.dtype('int32'))
        self.assertEqual(len(set(core.pattern('walking_bit',0,32,0).view(np.uint32))),32)
        self.assertGreater(len(set(core.pattern('address',0,257,0))),250)

    def test_healthy_buffers_produce_no_mismatch_artifacts(self):
        expected=core.pattern('address',0,1024,7)
        self.check(torch.from_numpy(expected.copy()),torch.from_numpy(expected.copy()),expected)
        self.assertEqual(list(self.root.iterdir()),[])

    def test_identically_corrupted_gpu_buffers_still_fail_cpu_reference(self):
        expected=core.pattern('weight_bits',0,8,0)
        actual=expected.copy();actual[3]^=0x800000
        with self.assertRaises(core.BufferMismatch) as caught:
            self.check(torch.from_numpy(actual.copy()),torch.from_numpy(actual.copy()),expected)
        record=caught.exception.record
        self.assertTrue(record['gpu_a_equals_b'])
        self.assertEqual(record['a_vs_expected']['changed_words'],1)
        self.assertTrue(record['reread_a_same_as_first'])
        self.assertFalse(record['hardware_fault_proven'])
        saved=np.load(self.root/'first_mismatch_cpu.npz')
        np.testing.assert_array_equal(saved['expected'],expected)
        np.testing.assert_array_equal(saved['a'],actual)

    def test_one_buffer_corruption_is_distinguished(self):
        expected=core.pattern('weight_bits',0,8,0)
        actual=expected.copy();actual[3]^=0x800000
        with self.assertRaises(core.BufferMismatch) as caught:
            self.check(torch.from_numpy(actual),torch.from_numpy(expected.copy()),expected)
        record=caught.exception.record
        self.assertFalse(record['gpu_a_equals_b'])
        self.assertEqual(record['b_vs_expected']['changed_words'],0)
        self.assertEqual(record['a_vs_expected']['samples'][0]['word_index'],399783)

    def test_failed_reread_keeps_first_evidence(self):
        expected=core.pattern('weight_bits',0,8,0)
        actual=expected.copy();actual[3]^=0x800000
        class ReadOnce:
            count=0
            def detach(self):return self
            def to(self,*args,**kwargs):
                self.count+=1
                if self.count>1:raise RuntimeError('simulated device-side assert')
                return torch.from_numpy(actual.copy())
        with patch.object(torch,'equal',return_value=False):
            with self.assertRaises(core.BufferMismatch) as caught:
                self.check(ReadOnce(),torch.from_numpy(expected.copy()),expected)
        self.assertIn('simulated device-side assert',caught.exception.record['reread_error'])
        self.assertTrue((self.root/'first_mismatch_cpu.npz').is_file())
        self.assertEqual(json.loads((self.root/'mismatch.json').read_text())['a_vs_expected']['changed_words'],1)

    def test_initial_cuda_failure_does_not_try_more_gpu_operations(self):
        with patch.object(torch,'equal',side_effect=RuntimeError('simulated CUDA error')) as equal:
            with self.assertRaisesRegex(RuntimeError,'simulated CUDA error'):
                self.check(None,None,core.pattern('zero',0,8,0))
        self.assertEqual(equal.call_count,1)
        self.assertEqual(list(self.root.iterdir()),[])

    def test_cpu_round_trip_exercises_every_pattern(self):
        clock=[0.0]
        def emit(row):
            if row['position']['phase']=='after_compute':clock[0]+=1.0
        with patch.object(core.time,'monotonic',side_effect=lambda:clock[0]):
            result=core.exercise(torch,seconds=7,mib=1,folder=self.root,stop={'requested':False},
                                 emit=emit,device='cpu',compute_seconds=0)
        self.assertEqual(result['completed_rounds'],7)
        self.assertTrue(result['all_patterns_completed'])
        self.assertFalse(result['hardware_cleared'])
        self.assertEqual(result['checked_bytes'],7*2*1024*1024)

    def test_stop_interrupts_before_a_round(self):
        with self.assertRaises(InterruptedError):
            core.exercise(torch,seconds=7,mib=1,folder=self.root,stop={'requested':True},
                          emit=lambda _:None,device='cpu',compute_seconds=0)

    def test_supervisor_rejects_stale_or_incomplete_success(self):
        good={'phase':'no_mismatch_observed','complete':True}
        self.assertTrue(cli.completion(good,0,None)['complete'])
        for worker,code,reason in ((good,2,None),(good,0,'wall_time_limit'),({},0,None),
                                  ({'phase':'checking','complete':True},0,None)):
            self.assertFalse(cli.completion(worker,code,reason)['complete'])

    def test_busy_gpu_and_unknown_process_listing_fail_closed(self):
        with patch.object(cli,'smi',return_value='123456789\n'):
            with self.assertRaisesRegex(RuntimeError,'existing compute'):cli.require_idle()
        with patch.object(cli,'smi',return_value='not supported\n'):
            with self.assertRaisesRegex(RuntimeError,'interpret'):cli.require_idle()
        with patch.object(cli,'smi',return_value=''):cli.require_idle()

    def test_kill_only_owned_subprocess_session(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        self.addCleanup(lambda:child.kill() if child.poll() is None else None)
        cli.kill_child(child)
        self.assertEqual(child.returncode,-signal.SIGTERM)

    def test_cpu_only_worker_reports_failed_instead_of_success(self):
        signals={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            with patch.object(cli,'validated',return_value={'seconds':30,'test_buffer_mib':64}), \
                 patch.object(cli,'require_idle'),patch.object(torch.cuda,'is_available',return_value=False):
                rc=cli.work(self.root)
        finally:
            for s,handler in signals.items():signal.signal(s,handler)
        self.assertEqual(rc,2)
        status=cli.read(self.root/'worker_summary.json')
        self.assertEqual(status['phase'],'failed')
        self.assertFalse(status['complete'])

    def run_controller(self,script,seconds):
        (self.root/'probe.pyz').write_text(script)
        signals={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            with patch.object(cli,'validated',return_value={'seconds':seconds,'gpu_uuid':'GPU-test'}), \
                 patch.object(cli,'leases',return_value=contextlib.nullcontext()), \
                 patch.object(cli,'smi',return_value='test environment'):
                return cli.control(self.root)
        finally:
            for s,handler in signals.items():signal.signal(s,handler)

    def test_controller_accepts_completed_real_child(self):
        script='import json,os\nfrom pathlib import Path\nassert os.environ["CUDA_VISIBLE_DEVICES"]=="GPU-test"\n'
        script+='Path(__file__).with_name("worker_summary.json").write_text(json.dumps({"phase":"no_mismatch_observed","complete":True}))\n'
        self.assertEqual(self.run_controller(script,30),0)
        result=cli.read(self.root/'summary.json')
        self.assertTrue(result['job_finished'])
        self.assertTrue(result['complete'])
        self.assertFalse(result['hardware_cleared'])

    def test_controller_wall_limit_overrides_stale_success(self):
        cli.write(self.root/'worker_summary.json',{'phase':'no_mismatch_observed','complete':True})
        # The injected request shortens the controller's 180-second margin for this test.
        self.assertEqual(self.run_controller('import time\ntime.sleep(30)\n',-179),2)
        result=cli.read(self.root/'summary.json')
        self.assertEqual(result['phase'],'wall_time_limit')
        self.assertFalse(result['complete'])
        self.assertTrue(result['job_finished'])

    def test_archive_help_and_reproducible_build(self):
        archive=build();first=archive.read_bytes()
        self.assertEqual(build().read_bytes(),first)
        result=subprocess.run([sys.executable,'-B',str(archive),'--help'],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('status',result.stdout)
        self.assertIn('stop',result.stdout)


if __name__=='__main__':unittest.main()
