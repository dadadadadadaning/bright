"""CPU orchestration tests: no CUDA worker, training, or BRIGHT data is used."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import recovery


class RecoveryOrchestration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        source = Path(os.environ['JOINT_ORIGINAL_TOOL']) if os.environ.get('JOINT_ORIGINAL_TOOL') else (
            Path(__file__).resolve().parent.parent / 'joint_recovery_source/region_memory_joint_train_v1')
        if not source.is_dir():
            source = Path(recovery.TOOL)
        self.tool = self.base / 'original_tool'
        shutil.copytree(source, self.tool)
        self.repo = self.base / 'repo'
        self.association = self.repo / 'bda_benchmark/experiments/damageformer_a1_cross_modal_superpixel_v1/association.py'
        self.association.parent.mkdir(parents=True)
        actual = Path('/workspace/bright')
        if not actual.is_dir():
            actual = Path('/root/shared-nvme/BRIGHT-master')
        shutil.copyfile(actual / self.association.relative_to(self.repo), self.association)
        self.guard_source = self.repo / 'bda_benchmark/experiments/damageformer_srac_evidence_followup_v1/native_pool.py'
        self.guard_source.parent.mkdir(parents=True)
        shutil.copyfile(actual / self.guard_source.relative_to(self.repo), self.guard_source)
        self.root = self.base / 'failed'
        self.pool = self.root / 'train/pool_debug'
        self.pool.mkdir(parents=True)
        checkpoint_dir = self.root / 'train/checkpoints'
        checkpoint_dir.mkdir()
        self.checkpoint = checkpoint_dir / 'checkpoint_001500.pt'
        self.checkpoint.write_bytes(b'trusted fixture only; never unpickled')
        self.snapshot = self.pool / 'input_009005.pth'
        self.snapshot.write_bytes(b'CPU fixture only; worker is mocked')
        self.config = dict(mode='train', steps=5000, repo=str(self.repo), device='cuda:0')
        self.protocol = dict(source_runtime_settings={'cublas_workspace_config': None})
        self.manifest = dict(source_hash=recovery.source_hash(self.tool), arguments=self.config,
                             protocol_info=self.protocol, initial_head_hashes={a:'init' for a in ('r0','rt','rf')})
        self.dump(self.root / 'train/manifest.json', self.manifest)
        self.dump(self.root / 'train/summary.json', dict(complete=False, error='CUDA index assert',
                  updates_per_arm={a:1500 for a in ('r0','rt','rf')}, last_checkpoint=str(self.checkpoint)))
        self.dump(self.checkpoint.with_suffix('.pt.json'), dict(step=1500, sha256=recovery.digest(self.checkpoint),
                  source_hash=self.manifest['source_hash'], protocol_info=self.protocol))
        self.dump(self.pool / 'failure.json', dict(phase='pool_kernel', image='example', call=9005,
                                                  input=str(self.snapshot)))
        self.dump(self.pool / 'guard.json', dict(backend='native'))
        self.dump(self.pool / 'manifest.json', dict(original_source=str(self.association),
                  original_sha256=recovery.ASSOCIATION_SHA, audit_source_sha256=recovery.digest(self.guard_source)))
        (self.root / 'train/train.jsonl').write_text(''.join(json.dumps({'step':s})+'\n' for s in range(1,1501)))
        self.q, self.soft = self.base / 'sar_q', self.base / 'sar_soft'
        self.q.mkdir(); self.soft.mkdir()
        self.dump(self.q / 'workflow_state.json', dict(format='BRIGHT_SAR_Q_MATCHED_V1',
                  complete=True, full_validation_evidence=True, failure=None,finished_at=1))
        self.dump(self.soft / 'workflow_state.json', dict(format='SAR_SOFT_CROSS_PIPELINE_V1',
                  source_root=str(self.q), complete=True, full_validation_evidence=True, failure=None))
        soft_state=recovery.read(self.soft/'workflow_state.json'); soft_state['finished_at']=1
        self.dump(self.soft/'workflow_state.json',soft_state)

    def tearDown(self):
        self.temp.cleanup()

    def dump(self, path, value):
        path.write_text(json.dumps(value))

    def args(self, stage='cpu'):
        return SimpleNamespace(stage=stage, failed_run=str(self.root), tool=str(self.tool),
                   output=str(self.base/'new_output'), after_sar=[str(self.q), str(self.soft)], wait_timeout=1)

    def verify(self, folder):
        return 'fixture-release' if Path(folder)==recovery.HERE else self.original_verify(folder)

    def worker(self, command, log, env):
        backend = command[command.index('--backend')+1] if '--backend' in command else 'train'
        self.events.append(backend)
        output = Path(command[command.index('--output')+1]); output.mkdir()
        Path(log).write_text('isolated child mock')
        record = dict(gates_passed=True, image='example', call=9005, source_layout_recorded=True)
        if backend == 'train':
            record.update(complete=True, full_validation_evidence=True, updates_per_arm={a:5000 for a in ('r0','rt','rf')})
        if backend == 'contiguous' and self.bad_contiguous:
            record['gates_passed']=False
        self.dump(output/'summary.json', record)
        return 1 if backend=='native' else 0

    def run_pipeline(self, stage='cpu', bad_contiguous=False):
        self.events=[]; self.bad_contiguous=bad_contiguous
        self.original_verify=recovery.verify_release
        original_digest=recovery.digest
        queue=SimpleNamespace(wait_for_idle=lambda *a,**k:self.events.append('idle') or {'idle':True})
        with mock.patch.object(recovery,'verify_release',side_effect=self.verify), \
             mock.patch.object(recovery,'digest',side_effect=lambda p:'fixture-helper-release' if Path(p)==recovery.HERE/'SHA256SUMS.json' else original_digest(p)), \
             mock.patch.object(recovery,'run_child',side_effect=self.worker), \
             mock.patch.dict(sys.modules, {'gpu_queue':queue}), redirect_stdout(io.StringIO()):
            recovery.pipeline(self.args(stage))

    def test_preserved_inspection_and_mutation_rejection(self):
        evidence=recovery.inspect_failed(self.root,self.tool)
        self.assertEqual(evidence['checkpoint_step'],1500)
        recovery.check_preserved(evidence)
        self.checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            recovery.check_preserved(evidence)
        with self.assertRaisesRegex(ValueError,'checksum'):
            recovery.inspect_failed(self.root,self.tool)

    def test_extra_original_python_file_rejected(self):
        (self.tool/'unbound.py').write_text('x=1')
        with self.assertRaisesRegex(ValueError,'Unbound'):
            recovery.inspect_failed(self.root,self.tool)

    def test_mismatched_updates_rejected(self):
        path=self.root/'train/summary.json'; data=recovery.read(path)
        data['updates_per_arm']['rf']=1499; self.dump(path,data)
        with self.assertRaisesRegex(ValueError,'same saved update'):
            recovery.inspect_failed(self.root,self.tool)

    def test_cpu_stage_never_waits_or_starts_gpu(self):
        with mock.patch.object(recovery,'wait_sources',side_effect=AssertionError('GPU wait prohibited')):
            self.run_pipeline('cpu')
        self.assertEqual(self.events,['cpu'])
        state=recovery.read(self.base/'new_output/workflow_state.json')
        self.assertTrue(state['complete']); self.assertFalse(state['training_started'])
        self.assertFalse(state['full_validation_evidence'])

    def test_native_failure_does_not_block_verified_candidate(self):
        self.run_pipeline('recover')
        self.assertEqual(self.events,['cpu','idle','native','contiguous','train'])
        state=recovery.read(self.base/'new_output/workflow_state.json')
        self.assertTrue(state['complete']); self.assertFalse(state['exact_historical_resume_claim'])

    def test_candidate_failure_stops_before_training(self):
        with self.assertRaisesRegex(RuntimeError,'contiguous'):
            self.run_pipeline('recover',True)
        self.assertNotIn('train',self.events)
        self.assertIsNotNone(recovery.read(self.base/'new_output/workflow_state.json')['failure'])

    def test_q_complete_soft_pending_blocks(self):
        data=recovery.read(self.soft/'workflow_state.json'); data.update(complete=False,current_stage='waiting_sar_source')
        self.dump(self.soft/'workflow_state.json',data)
        clock=iter([0,0,2])
        with self.assertRaises(TimeoutError):
            recovery.wait_sources([self.q,self.soft],1,pause=lambda n:None,
                                  now=lambda:next(clock),emit=lambda *a,**k:None)

    def test_silent_interrupted_source_is_not_terminal(self):
        data=recovery.read(self.q/'workflow_state.json'); data['complete']=False
        self.dump(self.q/'workflow_state.json',data)
        self.assertFalse(recovery.terminal_source(self.q)[0])

    def test_terminal_state_requires_valid_finished_timestamp(self):
        data=recovery.read(self.q/'workflow_state.json'); data.pop('finished_at')
        self.dump(self.q/'workflow_state.json',data)
        with self.assertRaisesRegex(ValueError,'timestamp'):
            recovery.terminal_source(self.q)
        data.update(complete=False,failure='failed',finished_at=float('nan'))
        self.dump(self.q/'workflow_state.json',data)
        with self.assertRaisesRegex(ValueError,'timestamp'):
            recovery.terminal_source(self.q)

    def test_cpu_runtime_metadata_does_not_probe_cuda(self):
        import torch
        import replay_worker
        with mock.patch.object(torch.cuda,'current_device',side_effect=AssertionError('CUDA probe')):
            value=replay_worker.current_runtime(cpu_only=True)
        self.assertTrue(value['cuda_not_probed'])

    def test_soft_source_mismatch_rejected(self):
        data=recovery.read(self.soft/'workflow_state.json'); data['source_root']=str(self.base/'other')
        self.dump(self.soft/'workflow_state.json',data)
        with self.assertRaisesRegex(ValueError,'different SAR-Q'):
            recovery.wait_sources([self.q,self.soft])

    def test_both_recorded_failures_are_terminal(self):
        for root in (self.q,self.soft):
            data=recovery.read(root/'workflow_state.json'); data.update(complete=False,failure='failed',finished_at=1)
            self.dump(root/'workflow_state.json',data)
        result=recovery.wait_sources([self.q,self.soft],emit=lambda *a,**k:None)
        self.assertTrue(all(row[1] for row in result))

    def test_duplicate_recovery_lock_rejected_and_released(self):
        with recovery.recovery_lock(self.base/'locks'):
            with self.assertRaisesRegex(RuntimeError,'Another'):
                with recovery.recovery_lock(self.base/'locks'):
                    pass
        with recovery.recovery_lock(self.base/'locks'):
            pass

    def test_child_environment_preserves_original_cublas(self):
        with mock.patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':'ambient'}):
            env=recovery.child_environment({'cublas_workspace_config':None})
            self.assertNotIn('CUBLAS_WORKSPACE_CONFIG',env)
        env=recovery.child_environment({'cublas_workspace_config':':4096:8'})
        self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
        self.assertEqual(env['CUDA_LAUNCH_BLOCKING'],'1')

    def test_child_environment_restores_original_visible_devices_and_threads(self):
        runtime=dict(cublas_workspace_config=None,environment={
            'CUDA_VISIBLE_DEVICES':'GPU-expected','NVIDIA_TF32_OVERRIDE':None,
            'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':None})
        with mock.patch.dict('os.environ',{'NVIDIA_TF32_OVERRIDE':'1','MKL_NUM_THREADS':'8'}):
            env=recovery.child_environment(runtime)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-expected')
        self.assertEqual(env['OMP_NUM_THREADS'],'4')
        self.assertNotIn('NVIDIA_TF32_OVERRIDE',env)
        self.assertNotIn('MKL_NUM_THREADS',env)


if __name__=='__main__':
    unittest.main()
