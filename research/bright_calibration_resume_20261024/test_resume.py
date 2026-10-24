"""CPU continuation contracts; these do not establish remote CUDA stability."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

import helpers
import resume


class TrainResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent = self.root/'parent'
        self.parent.mkdir()
        self.output = self.root/'output'
        self.output.mkdir()

    def load_file(self, path, label):
        path = Path(path)
        if not path.is_file():
            self.skipTest('Required original CPU integration source unavailable: '+str(path))
        spec = importlib.util.spec_from_file_location(label, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def experiment(self):
        package = Path(os.environ.get('BRIGHT_SEVERITY_PACKAGE',
                                      '/workspace/bright_severity_memory_20261023'))
        return self.load_file(package/'experiment.py', '_formal_resume_test_exp')

    def state_helpers(self):
        path = os.environ.get('BRIGHT_REFERENCE_STATE',
            '/workspace/bright-design-delivery/joint_anchor_fresh_chain_v2/deps/helpers/continuation.py')
        return self.load_file(path, '_formal_resume_test_state')

    def parent_rows(self):
        rows = [dict(step=i, optimizer_step=i, epoch=0, epoch_seed=3407, image=f'train_{i}',
                     event='fixture', variant='calibration', original_loss=.1) for i in range(1,804)]
        return rows

    def loop_fixture(self, *, fail_step=None, fail_val=None):
        real = self.experiment()
        events = []
        def update(*args,step,**kwargs):
            events.append(('update',step))
            if step == fail_step:
                raise RuntimeError(f'failed update{step}')
            return dict(original_loss=.1, training_loss=.1, optimizer_step=step, lr=step*2e-7)
        def checkpoint(path,model,optimizer,state,runtime,plan,step):
            events.append(('checkpoint',step))
            path.parent.mkdir(exist_ok=True)
            return dict(path=str(path),step=step,roundtrip_verified=True,sha256='fixture')
        exp = types.SimpleNamespace(update=update, save_checkpoint=checkpoint,
                                    capture_training_failure=real.capture_training_failure)
        images = [resume.TARGET_IMAGE if step == 804 else f'train_{step}' for step in range(1,2919)]
        runtime = types.SimpleNamespace(prepare=lambda image:types.SimpleNamespace(image=image,event='fixture'),
                                        val_names=[f'val_{i}' for i in range(450)])
        iterator = iter((0,images[step-1]) for step in range(801,2501))
        original = types.SimpleNamespace(_assert_example=lambda example:None)
        status = resume.initial_status(self.output,self.parent)
        status['last_checkpoint'] = dict(step=800,roundtrip_verified=True,path='source800.pt')
        def assess(step,split,names,folder):
            status['phase']=f'evaluate_{split}_{step}'
            events.append(('evaluate',step))
            if step == fail_val:
                raise RuntimeError(f'failed Val{step}')
        return exp,runtime,iterator,original,status,events,assess,set(images)

    def invoke_loop(self, fixture):
        exp,runtime,iterator,original,status,events,assess,images = fixture
        with (self.output/'train.jsonl').open('x') as journal, contextlib.redirect_stdout(io.StringIO()):
            resume.write_inherited_prefix(journal,self.parent_rows(),self.parent)
            with exp.capture_training_failure(self.output,status):
                resume.run_updates(exp,None,None,runtime,None,None,None,original,iterator,images,
                    self.parent_rows(),{'seed':3407,'post500_lr_multiplier':.2},self.output,status,journal,assess)
        return status,events

    def test_inherited_journal_contains_only_original1_to800_and_keeps_all_fields(self):
        rows = self.parent_rows()
        before = copy.deepcopy(rows)
        path = self.output/'train.jsonl'
        with path.open('x') as journal:
            resume.write_inherited_prefix(journal,rows,self.parent)
        inherited = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(inherited),800)
        for expected,actual in zip(rows,inherited):
            self.assertTrue(actual.pop('inherited'))
            self.assertEqual(actual.pop('inherited_source'),str(self.parent/'train.jsonl'))
            self.assertEqual(actual,expected)
        self.assertEqual(rows,before)

    def test_full_update_schedule_has_checkpoint_before_each_val_and_no_checkpoint804(self):
        status,events = self.invoke_loop(self.loop_fixture())
        self.assertEqual([step for kind,step in events if kind=='update'],list(range(801,2501)))
        self.assertEqual([step for kind,step in events if kind=='checkpoint'],list(range(850,2501,50)))
        self.assertEqual([step for kind,step in events if kind=='evaluate'],list(resume.EVALUATION_STEPS))
        for step in resume.EVALUATION_STEPS:
            position=events.index(('evaluate',step))
            self.assertEqual(events[position-1],('checkpoint',step))
        rows=[json.loads(line) for line in (self.output/'train.jsonl').read_text().splitlines()]
        self.assertEqual([row['step'] for row in rows],list(range(1,2501)))
        self.assertTrue(all(row['inherited'] is (row['step']<=800) for row in rows))
        self.assertTrue(status['training_complete'])
        self.assertFalse(status['complete'])
        self.assertEqual(status['new_complete_updates'],1700)
        self.assertEqual(status['last_checkpoint']['step'],2500)
        self.assertEqual(status['last_checkpoint']['execution_change'], resume.EXECUTION_CHANGE)
        self.assertTrue(all(row.get('execution_change') == resume.EXECUTION_CHANGE
                            for row in rows if not row['inherited']))
        for step in range(850, 2501, 50):
            sidecar = helpers.read(self.output / 'checkpoints' / f'checkpoint_{step:06d}.pt.json')
            self.assertEqual(sidecar['execution_change'], resume.EXECUTION_CHANGE)
            self.assertTrue(sidecar['roundtrip_verified'])

    def test_first_update_failure_stops_without_retry_or_promotion(self):
        fixture=self.loop_fixture(fail_step=804)
        with self.assertRaisesRegex(RuntimeError,'failed update804'):
            self.invoke_loop(fixture)
        status,events=fixture[4],fixture[5]
        self.assertEqual([step for kind,step in events if kind=='update'],list(range(801,805)))
        self.assertEqual(status['last_complete_step'],803)
        self.assertEqual(status['new_complete_updates'],3)
        self.assertEqual(status['completed_update_calls'],3)
        self.assertEqual(status['last_checkpoint']['step'],800)
        self.assertTrue(status['failed_step_optimizer_execution_unknown'])
        self.assertFalse(status['complete'])
        self.assertFalse(status['training_complete'])
        self.assertEqual(helpers.read(self.output/'first_failure.json')['phase'],'update_804')
        self.assertEqual(len((self.output/'train.jsonl').read_text().splitlines()),803)

    def test_endpoint_validation_failure_preserves_completed_training_and_checkpoint(self):
        fixture=self.loop_fixture(fail_val=2500)
        with self.assertRaisesRegex(RuntimeError,'failed Val2500'):
            self.invoke_loop(fixture)
        status=fixture[4]
        self.assertTrue(status['training_complete'])
        self.assertFalse(status['validation_complete'])
        self.assertFalse(status['full_validation_evidence'])
        self.assertFalse(status['complete'])
        self.assertEqual(status['last_complete_step'],2500)
        self.assertEqual(status['last_checkpoint']['step'],2500)
        self.assertEqual(status['completed_update_calls'],1700)
        self.assertFalse(status['failed_step_optimizer_execution_unknown'])

    def test_parent801_to803_and_known804_order_mismatch_stop_before_update(self):
        for changed_step in (801, 804):
            fixture = list(self.loop_fixture())
            names = [resume.TARGET_IMAGE if step == 804 else f'train_{step}'
                     for step in range(801,2501)]
            names[changed_step-801] = 'train_42'  # Legal Train member, incorrect order.
            fixture[2] = iter((0, name) for name in names)
            journal_path = self.output/'train.jsonl'
            if journal_path.exists():
                journal_path.unlink()
            failure_path = self.output/'first_failure.json'
            if failure_path.exists():
                failure_path.unlink()
            with self.assertRaisesRegex(ValueError, 'order'):
                self.invoke_loop(fixture)
            self.assertEqual(fixture[4]['last_complete_step'], changed_step-1)
            self.assertNotIn(('update',changed_step),fixture[5])
            self.assertFalse(fixture[4]['failed_step_optimizer_execution_unknown'])

    def test_post_optimizer_error_counter_is_not_a_claim_of_zero_executed_updates(self):
        weight=torch.nn.Parameter(torch.tensor(1.))
        def failed(*args,**kwargs):
            with torch.no_grad(): weight.add_(1)
            raise RuntimeError('post-update finite check failed')
        status=resume.initial_status(self.output,self.parent)
        with self.assertRaises(RuntimeError):
            resume.calibration_update(types.SimpleNamespace(update=failed),None,None,None,None,None,None,None,
                                    step=801,status=status,multiplier=.2)
        self.assertEqual(weight.item(),2.)
        self.assertEqual(status['completed_update_calls'],0)
        self.assertEqual(status['last_complete_step'],800)
        self.assertTrue(status['failed_step_optimizer_execution_unknown'])

    def test_real_checkpoint_restore_model_adamw_rng_and_iterator_match_continuous_training(self):
        exp,state=self.experiment(),self.state_helpers()
        before_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,before_threads)
        torch.manual_seed(73); random.seed(73); np.random.seed(73)
        names=[f'train_{i}' for i in range(1,811)]
        class Runtime:
            protocol_info={'cpu_integration':True}
            def batches(self,requested,train,seed):
                for index,name in enumerate(requested):
                    # Simulate loader/global RNG consumption while raw image data
                    # remain deterministic, as in the augmentation-free recipe.
                    random.random(); np.random.rand(); torch.rand(())
                    x=torch.tensor([[.1+index*.001,.2,.3]])
                    yield x,[name]
        runtime=Runtime()
        def new_model():
            return torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Dropout(.35),torch.nn.Linear(5,2))
        model=new_model()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.005)
        iterator=exp.batches(runtime,names,3407)
        rows=[]
        def update(head,opt,batch):
            head.train(); opt.zero_grad(set_to_none=True)
            loss=head(batch[0]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            return float(loss.detach())
        for step in range(1,801):
            epoch,batch=next(iterator)
            value=update(model,optimizer,batch)
            rows.append(dict(step=step,optimizer_step=step,epoch=epoch,epoch_seed=3407+epoch,
                             image=batch[-1][0],variant='calibration',original_loss=value))
        checkpoint=self.output/'checkpoints/checkpoint_000800.pt'
        plan=dict(variant='calibration',effective_mode='calibration',read_weight=0)
        sidecar=exp.save_checkpoint(checkpoint,model,optimizer,state,runtime,plan,800)
        self.assertTrue(sidecar['roundtrip_verified'])
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        checkpoint_hash=helpers.sha(checkpoint)
        expected_losses=[]
        for _ in range(801,805):
            _,batch=next(iterator)
            expected_losses.append(update(model,optimizer,batch))
        expected=(state.state_fingerprint(model.state_dict()),state.state_fingerprint(optimizer.state_dict()),
                  state.state_fingerprint(state.capture_rng()))
        # Construction and raw-loader reconstruction intentionally disturb RNG.
        resumed=new_model()
        resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=.5)
        helpers.restore_model_optimizer(torch,state,resumed,resumed_optimizer,saved,800)
        manifest=dict(seed=3407,variant='calibration',plan={'variant':'calibration'},actual_training_names=names)
        resumed_iterator=helpers.reconstruct_iterator(exp,runtime,manifest,rows,800)
        self.assertNotEqual(state.state_fingerprint(state.capture_rng()),state.state_fingerprint(saved['rng']))
        state.restore_rng(saved['rng'])
        actual_losses=[]
        for step in range(801,805):
            _,batch=next(resumed_iterator)
            self.assertEqual(batch[-1][0],names[step-1])
            actual_losses.append(update(resumed,resumed_optimizer,batch))
        actual=(state.state_fingerprint(resumed.state_dict()),state.state_fingerprint(resumed_optimizer.state_dict()),
                state.state_fingerprint(state.capture_rng()))
        self.assertEqual(actual_losses,expected_losses)
        self.assertEqual(actual,expected)
        self.assertEqual(helpers.sha(checkpoint),checkpoint_hash)
        self.assertEqual(saved['format'],exp.FORMAT+'_CHECKPOINT')

    def result_fixture(self):
        status=resume.initial_status(self.output,self.parent)
        status.update(training_complete=True,validation_complete=True,fixed_train_complete=True,
                      last_complete_step=2500,last_journaled_step=2500,new_complete_updates=1700,
                      completed_update_calls=1700,optimizer_updates_returned=1700,
                      last_checkpoint={'step':2500,'roundtrip_verified':True})
        results=[]
        for name in [f'val_step_{step:06d}' for step in resume.EVALUATION_STEPS]+['fixed_train_endpoint']:
            folder=self.output/name
            folder.mkdir()
            report=dict(complete=False,phase='evaluation_complete_integrity_pending',final_integrity_pending=True,
                        whole_val450_processed=name.startswith('val'),test_accessed=False)
            helpers.write(folder/'result.json',report)
            results.append((folder/'result.json',report))
            status['evaluations'][name]={'execution_complete':True}
        return status,results

    def test_only_new_evaluations_promoted_inherited_initial_and_val500_untouched(self):
        status,results=self.result_fixture()
        initial=self.parent/'initial_val.json'
        helpers.write(initial,{'complete':False,'phase':'evaluation_complete_integrity_pending'})
        before=initial.read_bytes()
        val500 = self.parent/'val_step_000500.json'
        helpers.write(val500, {'step':500, 'complete':False, 'final_integrity_pending':True})
        val500_before = val500.read_bytes()
        status['inherited_evaluations'] = {'initial_val': {'path':str(initial), 'complete':False},
                                          'val_step_000500': {'path':str(val500), 'complete':False}}
        resume.promote_new_results(None,self.output,status,results)
        self.assertTrue(status['complete'])
        self.assertTrue(status['full_validation_evidence'])
        self.assertFalse(status['study_complete'])
        self.assertFalse(status['test_accessed'])
        self.assertEqual(initial.read_bytes(),before)
        self.assertEqual(val500.read_bytes(),val500_before)
        self.assertFalse(status['inherited_evaluations']['val_step_000500']['complete'])
        self.assertTrue(all(path.with_suffix('.json.sha256').is_file() for path,_ in results))

    def test_incomplete_results_are_not_promoted(self):
        status,results=self.result_fixture()
        with self.assertRaisesRegex(ValueError,'all four new'):
            resume.promote_new_results(None,self.output,status,results[:-1])
        self.assertTrue(all(helpers.read(path)['complete'] is False for path,_ in results))
        self.assertFalse(status['complete'])

    def test_new_evaluation_is_calibration450_with_execution_identity_and_no_val500_rerun(self):
        names = [f'val_{i}' for i in range(450)]
        runtime = types.SimpleNamespace(val_names=names)
        status = resume.initial_status(self.output,self.parent)
        status['last_checkpoint'] = {'step':1000, 'execution_change':resume.EXECUTION_CHANGE}
        calls=[]
        def evaluate(runtime,model,factory,state,original,requested,folder,**kwargs):
            calls.append(kwargs)
            folder.mkdir()
            return dict(execution_complete=True, image_count=len(requested), requested_names=requested,
                        journal_verified=True, test_accessed=False, head_state_unchanged=True,
                        rng_restored=True, whole_val450_processed=True, metrics={},
                        complete=False, final_integrity_pending=True, **kwargs)
        exp=types.SimpleNamespace(evaluate=evaluate)
        results=[]
        with contextlib.redirect_stdout(io.StringIO()):
            report=resume.assess_calibration(exp,runtime,None,None,None,None,names,self.output,status,
                results,step=1000,split='val',folder='val_step_001000')
        self.assertEqual(report['execution_change'],resume.EXECUTION_CHANGE)
        self.assertEqual(calls[0]['variant'],'calibration')
        self.assertFalse(report['complete'])
        self.assertFalse(status['validation_complete'])
        self.assertEqual(len(results),1)
        for step, requested in ((500,names),(1500,names[:2])):
            with self.assertRaisesRegex(ValueError,'whole Val450'):
                resume.assess_calibration(exp,runtime,None,None,None,None,requested,self.output,status,
                    results,step=step,split='val',folder=f'val_step_{step:06d}')
        self.assertEqual(len(calls),1)

    def test_existing_output_or_input_directory_rejected_before_any_write(self):
        for target in (self.parent,self.output):
            sentinel={'phase':'failed','do_not_change':True}
            helpers.write(target/'summary.json',sentinel)
            args=types.SimpleNamespace(output=target,parent=self.parent,package=self.root)
            with self.assertRaises(ValueError):
                resume.run_worker(args)
            self.assertEqual(helpers.read(target/'summary.json'),sentinel)

    def test_fixed_protocol_and_no_diagnostic_checkpoint(self):
        self.assertEqual(resume.RESTORED_STEP,800)
        self.assertEqual(resume.END_STEP,2500)
        self.assertEqual(resume.EVALUATION_STEPS,(1000,1500,2000,2500))
        self.assertFalse(resume.checkpoint_due(800))
        self.assertFalse(resume.checkpoint_due(804))
        self.assertTrue(resume.checkpoint_due(850))
        status=resume.initial_status(self.output,self.parent)
        for key in ('complete','study_complete','test_accessed','backward_index_observer','anomaly_detection',
                    'saved_tensors_hooks_used','historical_failure_resolved','automatic_retry','other_variants_started'):
            self.assertIs(status[key],False,key)


if __name__ == '__main__':
    unittest.main(verbosity=2)
