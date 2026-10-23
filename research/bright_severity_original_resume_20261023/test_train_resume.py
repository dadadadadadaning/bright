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

import evidence
import train_resume as resume


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
                     event='fixture', variant='original', original_loss=.1) for i in range(1,158)]
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
            return dict(path=str(path),step=step,roundtrip_verified=True,sha256='fixture')
        exp = types.SimpleNamespace(update=update, save_checkpoint=checkpoint,
                                    capture_training_failure=real.capture_training_failure)
        images = [evidence.TARGET_IMAGE if step == 158 else f'train_{step}' for step in range(1,2919)]
        runtime = types.SimpleNamespace(prepare=lambda image:types.SimpleNamespace(image=image,event='fixture'),
                                        val_names=[f'val_{i}' for i in range(450)])
        iterator = iter((0,images[step-1]) for step in range(151,2501))
        original = types.SimpleNamespace(_assert_example=lambda example:None)
        status = resume.initial_status(self.output,self.parent)
        status['last_checkpoint'] = dict(step=150,roundtrip_verified=True,path='source150.pt')
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

    def test_inherited_journal_contains_only_original1_to150_and_keeps_all_fields(self):
        rows = self.parent_rows()
        before = copy.deepcopy(rows)
        path = self.output/'train.jsonl'
        with path.open('x') as journal:
            resume.write_inherited_prefix(journal,rows,self.parent)
        inherited = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(inherited),150)
        for expected,actual in zip(rows,inherited):
            self.assertTrue(actual.pop('inherited'))
            self.assertEqual(actual.pop('inherited_source'),str(self.parent/'train.jsonl'))
            self.assertEqual(actual,expected)
        self.assertEqual(rows,before)

    def test_full_update_schedule_has_checkpoint_before_each_val_and_no_checkpoint158(self):
        status,events = self.invoke_loop(self.loop_fixture())
        self.assertEqual([step for kind,step in events if kind=='update'],list(range(151,2501)))
        self.assertEqual([step for kind,step in events if kind=='checkpoint'],list(range(200,2501,50)))
        self.assertEqual([step for kind,step in events if kind=='evaluate'],list(resume.EVALUATION_STEPS))
        for step in resume.EVALUATION_STEPS:
            position=events.index(('evaluate',step))
            self.assertEqual(events[position-1],('checkpoint',step))
        rows=[json.loads(line) for line in (self.output/'train.jsonl').read_text().splitlines()]
        self.assertEqual([row['step'] for row in rows],list(range(1,2501)))
        self.assertTrue(all(row['inherited'] is (row['step']<=150) for row in rows))
        self.assertTrue(status['training_complete'])
        self.assertFalse(status['complete'])
        self.assertEqual(status['new_complete_updates'],2350)
        self.assertEqual(status['last_checkpoint']['step'],2500)

    def test_first_update_failure_stops_without_retry_or_promotion(self):
        fixture=self.loop_fixture(fail_step=158)
        with self.assertRaisesRegex(RuntimeError,'failed update158'):
            self.invoke_loop(fixture)
        status,events=fixture[4],fixture[5]
        self.assertEqual([step for kind,step in events if kind=='update'],list(range(151,159)))
        self.assertEqual(status['last_complete_step'],157)
        self.assertEqual(status['new_complete_updates'],7)
        self.assertEqual(status['completed_update_calls'],7)
        self.assertEqual(status['last_checkpoint']['step'],150)
        self.assertTrue(status['failed_step_optimizer_execution_unknown'])
        self.assertFalse(status['complete'])
        self.assertFalse(status['training_complete'])
        self.assertEqual(evidence.read(self.output/'first_failure.json')['phase'],'update_158')
        self.assertEqual(len((self.output/'train.jsonl').read_text().splitlines()),157)

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
        self.assertEqual(status['completed_update_calls'],2350)
        self.assertFalse(status['failed_step_optimizer_execution_unknown'])

    def test_post_optimizer_error_counter_is_not_a_claim_of_zero_executed_updates(self):
        weight=torch.nn.Parameter(torch.tensor(1.))
        def failed(*args,**kwargs):
            with torch.no_grad(): weight.add_(1)
            raise RuntimeError('post-update finite check failed')
        status=resume.initial_status(self.output,self.parent)
        with self.assertRaises(RuntimeError):
            resume.original_update(types.SimpleNamespace(update=failed),None,None,None,None,None,None,None,
                                    step=151,status=status,multiplier=.2)
        self.assertEqual(weight.item(),2.)
        self.assertEqual(status['completed_update_calls'],0)
        self.assertEqual(status['last_complete_step'],150)
        self.assertTrue(status['failed_step_optimizer_execution_unknown'])

    def test_real_checkpoint_restore_model_adamw_rng_and_iterator_match_continuous_training(self):
        exp,state=self.experiment(),self.state_helpers()
        before_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,before_threads)
        torch.manual_seed(73); random.seed(73); np.random.seed(73)
        names=[f'train_{i}' for i in range(1,161)]
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
        for step in range(1,151):
            epoch,batch=next(iterator)
            value=update(model,optimizer,batch)
            rows.append(dict(step=step,optimizer_step=step,epoch=epoch,epoch_seed=3407+epoch,
                             image=batch[-1][0],variant='original',original_loss=value))
        checkpoint=self.output/'checkpoints/checkpoint_000150.pt'
        plan=dict(variant='original',effective_mode='original',read_weight=0)
        sidecar=exp.save_checkpoint(checkpoint,model,optimizer,state,runtime,plan,150)
        self.assertTrue(sidecar['roundtrip_verified'])
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        checkpoint_hash=evidence.sha(checkpoint)
        expected_losses=[]
        for _ in range(151,159):
            _,batch=next(iterator)
            expected_losses.append(update(model,optimizer,batch))
        expected=(state.state_fingerprint(model.state_dict()),state.state_fingerprint(optimizer.state_dict()),
                  state.state_fingerprint(state.capture_rng()))
        # Construction and raw-loader reconstruction intentionally disturb RNG.
        resumed=new_model()
        resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=.5)
        evidence.restore_model_optimizer(torch,state,resumed,resumed_optimizer,saved,150)
        manifest=dict(seed=3407,actual_training_names=names)
        resumed_iterator=evidence.reconstruct_iterator(exp,runtime,manifest,rows,150)
        self.assertNotEqual(state.state_fingerprint(state.capture_rng()),state.state_fingerprint(saved['rng']))
        state.restore_rng(saved['rng'])
        actual_losses=[]
        for step in range(151,159):
            _,batch=next(resumed_iterator)
            self.assertEqual(batch[-1][0],names[step-1])
            actual_losses.append(update(resumed,resumed_optimizer,batch))
        actual=(state.state_fingerprint(resumed.state_dict()),state.state_fingerprint(resumed_optimizer.state_dict()),
                state.state_fingerprint(state.capture_rng()))
        self.assertEqual(actual_losses,expected_losses)
        self.assertEqual(actual,expected)
        self.assertEqual(evidence.sha(checkpoint),checkpoint_hash)
        self.assertEqual(saved['format'],exp.FORMAT+'_CHECKPOINT')

    def result_fixture(self):
        status=resume.initial_status(self.output,self.parent)
        status.update(training_complete=True,validation_complete=True,fixed_train_complete=True,
                      last_complete_step=2500,last_journaled_step=2500,new_complete_updates=2350,
                      completed_update_calls=2350,optimizer_updates_returned=2350,
                      last_checkpoint={'step':2500,'roundtrip_verified':True})
        results=[]
        for name in [f'val_step_{step:06d}' for step in resume.EVALUATION_STEPS]+['fixed_train_endpoint']:
            folder=self.output/name
            folder.mkdir()
            report=dict(complete=False,phase='evaluation_complete_integrity_pending',final_integrity_pending=True,
                        whole_val450_processed=name.startswith('val'),test_accessed=False)
            evidence.write(folder/'result.json',report)
            results.append((folder/'result.json',report))
            status['evaluations'][name]={'execution_complete':True}
        return status,results

    def test_only_new_complete_evaluations_promoted_original_initial_val_untouched(self):
        status,results=self.result_fixture()
        initial=self.parent/'initial_val.json'
        evidence.write(initial,{'complete':False,'phase':'evaluation_complete_integrity_pending'})
        before=initial.read_bytes()
        resume.promote_new_results(None,self.output,status,results)
        self.assertTrue(status['complete'])
        self.assertTrue(status['full_validation_evidence'])
        self.assertFalse(status['study_complete'])
        self.assertFalse(status['test_accessed'])
        self.assertEqual(initial.read_bytes(),before)
        self.assertTrue(all(path.with_suffix('.json.sha256').is_file() for path,_ in results))

    def test_incomplete_results_are_not_promoted(self):
        status,results=self.result_fixture()
        with self.assertRaisesRegex(ValueError,'all five'):
            resume.promote_new_results(None,self.output,status,results[:-1])
        self.assertTrue(all(evidence.read(path)['complete'] is False for path,_ in results))
        self.assertFalse(status['complete'])

    def test_existing_output_or_input_directory_rejected_before_any_write(self):
        for target in (self.parent,self.output):
            sentinel={'phase':'failed','do_not_change':True}
            evidence.write(target/'summary.json',sentinel)
            args=types.SimpleNamespace(output=target,parent=self.parent,package=self.root,
                                       baseline=self.root/'observed',plain=self.root/'plain')
            with self.assertRaises(ValueError):
                resume.run_worker(args)
            self.assertEqual(evidence.read(target/'summary.json'),sentinel)

    def test_fixed_protocol_and_no_diagnostic_checkpoint(self):
        self.assertEqual(resume.RESTORED_STEP,150)
        self.assertEqual(resume.END_STEP,2500)
        self.assertEqual(resume.EVALUATION_STEPS,(500,1000,1500,2000,2500))
        self.assertFalse(resume.checkpoint_due(150))
        self.assertFalse(resume.checkpoint_due(158))
        self.assertTrue(resume.checkpoint_due(200))
        status=resume.initial_status(self.output,self.parent)
        for key in ('complete','study_complete','test_accessed','backward_index_observer','anomaly_detection',
                    'saved_tensors_hooks_used','historical_failure_resolved','automatic_retry','other_variants_started'):
            self.assertIs(status[key],False,key)


if __name__ == '__main__':
    unittest.main(verbosity=2)
