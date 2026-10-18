"""CPU integration of audit wiring; no original dataset or server paths needed."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

import audit
from test_slot_stats import fixture as slot_fixture


class TinyHead(torch.nn.Module):
    """Uniform-attention fixture retaining the historical Reader interface."""
    def __init__(self):
        super().__init__()
        self.hidden, self.heads = 4, 1
        self.query_projection = torch.nn.Linear(4, 4, bias=False)
        self.key_projection = torch.nn.Linear(4, 4, bias=False)
        self.value_projection = torch.nn.Identity()
        self.label_embedding = torch.nn.Identity()
        self.context_projection = torch.nn.Identity()
        torch.nn.init.zeros_(self.query_projection.weight)
        torch.nn.init.zeros_(self.key_projection.weight)
        self.eval().requires_grad_(False)

    def _encode_slots(self, feature, parent, fraction, valid):
        return torch.where(valid[..., None], feature[..., :4], 0.)

    def forward(self, **fields):
        if any(name in fields for name in ('target', 'labels_gt', 'query_labels')):
            raise AssertionError('Query GT entered the head')
        query = self._encode_slots(fields['slot_f'][0], fields['region_t'][0],
                                   fields['slot_fraction'][0], fields['slot_valid'][0])
        ids = fields['support_ids']
        safe = ids.clamp_min(0)
        table = self._encode_slots(fields['support_f'], fields['support_t'],
                                   fields['support_fraction'], fields['support_valid'])
        labels = torch.where(fields['support_valid'][..., None], fields['support_labels'], 0.)
        mask = (ids[..., None] >= 0) & fields['support_valid'][safe]
        values = (table + labels)[safe]
        pooled = (values * mask[..., None]).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)[:, None]
        context = pooled[:, None].expand(-1, 2, -1)
        context = torch.where(fields['slot_valid'][0, ..., None], context, 0.)
        base = fields['base_probability_full']
        # Prediction actually depends on candidate labels, while query GT is absent.
        delta = context[:, 1].T.reshape(1, 4, 1, -1)
        probability = (base.log() + delta).softmax(1)
        return dict(probabilities=probability, slot_encoding=query[None],
                    memory_context=context[None])


def example_fixture(name='eventA_val'):
    feature = torch.ones(1, 256, 1, 2)
    feature[:, 0, 0] = torch.tensor([1., 3.])
    building = torch.tensor([[[[.25, .75]]]])
    target = torch.tensor([[[1, 2]]])
    q = torch.zeros(1, 9, 2)
    q[:, 0] = 1
    ids = torch.zeros(9, 2, dtype=torch.long)
    ids[0] = torch.tensor([0, 1])
    feature, q, ids, building, target, slots = slot_fixture(feature, building, target, q=q, ids=ids, count=2)
    common = dict(pixel_features=feature, region_t=torch.zeros(1, 2, 128),
                  region_f=feature[0].flatten(1).T[None], q=q, ids=ids,
                  base_probability_full=torch.tensor([[[[.1, .1]], [[.6, .2]], [[.2, .6]], [[.1, .1]]]]))
    query = {k:slots[k] for k in ('slot_f', 'slot_fraction', 'slot_valid', 'route_q', 'route_ids')}
    ids_selected = torch.tensor([[0, 1], [1, 0]])
    local_labels = torch.tensor([[[.1, .6, .2, .1]]*2, [[.1, .2, .6, .1]]*2])
    memory_labels = torch.tensor([[[.1, .1, .7, .1]]*2, [[.1, .7, .1, .1]]*2])
    shared = dict(support_ids=ids_selected, support_f=slots['slot_f'][0],
                  support_t=common['region_t'][0], support_fraction=slots['slot_fraction'][0],
                  support_valid=slots['slot_valid'][0])
    supports = {arm:dict(**query, **shared, support_labels=labels)
                for arm,labels in (('b',local_labels),('d',memory_labels))}
    example = SimpleNamespace(image=name, event=name.split('_')[0], target=target,
                              common=common, supports=supports,
                              diagnostics={'query_GT_used_for_supports':False})
    data = dict(feature=feature,result={'q':q,'ids':ids},building=building,slots={'soft':slots})
    return example, data


class AuditIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.heads = {arm:TinyHead() for arm in ('b','d')}
        self.images = ['train0', 'train1']
        self.events = ['trainEvent0', 'trainEvent1']

    def analyze(self, example, data):
        return audit.analyze_example(example,data,self.heads,self.images,self.events,2)

    def test_real_helpers_together_and_query_gt_does_not_change_predictions(self):
        example,data = example_fixture()
        first = self.analyze(example,data)
        changed = copy.copy(example)
        changed.target = torch.tensor([[[3, 1]]])
        second = self.analyze(changed,data)
        for key in first['predictions']:
            self.assertTrue((first['predictions'][key] == second['predictions'][key]).all())
        self.assertFalse(torch.equal(first['slots']['query_labels'],second['slots']['query_labels']))
        for key in ('reader_d_damaged_label_mass','reader_b_damaged_label_mass',
                    'memory_unique_source_images','memory_cosine_vote_damaged'):
            torch.testing.assert_close(first['rows'][key],second['rows'][key])
        self.assertFalse(torch.equal(first['rows']['local_gt_oracle_cosine_vote_damaged'],
                                     second['rows']['local_gt_oracle_cosine_vote_damaged']))
        self.assertEqual(first['neighbor_definitions']['parent_budget'],2)

    def test_selected_cases_use_original_example_image_field_and_no_nan_json(self):
        example,data = example_fixture()
        measured = self.analyze(example,data)
        cases = audit.selected_cases(example,measured,self.images,self.events)
        self.assertGreater(len(cases),0)
        self.assertTrue(all(row['image'] == example.image for row in cases))
        self.assertTrue(all(row['actual_parent_budget'] == 2 for row in cases))
        json.dumps(cases,allow_nan=False)

    def test_nonfinite_prediction_is_rejected_before_argmax(self):
        example,data=example_fixture()
        original=self.heads['d'].forward
        def broken(**fields):
            result=original(**fields)
            result['probabilities']=result['probabilities'].clone()
            result['probabilities'][0,0,0,0]=float('nan')
            return result
        with mock.patch.object(self.heads['d'],'forward',side_effect=broken):
            with self.assertRaisesRegex(ValueError,'Invalid classifier probability'):
                self.analyze(example,data)

    def test_worker_smoke_real_aggregation_and_completion_contract(self):
        import source_io
        heads = self.heads
        class Runtime:
            def __init__(self,args,output):
                self.args,self.output = args,output
                self.train_names = ['train0','train1']
                self.val_names = ['eventA_val','eventB_val']
                self.device = torch.device('cpu')
                self.protocol_info = {'fixture':True}
                self.input_ledger = {Path('/fixture'): 'fixture-hash'}
                self.deps = SimpleNamespace(event=lambda name:name.split('_')[0])
                self.bank = {'images':torch.tensor([0,1]),'events':torch.tensor([0,1]),
                             'event_names':['trainEvent0','trainEvent1']}
                ex,_ = example_fixture()
                table = ex.supports['d']
                self.values = {'rules':{'soft':{'slot_labels':table['support_labels'],
                                               'slot_valid':table['support_valid']}}}
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def smoke_names(self): return self.train_names,self.val_names
            def batches(self,names,train=False,seed=0):
                if train: raise AssertionError('Diagnostic must not forward Train')
                yield from names
            def extract(self,name): return example_fixture(name)[1]
            def prepare(self,name):
                self.extract(name)  # Parent CaptureRuntime must see this call.
                return example_fixture(name)[0]
        def blank():
            return {arm:{'supported_parent_queries':0} for arm in ('a','b','c','d')}
        def coverage(example):
            return {arm:{'supported_parent_queries':2} for arm in ('a','b','c','d')}
        def merge(total,current):
            for arm in total:
                total[arm]['supported_parent_queries'] += current[arm]['supported_parent_queries']
        train = SimpleNamespace(_formal_runtime=lambda runtime,smoke:None,
            _assert_example=lambda example:None,_blank_coverage=blank,_coverage=coverage,
            _merge_coverage=merge,_checked_verify=lambda runtime:{'frozen':True},
            _restored_guard=lambda runtime:{'restored':True})
        common = SimpleNamespace(module_hash=lambda head:'unchanged-fixture-head')
        modules = SimpleNamespace(torch=torch,Runtime=Runtime,train=train,common=common)
        saved = {'config':{'seed':3407,'chunk_size':2},'protocol_info':{'fixture':True},
                 'input_file_sha256':{'/fixture':'fixture-hash'}}
        static = {'root':'/fixture/source','tool':'/fixture/tool',
                  'manifest':{'arguments':{},'train_names':['train0','train1'],'val_names':['eventA_val','eventB_val']},
                  'meta':{'head_hashes':{'b':'unchanged-fixture-head','d':'unchanged-fixture-head'}},
                  'old_files':{}}
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            output = Path(folder)/'smoke'
            args = SimpleNamespace(output=str(output),step=500,smoke=True,train_root='fixture',
                                   tool_dir='fixture',command='worker')
            stack.enter_context(mock.patch.object(audit,'release_hashes',return_value={'fixture':'hash'}))
            stack.enter_context(mock.patch.object(audit,'source_static',return_value=static))
            stack.enter_context(mock.patch.object(audit,'prepare_heads',return_value=heads))
            stack.enter_context(mock.patch.object(source_io,'import_runtime',return_value=modules))
            stack.enter_context(mock.patch.object(source_io,'load_saved',return_value=saved))
            stack.enter_context(mock.patch.object(source_io,'runtime_arguments',return_value=SimpleNamespace(seed=3407,chunk_size=2)))
            stack.enter_context(mock.patch.object(source_io,'check_source_unchanged',return_value=True))
            with redirect_stdout(io.StringIO()):
                audit.worker(args)
            result = json.loads((output/'result.json').read_text())
            summary = json.loads((output/'summary.json').read_text())
            self.assertTrue(result['complete'])
            self.assertTrue(summary['complete'])
            self.assertFalse(summary['full_validation_evidence'])
            self.assertFalse(summary['final_integrity_pending'])
            self.assertEqual(result['image_count'],2)
            self.assertEqual(result['coverage']['d']['supported_parent_queries'],4)
            self.assertEqual(result['pixels']['all_labeled']['metrics']['d']['valid_pixels'],4)
            self.assertEqual(set(result['per_event']),{'eventA','eventB'})
            self.assertTrue((output/'REPORT.txt').is_file())
            self.assertEqual(len((output/'images.jsonl').read_text().splitlines()),2)
            self.assertTrue(audit.validate_worker_result(output,smoke=True,step=500))
            summary['final_integrity_pending'] = True
            (output/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'required diagnostic evidence'):
                audit.validate_worker_result(output,smoke=True,step=500)

    def test_output_cannot_overlap_sources_or_banks(self):
        static={'root':'/fixture/run','tool':'/fixture/tool',
                'manifest':{'arguments':{'bank_root':'/fixture/bank','values':'/fixture/values'}}}
        for path in ('/fixture/run/new','/fixture/bank/new','/fixture','/fixture/tool/new'):
            with self.assertRaisesRegex(ValueError,'overlaps protected'):
                audit.validate_destination(path,static)
        audit.validate_destination('/fixture/audit/new',static)

    def test_exit_zero_report_without_replay_is_not_formal_evidence(self):
        import source_io
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            result=dict(format=audit.FORMAT,complete=True,execution_complete=True,
                        final_integrity_pending=False,full_validation_evidence=True,
                        training=False,test_accessed=False,smoke=False,checkpoint_step=500,
                        image_count=450,replay={'passed':False})
            audit.write_json(root/'result.json',result)
            audit.write_json(root/'summary.json',dict(result,result_sha256=source_io.digest(root/'result.json')))
            with self.assertRaisesRegex(ValueError,'original Val450 replay'):
                audit.validate_worker_result(root,smoke=False,step=500)
            result['replay']['passed']=True
            audit.write_json(root/'result.json',result)
            with self.assertRaisesRegex(ValueError,'checksum'):
                audit.validate_worker_result(root,smoke=False,step=500)


if __name__=='__main__':
    unittest.main()
