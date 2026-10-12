"""Synthetic integration: real retrieval, frozen readers, routing, and metrics."""
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from . import audit
from .core import hard_counts
from .test_core import frozen_head
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import encode_regions
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2.memory import device_bank, classify_regions


def fixture():
    torch.manual_seed(23)
    dim = 8
    # Three used slots: slot2 is soft-only; slot3 is strictly empty.
    q = torch.tensor([[[.7,.3,.65,.2], [.2,.6,.25,.7], [.1,.1,.1,.1]]])
    ids = torch.arange(3)[None,:,None].expand(1,3,4).clone()
    route = dict(q=q, ids=ids, k=4, shape=(2,2))
    head = Readout(dim).eval().requires_grad_(False)
    # Actual readout accepts 128-channel fused content independent of token dim.
    in_channels = head.embedding.score[0].in_channels if isinstance(head.embedding.score, torch.nn.Sequential) else head.embedding.score.in_channels
    features = torch.randn(1,in_channels,2,2)
    encoded = encode_regions(head, features, route)
    base = head.classifier(encoded['tokens'])
    bank = device_bank(dict(tokens=torch.randn(12,dim), distribution=F.one_hot(torch.arange(12)%4,4).float(),
        images=torch.arange(12), events=torch.arange(12)%3, hard_count=torch.ones(12,dtype=torch.long),
        labelled_mass=torch.ones(12), purity=torch.ones(12), entropy=torch.zeros(12),
        image_names=['bank_%d'%i for i in range(12)], event_names=['event_%d'%i for i in range(3)]), torch.device('cpu'))
    state = dict(heads=dict(normal=frozen_head(), nomemory=frozen_head()), banks=dict(hard=bank,all=bank),
        settings=dict(topk=8, query_chunk=2, seed=3407, allow_same_event=False))
    return head, features, route, encoded, base, state


class AuditIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.old_threads)

    def test_actual_predictions_reproduce_parent_all_interventions(self):
        _, _, route, encoded, base, state = fixture()
        counts=hard_counts(route); name='Val_event_sample'
        logits, observed, traces, replay = audit.classify_all(encoded['tokens'], encoded['mass'], base, counts,
            state, name, torch.Generator().manual_seed(3912))
        for case, mode, intervention in [('normal','balanced_hard','normal'),
                ('shuffle_labels','balanced_hard','shuffle_labels'), ('off','balanced_hard','off'),
                ('nomemory','nomemory','normal')]:
            reader=state['heads']['nomemory' if case=='nomemory' else 'normal']
            expected, _ = classify_regions(reader, encoded['tokens'], encoded['mass'], base,
                state['banks'], [name], [audit.common.event(name)], mode, 8, 2, True,
                intervention, torch.Generator().manual_seed(3912))
            torch.testing.assert_close(logits[case], expected, atol=0, rtol=0)
        self.assertEqual(sum(len(row['selected']) for row in traces),3)
        self.assertEqual(counts[0,2],0)
        for values in logits.values(): torch.testing.assert_close(values[0,3],base[0,3],atol=0,rtol=0)
        self.assertEqual(set(observed),{'normal','shuffle_labels','off'})
        self.assertLessEqual(replay['label_shuffle_attention_max_abs'],1e-6)

    def test_gt_free_api_and_no_gradients(self):
        self.assertFalse({'gt','target','labelled_mass'} & set(inspect.signature(audit.classify_all).parameters))
        _,_,route,encoded,base,state=fixture()
        before=copy.deepcopy(state['heads']['normal'].state_dict())
        audit.classify_all(encoded['tokens'],encoded['mass'],base,hard_counts(route),state,'image',torch.Generator().manual_seed(3))
        for key,val in state['heads']['normal'].state_dict().items():
            torch.testing.assert_close(val,before[key],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for h in state['heads'].values() for p in h.parameters()))

    def test_two_image_pipeline_counts_tail_and_pixel_pairs(self):
        head,features,route,encoded,base,state=fixture()
        names=['Val_event_sample1','Val_event_sample2']
        building=torch.tensor([[[[.8,.8],[.2,.2]]]])
        reference=dict(features=dict(roi=features), building=building,
            pixel=torch.randn(1,4,8,8), prob=torch.rand(1,4,8,8))
        bundle=dict(heads=dict(hard=head), front=object(), device=torch.device('cpu'),
            val=names, manifest=dict(arguments=dict(threshold=.5)))
        data=[]
        for name in names:
            data.append((torch.randn(1,3,8,8),torch.randn(1,3,8,8),None,
                torch.arange(64).reshape(1,8,8)%4,[name]))
        with patch.object(audit.common,'parse_batch',side_effect=lambda b,d:b), \
             patch.object(audit.common,'capture',return_value=reference), \
             patch.object(audit.common,'routes',return_value=(route,route)):
            result=audit.evaluate(bundle,state,data,SimpleNamespace(smoke=True))
        self.assertEqual(result['image_count'],2)
        self.assertEqual(result['routing']['active_slots'],6)
        self.assertEqual(result['routing']['tail_slots'],2)
        self.assertAlmostEqual(result['routing']['mean_pixel_tail_fraction'],.1,places=6)
        self.assertEqual(result['reader_stats']['normal']['all']['queries'],6)
        self.assertEqual(result['reader_stats']['normal']['tail']['queries'],2)
        self.assertEqual(len(result['final_metrics']),9)
        self.assertEqual(result['final_metrics']['normal']['count'],128)
        self.assertIn('nomemory__normal',result['diagnostics']['groups']['all']['comparisons'])
        self.assertEqual(result['diagnostics']['test_accessed'],False)

    def test_region_difference_handles_common_logit_offset_and_empty_tail(self):
        base=torch.zeros(1,2,4); candidate=base+100
        rows=audit.differences(base,candidate,torch.ones(1,2),torch.ones(1,2,dtype=torch.long))
        self.assertEqual(rows['all']['centered_delta_l1_sum'],0)
        self.assertEqual(rows['all']['changed_regions'],0)
        final=audit.finish_differences(dict(off=rows))
        self.assertIsNone(final['off']['groups']['tail']['prob_l1'])

    def test_smoke_orchestration_writes_only_new_diagnostics(self):
        head,features,route,_,_,state=fixture()
        state.update(files={}, summary=dict(final_step=5000))
        names=['Val_event_sample1','Val_event_sample2']
        reference=dict(features=dict(roi=features), building=torch.ones(1,1,2,2)*.8,
            pixel=torch.randn(1,4,8,8), prob=torch.rand(1,4,8,8))
        bundle=dict(heads=dict(hard=head), front=object(), device=torch.device('cpu'),
            val=names, manifest=dict(arguments=dict(threshold=.5),checkpoint_step=3000),
            provenance=dict(source_head_step=5000))
        data=[(torch.randn(1,3,8,8),torch.randn(1,3,8,8),None,
            torch.arange(64).reshape(1,8,8)%4,[name]) for name in names]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'new'
            args=SimpleNamespace(output=str(output),source_head='hard',pool_mode='hard',
                read_mode='hard',smoke=True,workers=0,dataset='synthetic')
            with patch.object(audit.common,'load_source',return_value=bundle), \
                 patch.object(audit,'load_trained',return_value=state), \
                 patch.object(audit,'baseline_policy',return_value=None), \
                 patch.object(audit.common,'source_identity',return_value={}), \
                 patch.object(audit.common,'loader',return_value=data), \
                 patch.object(audit.common,'parse_batch',side_effect=lambda b,d:b), \
                 patch.object(audit.common,'capture',return_value=reference), \
                 patch.object(audit.common,'routes',return_value=(route,route)), \
                 patch.object(audit,'initial_gate',return_value=dict(passed=None,status='smoke_not_full_val')), \
                 patch.object(audit,'assert_trained_unchanged') as unchanged:
                audit.run(args)
            unchanged.assert_called_once_with(state,bundle)
            summary=json.loads((output/'summary.json').read_text())
            self.assertTrue(summary['complete']); self.assertTrue(summary['smoke'])
            self.assertIsNone(summary['reproduction']['passed'])
            self.assertEqual(summary['memory_head_step'],5000)
            self.assertEqual(summary['front_checkpoint_step'],3000)
            self.assertTrue((output/'audit_final.json').is_file())
            self.assertTrue((output/'tail_pixel_bins.json').is_file())
            self.assertTrue(all(p.suffix=='.json' for p in output.iterdir()))


if __name__=='__main__': unittest.main()
