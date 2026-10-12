"""Small CPU integration with real readout, retrieval, routing and paired metrics."""
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from . import audit
from bda_benchmark.experiments.damageformer_memory_readout_audit_v1.test_audit import fixture
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2.memory import classify_regions


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_production_normal_nomemory_match_parent(self):
        _, _, _, encoded, base, state = fixture()
        expected_normal, _ = classify_regions(state['heads']['normal'], encoded['tokens'], encoded['mass'],
            base, state['banks'], ['Val_event_image'], [audit.common.event('Val_event_image')],
            'balanced_hard', 8, 2, True)
        expected_no, _ = classify_regions(state['heads']['nomemory'], encoded['tokens'], encoded['mass'],
            base, state['banks'], ['Val_event_image'], [audit.common.event('Val_event_image')],
            'nomemory', 8, 2, True)
        with patch.object(audit, 'retrieve', wraps=audit.retrieve) as retrieval:
            logits, chunks, replay = audit.classify_all(encoded['tokens'], encoded['mass'], base, state, 'Val_event_image')
        self.assertEqual(retrieval.call_count, 2)  # 3 active slots, chunks of 2; NOT once per case.
        torch.testing.assert_close(logits['normal'], expected_normal, atol=0, rtol=0)
        torch.testing.assert_close(logits['nomemory'], expected_no, atol=0, rtol=0)
        self.assertEqual(sum(len(row['selected']) for row in chunks), 3)
        for values in logits.values():
            torch.testing.assert_close(values[0, 3], base[0, 3], atol=0, rtol=0)
        self.assertEqual(replay['identity_region_argmax_disagreements'], 0)

    def test_gt_free_api_and_frozen_weights(self):
        self.assertFalse({'gt', 'target', 'building', 'truth'} & set(inspect.signature(audit.classify_all).parameters))
        _, _, _, encoded, base, state = fixture()
        before = copy.deepcopy(state['heads']['normal'].state_dict())
        audit.classify_all(encoded['tokens'], encoded['mass'], base, state, 'Val_event_image')
        for key, value in state['heads']['normal'].state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for h in state['heads'].values() for p in h.parameters()))

    def pipeline(self):
        head, features, route, _, _, state = fixture()
        state.update(files={}, summary=dict(final_step=5000))
        names = ['Val_event_image1', 'Val_event_image2']
        reference = dict(features=dict(roi=features), building=torch.ones(1, 1, 2, 2)*.8,
            pixel=torch.randn(1, 4, 8, 8), prob=torch.rand(1, 4, 8, 8))
        bundle = dict(heads=dict(hard=head), front=object(), device=torch.device('cpu'), val=names,
            manifest=dict(arguments=dict(threshold=.5), checkpoint_step=3000), provenance=dict(source_head_step=5000))
        data = [(torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8), None,
                 torch.arange(64).reshape(1, 8, 8)%4, [name]) for name in names]
        return bundle, state, data, reference, route

    def test_evaluate_full_readout_and_paired_diagnostics(self):
        bundle, state, data, reference, route = self.pipeline()
        q_before = route['q'].clone()
        with patch.object(audit.common, 'parse_batch', side_effect=lambda b, d: b), \
             patch.object(audit.common, 'capture', return_value=reference), \
             patch.object(audit.common, 'routes', return_value=(route, route)):
            result = audit.evaluate(bundle, state, data, SimpleNamespace(smoke=True))
        self.assertEqual(result['image_count'], 2)
        self.assertEqual(result['routing']['active_slots'], 6)
        self.assertEqual(result['routing']['tail_slots'], 2)
        self.assertEqual(len(result['final_metrics']), 8)
        self.assertTrue(result['identity_passed'])
        self.assertEqual(result['replay']['identity_pixel_argmax_disagreements'], 0)
        self.assertEqual(result['reader_stats']['cosine']['all']['queries'], 6)
        self.assertEqual(result['reader_stats']['cosine']['tail']['queries'], 2)
        pairs = result['diagnostics']['groups']['all']['comparisons']
        self.assertIn('uniform__cosine', pairs)
        self.assertIn('nomemory__cosine', pairs)
        self.assertEqual(pairs['normal__identity']['net'], 0)
        torch.testing.assert_close(route['q'], q_before, atol=0, rtol=0)

    def test_run_smoke_only_writes_new_json(self):
        bundle, state, data, reference, route = self.pipeline()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new'
            args = SimpleNamespace(output=str(output), source_head='hard', pool_mode='hard',
                read_mode='hard', smoke=True, workers=0, dataset='synthetic')
            parent = dict(files={})
            with patch.object(audit.common, 'load_source', return_value=bundle), \
                 patch.object(audit, 'load_trained', return_value=state), \
                 patch.object(audit, 'verify_parent_state'), \
                 patch.object(audit, 'assert_parent_unchanged') as parent_check, \
                 patch.object(audit, 'baseline_policy', return_value=None), \
                 patch.object(audit.common, 'source_identity', return_value={}), \
                 patch.object(audit.common, 'loader', return_value=data), \
                 patch.object(audit.common, 'parse_batch', side_effect=lambda b, d: b), \
                 patch.object(audit.common, 'capture', return_value=reference), \
                 patch.object(audit.common, 'routes', return_value=(route, route)), \
                 patch.object(audit, 'initial_gate', return_value=dict(passed=None)), \
                 patch.object(audit, 'assert_trained_unchanged') as frozen_check:
                audit.run(args, parent)
            frozen_check.assert_called_once_with(state, bundle)
            parent_check.assert_called_once_with(parent)
            summary = json.loads((output/'summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertTrue(summary['identity_passed'])
            self.assertTrue(summary['smoke'])
            self.assertIsNone(summary['reproduction']['passed'])
            self.assertEqual(summary['memory_head_step'], 5000)
            self.assertTrue(all(p.suffix == '.json' for p in output.iterdir()))

    def test_changing_query_gt_changes_metrics_but_not_predictions(self):
        bundle, state, data, reference, route = self.pipeline()
        observed = []
        original_update = audit.DiagnosticAccumulator.update
        def capture(accumulator, gt, predictions, building, names):
            observed.append({key: value.clone() for key, value in predictions.items()})
            return original_update(accumulator, gt, predictions, building, names)
        results = []
        with patch.object(audit.common, 'parse_batch', side_effect=lambda b, d: b), \
             patch.object(audit.common, 'capture', return_value=reference), \
             patch.object(audit.common, 'routes', return_value=(route, route)), \
             patch.object(audit.DiagnosticAccumulator, 'update', new=capture):
            for label in (0, 2):
                changed = [(pre, post, loc, torch.full_like(gt, label), names)
                           for pre, post, loc, gt, names in data]
                results.append(audit.evaluate(bundle, state, changed, SimpleNamespace(smoke=True)))
        for before, after in zip(observed[:2], observed[2:]):
            for name in before:
                torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)
        self.assertNotEqual(results[0]['final_metrics']['normal']['support'],
                            results[1]['final_metrics']['normal']['support'])

    def test_identity_failure_preserves_evidence_and_blocks_completion(self):
        bundle, state, _, _, _ = self.pipeline()
        result = dict(final_metrics={}, identity_passed=False, diagnostics={}, replay={})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new'
            args = SimpleNamespace(output=str(output), source_head='hard', pool_mode='hard', read_mode='hard',
                                   smoke=True, workers=0, dataset='synthetic')
            with patch.object(audit.common, 'load_source', return_value=bundle), \
                 patch.object(audit, 'load_trained', return_value=state), \
                 patch.object(audit, 'verify_parent_state'), \
                 patch.object(audit, 'baseline_policy', return_value=None), \
                 patch.object(audit.common, 'source_identity', return_value={}), \
                 patch.object(audit.common, 'loader', return_value=[]), \
                 patch.object(audit, 'evaluate', return_value=result), \
                 patch.object(audit, 'initial_gate') as gate:
                with self.assertRaisesRegex(RuntimeError, 'Identity replay'):
                    audit.run(args, dict(files={}))
            gate.assert_not_called()
            failed = json.loads((output/'summary.json').read_text())
            self.assertFalse(failed['complete'])
            self.assertFalse(failed['identity_passed'])
            self.assertTrue((output/'reader_evidence.json').is_file())

    def test_replay_counts_accumulate_not_max(self):
        result = audit.empty_replay()
        row = dict(identity_region_argmax_disagreements=2, max_abs_logits=.1)
        audit.merge_replay(result, row); audit.merge_replay(result, row)
        self.assertEqual(result['identity_region_argmax_disagreements'], 4)
        self.assertEqual(result['max_abs_logits'], .1)


if __name__ == '__main__': unittest.main()
