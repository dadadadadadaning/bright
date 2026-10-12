"""Tiny label cases verify denominators and dynamic paired error evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from .evidence import B_LABELS, Evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_dynamic_pairs_exact_harm_rescue_and_ignore(self):
        gt = torch.tensor([[[0, 1, 2, 3, 255, -1]]])
        baseline = torch.tensor([[[0, 2, 2, 0, 99, 99]]])
        candidate = torch.tensor([[[1, 1, 1, 3, 99, 99]]])
        ev = Evidence(("p", "q_read"), (("p", "q_read"),))
        ev.update(gt, dict(p=baseline, q_read=candidate), torch.tensor([[[[0., .2, .5, 1., .9, .9]]]]), "event_a_01")
        result = ev.finalize()
        pair = result["paired"]["p__q_read"]
        self.assertEqual((pair["count"], pair["harmed"], pair["rescued"], pair["net"]), (4, 2, 2, 0))
        self.assertEqual((pair["reference_correct"], pair["reference_wrong"]), (2, 2))
        self.assertEqual(pair["harm_rate"], 1.)
        self.assertEqual(pair["rescue_rate"], 1.)
        self.assertEqual(pair["damaged_to_intact_harm"], 1)
        self.assertEqual(pair["harm_gt_to_wrong"][0][1], 1)
        self.assertEqual(pair["rescue_gt_from_wrong"][3][0], 1)
        self.assertEqual(result["metrics"]["p"]["support"], [1, 1, 1, 1])
        self.assertEqual(result["metrics"]["p"]["background_to_building"], 0)
        self.assertEqual(result["metrics"]["q_read"]["background_to_building"], 1)
        event = result["per_event"]["event_a"]
        self.assertEqual(event["paired"], result["paired"])
        self.assertEqual(result["per_image"][0]["confusion"]["p"], result["metrics"]["p"]["confusion"])
        json.dumps(result, allow_nan=False)

    def test_bin_endpoints_including_high_confidence(self):
        gt = torch.tensor([[[0, 0, 1, 1, 2, 2, 3]]])
        b = torch.tensor([[[[0., .1, .4, .6, .899, .9, 1.]]]])
        ev = Evidence(("region", "edge_mean"), (("region", "edge_mean"),))
        ev.update(gt, dict(region=gt.clone(), edge_mean=gt.clone()), b, "flood_01")
        groups = ev.finalize()["groups"]
        self.assertEqual([groups[key]["metrics"]["region"]["count"] for key in B_LABELS], [1, 1, 1, 2, 2])
        self.assertEqual(groups["roi_true_negative"]["metrics"]["region"]["count"], 2)
        self.assertEqual(groups["roi_false_negative"]["metrics"]["region"]["count"], 1)
        self.assertEqual(groups["roi_true_positive"]["metrics"]["region"]["count"], 4)
        self.assertEqual(groups["roi_false_positive"]["metrics"]["region"]["count"], 0)

    def test_class_boundary_sides_and_interior_partition(self):
        gt = torch.zeros((1, 11, 11), dtype=torch.long)
        gt[:, :, 5:] = 2
        ev = Evidence(("p", "q_read", "content_read"), (("p", "q_read"), ("p", "content_read"), ("q_read", "content_read")))
        pred = {mode: gt.clone() for mode in ev.modes}
        ev.update(gt, pred, torch.full((1, 1, 2, 2), .95), "quake_0")
        result = ev.finalize()
        groups = result["groups"]
        self.assertEqual(groups["boundary"]["metrics"]["p"]["count"], 66)
        self.assertEqual(groups["interior"]["metrics"]["p"]["count"], 55)
        self.assertEqual(groups["Background__boundary"]["metrics"]["p"]["count"], 33)
        self.assertEqual(groups["Damaged__boundary"]["metrics"]["p"]["count"], 33)
        self.assertEqual(groups["proxy_small"]["metrics"]["p"]["count"], 66)
        self.assertEqual(result["component_counts"]["size_small"], 1)
        for name in result["paired"]:
            self.assertEqual(result["paired"][name]["count"], 121)

    def test_all_ignored_is_json_safe_and_does_not_count_as_background(self):
        gt = torch.full((1, 3, 3), 255)
        ev = Evidence(("p", "q"), (("p", "q"),))
        ev.update(gt, dict(p=gt.clone(), q=gt.clone()), torch.ones(1, 1, 1, 1), "ignored_0")
        result = ev.finalize()
        self.assertEqual(result["metrics"]["p"]["count"], 0)
        self.assertIsNone(result["metrics"]["p"]["valid_miou"])
        self.assertIsNone(result["paired"]["p__q"]["harm_rate"])
        self.assertEqual(result["groups"]["B_09_1"]["metrics"]["p"]["count"], 0)
        json.dumps(result, allow_nan=False)

    def test_multiple_images_sum_and_event_pairs(self):
        ev = Evidence(("p", "q"), (("p", "q"),))
        gt = torch.tensor([[[1, 2]]])
        for index in range(2):
            ev.update(gt, dict(p=gt, q=torch.tensor([[[0, 2]]])), torch.ones(1, 1, 1, 2), "event_" + str(index))
        result = ev.finalize()
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(result["paired"]["p__q"]["harmed"], 2)
        self.assertEqual(result["per_event"]["event"]["paired"]["p__q"]["harmed"], 2)
        self.assertTrue(np.array_equal(np.sum([row["confusion"]["p"] for row in result["per_image"]], axis=0), result["metrics"]["p"]["confusion"]))

    def test_invalid_contracts_fail_before_counts(self):
        for modes, pairs in (((), ()), (("p", "p"), ()), (("p",), (("p", "q"),)), (("p",), (("p", "p"),)), (("p", "q"), (("p", "q"), ("p", "q")))):
            with self.assertRaises(ValueError):
                Evidence(modes, pairs)
        ev = Evidence(("p",), ())
        gt = torch.zeros((1, 2, 2), dtype=torch.long)
        with self.assertRaises(ValueError):
            ev.update(gt, dict(p=gt + 4), torch.zeros(1, 1, 1, 1), "x")
        with self.assertRaises(ValueError):
            ev.update(gt, dict(p=gt), torch.full((1, 1, 1, 1), float("nan")), "x")
        with self.assertRaises(ValueError):
            ev.update(gt, dict(q=gt), torch.zeros(1, 1, 1, 1), "x")
        self.assertEqual(ev.finalize()["image_count"], 0)


class BoundaryAuditTests(unittest.TestCase):
    @staticmethod
    def synthetic_front():
        from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import (
            StructureTokenAdapter, SuperpixelEmbedding, project_region_probabilities,
            soft_assignment_boundary,
        )
        torch.set_num_threads(2)
        torch.manual_seed(91)
        model = torch.nn.Module()
        model.adapter = StructureTokenAdapter(4, 4)
        model.embedding = SuperpixelEmbedding(4, 4, 4)
        model.region_classifier = torch.nn.Linear(4, 4)
        with torch.no_grad():
            model.adapter.layer_scale.fill_(2.)
        model.requires_grad_(False).eval()
        q = torch.rand(1, 9, 6).softmax(1)
        ids = torch.tensor([[index % 2] * 6 for index in range(9)])
        shape = (2, 3)
        auxiliary = dict(fused_roi=torch.randn(1, 4, *shape), p_building=torch.rand(1, 1, 4, 6))
        edge = soft_assignment_boundary(q, ids, shape)
        entropy = torch.rand(1, 1, *shape)
        adapted, _ = model.adapter(auxiliary["fused_roi"], auxiliary["p_building"], edge, entropy)
        tokens, _, _, _ = model.embedding(adapted, auxiliary["fused_roi"], q, ids, shape, 2)
        probability = project_region_probabilities(model.region_classifier(tokens), q, ids, shape, (4, 6))
        native = dict(q=q, ids=ids, shape=shape, count=2, prob=probability, soft_boundary=edge,
                      entropy_map=entropy, old_pixel=torch.randn(1, 4, 4, 6))
        return model, native, auxiliary

    def test_recomputes_E_path_with_fixed_Q_and_native_probability(self):
        from .boundary_audit import boundary_interventions
        from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.support import module_fingerprint
        model, native, auxiliary = self.synthetic_front()
        fingerprint = module_fingerprint(model)
        probabilities, building, maps, checks = boundary_interventions(model, native, auxiliary, (4, 6))
        self.assertTrue(checks["old_probability_reproduced"])
        self.assertTrue(all(checks["fixed_tensors_unchanged"].values()))
        self.assertEqual(fingerprint, module_fingerprint(model))
        self.assertIs(probabilities["region"], native["prob"])
        self.assertEqual(tuple(building.shape), (1, 1, 2, 3))
        self.assertGreater(float((probabilities["edge_zero"] - probabilities["region"]).abs().max()), 1e-7)
        self.assertGreater(float((maps["old"] - maps["change"]).abs().max()), .1)

    def test_distributions_use_GT_area_fractions_and_preserve_ignored_area(self):
        from .boundary_audit import BoundaryDistributions
        stats = BoundaryDistributions()
        gt = torch.tensor([[[0, 1], [2, 255]]])
        maps = {key: torch.full((1, 1, 1, 1), value) for key, value in dict(old=.8, mean=.8, zero=0., change=.2).items()}
        _, area_maps = stats.update(maps, gt, "tiny_01")
        self.assertTrue(torch.equal(area_maps["GT_class_area_fractions"], torch.tensor([[[[.25]], [[.25]], [[.25]], [[0.]]]])))
        result = stats.finalize()
        self.assertEqual(result["feature_grid_distributions"]["old"]["Background"]["area_weight_feature_pixels"], .25)
        self.assertIsNone(result["feature_grid_distributions"]["change"]["Destroyed"]["mean"])
        self.assertAlmostEqual(result["feature_grid_distributions"]["old"]["boundary"]["area_weight_feature_pixels"], .75)
        json.dumps(result, allow_nan=False)

    def test_smoke_run_writes_reviewable_evidence_and_refuses_overwrite(self):
        self._run_synthetic(smoke=True, fail_gate=False)

    def test_reproduction_failure_preserves_evidence(self):
        self._run_synthetic(smoke=False, fail_gate=True)

    def test_subset_selection_cannot_masquerade_as_full_or_resume(self):
        from .boundary_audit import selected_validation
        names = [str(i) for i in range(450)]
        args = SimpleNamespace(smoke=False, start_index=330, max_images=10)
        chosen, scope = selected_validation(args, names)
        self.assertEqual(chosen, names[330:340])
        self.assertFalse(scope['full_validation'])
        self.assertFalse(scope['resumed'])
        self.assertTrue(scope['diagnostic_subset'])
        args.start_index, args.max_images = 0, 450
        self.assertFalse(selected_validation(args, names)[1]['full_validation'])
        args.max_images = 0
        self.assertTrue(selected_validation(args, names)[1]['full_validation'])
        args.smoke, args.start_index = True, 330
        with self.assertRaises(ValueError):
            selected_validation(args, names)

    def test_invalid_projection_records_failure_and_does_not_hide_it(self):
        from . import boundary_audit as audit
        from .probability_check import ProbabilityContractError
        model, native, auxiliary = self.synthetic_front()
        gt = torch.arange(24).reshape(1, 4, 6) % 4
        pre = torch.zeros(1, 3, 4, 6)
        parsed = (pre, pre, torch.zeros_like(gt), gt, ['synthetic_331'])
        original = audit.project_region_probabilities
        calls = []

        def corrupt_only_zero(*args):
            calls.append(1)
            output = original(*args)
            return output * .9 if len(calls) == 2 else output

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'probe'
            args = SimpleNamespace(workers=0, threshold=.5, output=str(output), train_list='train_set.txt',
                val_list='val_set.txt', dataset='synthetic', seed=4, smoke=False, device='cpu',
                start_index=330, max_images=10)
            with patch.object(audit, 'load_fixed', return_value=(model, dict(step=7), {})), \
                 patch.object(audit, 'lists', return_value=([], [str(i) for i in range(450)])), \
                 patch.object(audit, 'loader', return_value=[object()]) as loader_mock, \
                 patch.object(audit, 'parse_batch', return_value=parsed), \
                 patch.object(audit, 'capture', return_value=(native, auxiliary)), \
                 patch.object(audit, 'runtime_diagnostics', return_value={'torch_version': 'test'}), \
                 patch.object(audit, 'project_region_probabilities', side_effect=corrupt_only_zero):
                with self.assertRaises(ProbabilityContractError):
                    audit.run(args)
            self.assertEqual(loader_mock.call_args.args[1], [str(i) for i in range(330, 340)])
            failed = json.loads((output / 'failure.json').read_text())
            self.assertEqual(failed['val_position_one_based'], 331)
            self.assertEqual(failed['image'], 'synthetic_331')
            self.assertFalse(failed['complete'])
            self.assertFalse(failed['audit_scope']['full_validation'])
            self.assertIn('projection_replays', failed['probability_diagnostics'])
            partial = json.loads((output / 'partial_audit.json').read_text())
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['image_count'], 0)
            self.assertFalse((output / 'audit.json').exists())

    def test_successful_late_probe_does_not_apply_full_val_gate(self):
        from . import boundary_audit as audit
        model, native, auxiliary = self.synthetic_front()
        gt = torch.arange(24).reshape(1, 4, 6) % 4
        pre = torch.zeros(1, 3, 4, 6)
        parsed = (pre, pre, torch.zeros_like(gt), gt, ['synthetic_331'])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'probe'
            args = SimpleNamespace(workers=0, threshold=.5, output=str(output), train_list='train_set.txt',
                val_list='val_set.txt', dataset='synthetic', seed=4, smoke=False, device='cpu',
                start_index=330, max_images=1)
            with patch.object(audit, 'load_fixed', return_value=(model, dict(step=7), {})), \
                 patch.object(audit, 'lists', return_value=([], [str(i) for i in range(450)])), \
                 patch.object(audit, 'loader', return_value=[object()]), \
                 patch.object(audit, 'parse_batch', return_value=parsed), \
                 patch.object(audit, 'capture', return_value=(native, auxiliary)), \
                 patch.object(audit, 'reproduction_gate', side_effect=AssertionError('must not call full gate')):
                result = audit.run(args)
            self.assertTrue(result['complete'])
            self.assertFalse(result['audit_scope']['full_validation'])
            self.assertIsNone(result['reproduction']['passed'])
            self.assertEqual(result['representative_map_files'], [])

    def _run_synthetic(self, smoke, fail_gate):
        from . import boundary_audit as audit
        model, native, auxiliary = self.synthetic_front()
        gt = torch.arange(24).reshape(1, 4, 6) % 4
        pre = torch.zeros(1, 3, 4, 6)
        parsed = (pre, pre, torch.zeros_like(gt), gt, ["synthetic_01"])
        payload = dict(step=7, run_complete=False, load_report=dict(expected_valid_miou=0.), valid_miou=0.)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            args = SimpleNamespace(workers=0, threshold=.5, output=str(output), train_list="train_set.txt", val_list="val_set.txt", dataset="synthetic", seed=4, smoke=smoke, device="cpu")
            # Only I/O boundaries are replaced; E prediction and statistics run normally.
            with patch.object(audit, "load_fixed", return_value=(model, payload, {})), patch.object(audit, "lists", return_value=([], ["synthetic_01"])), patch.object(audit, "loader", return_value=[object()]), patch.object(audit, "parse_batch", return_value=parsed), patch.object(audit, "capture", return_value=(native, auxiliary)):
                if fail_gate:
                    with self.assertRaisesRegex(RuntimeError, "Evidence saved"):
                        audit.run(args)
                else:
                    result = audit.run(args)
                    self.assertTrue(result["complete"])
                with self.assertRaisesRegex(ValueError, "Preserve existing"):
                    audit.run(args)
            saved = json.loads((output / "audit.json").read_text())
            self.assertEqual(saved["complete"], not fail_gate)
            self.assertTrue(saved["weights_and_buffers_unchanged"])
            self.assertFalse(saved["test_accessed"])
            self.assertTrue((output / "feature_maps_00.pt").is_file())
            self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
