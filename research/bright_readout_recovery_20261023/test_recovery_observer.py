import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from recovery_observer import (ObservationError, UpdateObserver, compact_view,
                               observed_update)


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def test_expanded_indices_snapshot_has_no_channel_expansion(self):
        indices = torch.arange(18).reshape(1, 2, 9)
        expanded = indices[:, 0, :, None].expand(-1, -1, 128)
        observer = UpdateObserver(self.output, 801)
        packed = observer.pack(expanded)
        self.assertEqual(tuple(compact_view(expanded).shape), (1, 9))
        self.assertEqual(observer.bytes, 9 * 8)
        self.assertTrue(torch.equal(observer.unpack(packed), expanded))

    def test_repeated_saved_view_is_deduplicated(self):
        observer = UpdateObserver(self.output, 801)
        tensor = torch.arange(16)
        observer.pack(tensor)
        observer.pack(tensor)
        self.assertEqual(observer.bytes, 16 * 8)
        self.assertEqual(observer.saved_integer_unique, 1)

    def test_saved_indices_detect_untracked_data_mutation(self):
        observer = UpdateObserver(self.output, 801)
        tensor = torch.arange(16)
        packed = observer.pack(tensor)
        version = tensor._version
        tensor.data[2] = 2**51 + 2
        self.assertEqual(tensor._version, version)
        with self.assertRaisesRegex(ObservationError, "autograd_unpack"):
            observer.unpack(packed)
        observer.record_failure(ObservationError("index changed"))
        evidence = torch.load(self.output / "diagnostic_step_000801/cpu_evidence.pt", weights_only=True)
        self.assertEqual(evidence["before"][2], 2)
        self.assertEqual(evidence["after"][2], 2**51 + 2)

    def test_saved_float_keeps_version_guard(self):
        observer = UpdateObserver(self.output, 801)
        tensor = torch.ones(3)
        packed = observer.pack(tensor)
        tensor.add_(1)
        with self.assertRaisesRegex(ObservationError, "version"):
            observer.unpack(packed)

    def test_canonical_route_changes_fail_before_backward(self):
        observer = UpdateObserver(self.output, 801)
        q = torch.tensor([0.25, 0.75])
        observer.watch("q", q)
        q.data[0] = 0.125
        with self.assertRaisesRegex(ObservationError, "before_backward"):
            observer.check_watched("before_backward")

    def test_budget_exhaustion_is_explicit(self):
        observer = UpdateObserver(self.output, 801, max_bytes=8)
        with self.assertRaisesRegex(ObservationError, "coverage is incomplete"):
            observer.pack(torch.arange(2))
        self.assertEqual(observer.bytes, 0)
        self.assertEqual(observer.failure["kind"], "observation_budget_exceeded")

    def test_failure_record_never_reads_live_tensors(self):
        observer = UpdateObserver(self.output, 801)
        observer.watch("ids", torch.arange(4))
        with patch("recovery_observer._cpu", side_effect=AssertionError("GPU read after assert")):
            observer.record_failure(RuntimeError("CUDA device-side assert"))
        data = json.loads((self.output / "diagnostic_step_000801/failure.json").read_text())
        self.assertNotIn("evidence_write_error", data)
        self.assertEqual(data["cpu_snapshot_keys"], ["ids"])

    def test_hooks_preserve_gather_and_sort_gradient(self):
        x = torch.tensor([2.0, -1.0, 4.0], requires_grad=True)
        original = x.detach().clone().requires_grad_()
        ids = torch.tensor([0, 2, 1])
        expected = original.gather(0, ids).sort().values.square().sum()
        expected.backward()
        observer = UpdateObserver(self.output, 801)
        with observer.saved_context():
            actual = x.gather(0, ids).sort().values.square().sum()
            actual.backward()
        self.assertTrue(torch.equal(x.grad, original.grad))
        self.assertGreater(observer.unpacked_integer_calls, 0)

    def test_nondetailed_delegates_original_update(self):
        seen = []
        def update(*args, **kwargs):
            seen.append((args, kwargs))
            return {"sentinel": True}
        experiment = SimpleNamespace(update=update)
        result = observed_update(experiment, 1, 2, 3, 4, 5, 6, 7,
                                 step=805, read_weight=0.01, multiplier=0.2,
                                 output=self.output, detailed=False)
        self.assertEqual(result, {"sentinel": True})
        self.assertEqual(seen[0][0], (1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(seen[0][1], dict(step=805, read_weight=.01, multiplier=.2))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_projection = torch.nn.Linear(4, 4)

    def forward(self, *, q, ids, route_q, route_ids, support_ids,
                pixel_features=None, region_f=None):
        features = q.gather(1, ids)
        logits = self.query_projection(features).reshape(1, 4, 1, 1)
        probability = logits.softmax(1)
        return {"probabilities": probability,
                "read_probability": probability.reshape(1, 1, 1, 4).expand(1, 1, 2, 4),
                "read_valid": torch.ones(1, 1, 2, dtype=torch.bool)}


class RecipeTests(unittest.TestCase):
    def test_instrumented_update_matches_published_recipe(self):
        self.check_recipe(0)

    def test_instrumented_auxiliary_update_matches_published_recipe(self):
        self.check_recipe(.01)

    def check_recipe(self, read_weight):
        source = Path("/workspace/bright_next_step_publish_remote_20261021/research/bright_memory_readout_design_20261022/experiment.py")
        if not source.is_file():
            self.skipTest("Published experiment source unavailable; integration fixture is local")
        spec = importlib.util.spec_from_file_location("published_experiment", source)
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)
        sys.path.insert(0, str(source.parent))
        self.addCleanup(lambda: sys.path.remove(str(source.parent)))
        torch.manual_seed(3407)
        model = ToyModel()
        reference = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.005)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.0001, weight_decay=.005)
        example = SimpleNamespace(common={"q": torch.tensor([[.1, .2, .3, .4]]),
                                          "ids": torch.tensor([[0, 1, 2, 3]]),
                                          "pixel_features": torch.zeros(1, 1, 1, 1),
                                          "region_f": torch.zeros(1, 1, 1)},
                                  target=torch.zeros(1, 1, 1, dtype=torch.long))
        support = {"route_q": torch.ones(1), "route_ids": torch.zeros(1, dtype=torch.long),
                   "support_ids": torch.zeros(1, dtype=torch.long)}
        factory = SimpleNamespace(prepare=lambda example, arms: {"d": support})
        runtime = SimpleNamespace(loss=lambda p, target: -p[:, 0].log().mean())
        state = SimpleNamespace(lr_at_step=lambda *a, **kw: .0001)
        slots = SimpleNamespace(slot_labels=lambda *a, **kw: (
            torch.tensor([[[[1., 0, 0, 0], [0, 1., 0, 0]]]]), torch.ones(1, 1, 2)))
        with tempfile.TemporaryDirectory() as directory:
            expected = experiment.update(reference, reference_optimizer, example, runtime, factory, state, slots,
                                         step=1, read_weight=read_weight, multiplier=.2)
            actual = observed_update(experiment, model, optimizer, example, runtime, factory, state, slots,
                                     step=1, read_weight=read_weight, multiplier=.2, output=directory, detailed=True)
        self.assertEqual(actual, expected)
        for a, b in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(optimizer.state.values(), reference_optimizer.state.values()):
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]))


if __name__ == "__main__":
    unittest.main()
