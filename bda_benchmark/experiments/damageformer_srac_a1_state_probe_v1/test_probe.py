"""Synthetic CPU-only checks; no BRIGHT files are required."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from .probe import FrozenStateChanged, StateTrace, compare_state


class StateProbeTests(unittest.TestCase):
    def test_parameter_and_buffer_keys_are_distinguished(self):
        model = nn.BatchNorm1d(3)
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            model.weight[0] += 0.25
            model.running_mean[1] += 0.125
        changes = {row["key"]: row for row in compare_state(initial, model)}
        self.assertEqual(changes["weight"]["kind"], "parameter")
        self.assertEqual(changes["running_mean"]["kind"], "buffer")
        self.assertEqual(changes["weight"]["changed_elements"], 1)
        self.assertEqual(changes["running_mean"]["changed_elements"], 1)

    def test_nan_bits_are_compared_exactly(self):
        model = nn.Module()
        model.register_buffer("stored", torch.tensor([float("nan")]))
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        self.assertEqual(compare_state(initial, model), [])

    def test_first_change_records_phase_and_stops(self):
        model = nn.BatchNorm1d(2)
        with tempfile.TemporaryDirectory() as folder:
            trace = StateTrace(Path(folder), 20)
            trace.capture(model)
            trace.poll("before_initial_val", force=True)
            with torch.no_grad():
                model.running_mean[0] += 1
            with self.assertRaises(FrozenStateChanged):
                trace.poll("after_initial_val")
            self.assertEqual(trace.first_change["phase"], "after_initial_val")
            self.assertEqual(trace.first_change["changed_state"][0]["key"], "running_mean")

    def test_version_bump_without_byte_change_is_not_mutation(self):
        model = nn.BatchNorm1d(2)
        with tempfile.TemporaryDirectory() as folder:
            trace = StateTrace(Path(folder), 20)
            trace.capture(model)
            with torch.no_grad():
                model.running_mean.add_(0)
            trace.poll("after_train_step_1")
            self.assertIsNone(trace.first_change)


if __name__ == "__main__":
    unittest.main()
