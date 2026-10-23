import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from graph_observer import (BackwardIndexObserver, ObservationError, compact_view,
                            index_statistics, normalize_dim)


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def observer(self, **kwargs):
        observer = BackwardIndexObserver(self.output, 158, **kwargs)
        self.addCleanup(observer.close)
        return observer

    def test_real_gather_sort_gradients_unchanged(self):
        base = torch.tensor([[0.2, 0.7, -0.3, 0.8], [0.4, -0.1, 0.9, 0.3]])
        indices = torch.tensor([[0, 3, 1], [2, 0, 1]])
        def calculate(observed):
            value = base.clone().requires_grad_()
            with torch.autograd.detect_anomaly():
                gathered = value.gather(-1, indices)
                sorted_value = value.sort(-1).values
                loss = gathered.square().sum() + (sorted_value * torch.arange(4)).sum()
                if observed:
                    observer.attach(loss)
                loss.backward()
            return value.grad
        expected = calculate(False)
        observer = self.observer()
        actual = calculate(True)
        self.assertTrue(torch.equal(expected, actual))
        summary = observer.finish()
        self.assertEqual(summary["observed_index_nodes"], 2)
        self.assertEqual(summary["prehook_calls"], 2)
        self.assertTrue(all(n["forward_traceback"] for n in summary["nodes"]))
        self.assertTrue(all(n["dim"] == 1 for n in summary["nodes"]))

    def test_expanded_index_compaction_and_dedup(self):
        value = torch.randn(5, 4, requires_grad=True)
        index = torch.tensor([[0, 2]]).expand(5, 2)
        observer = self.observer(max_bytes=16)
        observer.watch("index", index)
        observer.attach(value.gather(1, index).sum())
        self.assertEqual(compact_view(index).shape, (2,))
        self.assertEqual(observer.snapshot_bytes, 16)
        self.assertEqual(len(observer._snapshots), 1)
        self.assertEqual(observer.nodes[0]["compact_shape"], [2])

    def test_out_of_bounds_data_mutation_caught_before_gather_backward(self):
        value = torch.randn(2, 4, requires_grad=True)
        index = torch.tensor([[0, 1], [2, 3]])
        loss = value.gather(1, index).sum()
        observer = self.observer()
        observer.attach(loss)
        index.data[0, 0] = 2 ** 51 + 25
        with self.assertRaisesRegex(ObservationError, "out of bounds"):
            loss.backward()
        self.assertIsNone(value.grad)
        summary = json.loads(observer._path("failure").read_text())
        self.assertFalse(summary["latest_node"]["current_statistics"]["in_bounds"])
        artifact = torch.load(observer._path("evidence", "pt"), weights_only=True)
        self.assertEqual(artifact["changed_elements"], 1)
        self.assertEqual(artifact["changed_after"].item(), 2 ** 51 + 25)

    def test_changed_valid_sort_saved_index_detected(self):
        value = torch.tensor([3., 1., 2.], requires_grad=True)
        sorted_value = value.sort().values
        observer = self.observer()
        observer.attach(sorted_value.sum())
        sorted_value.grad_fn._saved_indices.data[0] = 0
        with self.assertRaises(ObservationError):
            sorted_value.sum().backward()
        self.assertTrue(observer.latest_node["current_statistics"]["in_bounds"])
        self.assertTrue(observer.latest_node["values_changed"])

    def test_native_version_guard_preserved(self):
        value = torch.randn(2, 4, requires_grad=True)
        index = torch.tensor([[0, 1], [2, 3]])
        loss = value.gather(1, index).sum()
        observer = self.observer()
        observer.attach(loss)
        index[0, 0] = 2  # normal write must retain PyTorch's native version error
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            loss.backward()
        self.assertEqual(observer.prehook_calls, 0)
        self.assertEqual(observer.latest_node["stage"], "prehook_saved_tensor_access")

    def test_preexisting_bad_sort_index_attach_fails(self):
        value = torch.randn(4, requires_grad=True)
        sorted_value = value.sort().values
        sorted_value.grad_fn._saved_indices.data[0] = -1
        observer = self.observer()
        with self.assertRaises(ObservationError):
            observer.attach(sorted_value.sum())
        self.assertFalse(observer.nodes[0]["statistics"]["in_bounds"])

    def test_budget_failure_is_explicit(self):
        value = torch.randn(4, requires_grad=True)
        observer = self.observer(max_bytes=31)
        with self.assertRaisesRegex(ObservationError, "coverage incomplete"):
            observer.attach(value.sort().values.sum())
        self.assertFalse(observer.coverage_complete)
        self.assertEqual(observer.snapshot_bytes, 0)

    def test_shared_graph_node_registered_once(self):
        value = torch.randn(4, requires_grad=True)
        y = value.gather(0, torch.tensor([0, 3]))
        loss = y.sum() + y.square().sum()
        observer = self.observer()
        observer.attach(loss)
        loss.backward()
        self.assertEqual(len(observer.nodes), 1)
        self.assertEqual(observer.prehook_calls, 1)

    def test_watches_detect_change_and_close_removes_handles(self):
        index = torch.tensor([0, 1])
        value = torch.randn(4, requires_grad=True)
        observer = self.observer()
        observer.watch("parent.ids", index)
        loss = value.gather(0, index).sum()
        observer.attach(loss)
        observer.check_watched("before_backward")
        index.data[0] = 2
        with self.assertRaises(ObservationError):
            observer.check_watched("after_forward")
        observer.close()
        self.assertFalse(observer._handles)
        self.assertFalse(observer._snapshots)
        self.assertFalse(observer._node_refs)

    def test_record_failure_is_cpu_only_and_disk_bounded(self):
        observer = self.observer()
        observer.evidence_limit_bytes = 70 * 1024
        index = torch.arange(20000)
        observer.watch("parent.ids", index)
        observer.check_watched("ready")
        # Failure recording must use existing CPU snapshots, not live tensor reads.
        with mock.patch("graph_observer.compact_view", side_effect=AssertionError("GPU read")):
            summary = observer.record_failure(RuntimeError("CUDA device assert"))
        artifact = Path(summary["evidence"]["path"])
        self.assertLessEqual(artifact.stat().st_size, observer.evidence_limit_bytes)
        self.assertTrue(summary["failure_recording_cpu_only"])
        self.assertTrue(summary["evidence"]["truncated"])

    def test_unsigned_negative_dim_and_bounds(self):
        self.assertEqual(normalize_dim(2**64 - 1, 3), 2)
        self.assertEqual(normalize_dim(-2, 3), 1)
        self.assertFalse(index_statistics(torch.tensor([-1, 0]), 4)["in_bounds"])
        self.assertTrue(index_statistics(torch.tensor([], dtype=torch.int64), 0)["in_bounds"])
        with self.assertRaises(ObservationError):
            normalize_dim(4, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
