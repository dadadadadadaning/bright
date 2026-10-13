"""CPU-only GPU queue contracts; no GPU query or CUDA context is created."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import gpu_queue
import pipeline


CARD = [["0", "GPU-test-4090", "00000000:01:00.0"]]


class GPUQueueTests(unittest.TestCase):
    def test_single_card_default_maps_to_uuid(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(gpu_queue, "_query", return_value=CARD):
            self.assertEqual(gpu_queue.resolve_gpu(0), "GPU-test-4090")

    def test_single_card_numeric_visible_maps_to_uuid(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True), patch.object(gpu_queue, "_query", return_value=CARD):
            self.assertEqual(gpu_queue.resolve_gpu(0), "GPU-test-4090")

    def test_single_card_uuid_visible_maps_to_uuid(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test"}, clear=True), patch.object(gpu_queue, "_query", return_value=CARD):
            self.assertEqual(gpu_queue.resolve_gpu(0), "GPU-test-4090")

    def test_missing_or_disabled_card_fails_closed(self):
        for visible in ("", "-1", "MIG-unsupported", "2"):
            with self.subTest(visible=visible), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}, clear=True), patch.object(gpu_queue, "_query", return_value=CARD):
                with self.assertRaises(ValueError):
                    gpu_queue.resolve_gpu(0)
        with patch.dict(os.environ, {}, clear=True), patch.object(gpu_queue, "_query", return_value=CARD):
            with self.assertRaises(ValueError):
                gpu_queue.resolve_gpu(1)

    def test_multiple_cards_require_unambiguous_mapping(self):
        cards = CARD + [["1", "GPU-second", "00000000:02:00.0"]]
        with patch.dict(os.environ, {}, clear=True), patch.object(gpu_queue, "_query", return_value=cards):
            with self.assertRaises(ValueError):
                gpu_queue.resolve_gpu(0)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-second"}, clear=True), patch.object(gpu_queue, "_query", return_value=cards):
            self.assertEqual(gpu_queue.resolve_gpu(0), "GPU-second")

    def test_busy_compute_pid_is_not_idle_even_when_utilization_zero(self):
        with patch.object(gpu_queue, "_query", side_effect=[[["GPU-test-4090", "123"]], [["0", "18000"]]]):
            row = gpu_queue.observation("GPU-test-4090")
        self.assertFalse(row["idle"])
        self.assertEqual(row["compute_pids"], [123])

    def test_other_gpu_process_does_not_block_selected_gpu(self):
        with patch.object(gpu_queue, "_query", side_effect=[[["GPU-other", "123"]], [["0", "18000"]]]):
            row = gpu_queue.observation("GPU-test-4090")
        self.assertTrue(row["idle"])

    def test_busy_utilization_is_not_idle_without_compute_pid(self):
        with patch.object(gpu_queue, "_query", side_effect=[[], [["50", "18000"]]]):
            self.assertFalse(gpu_queue.observation("GPU-test-4090")["idle"])

    def test_busy_resets_idle_streak_before_start(self):
        rows = [dict(idle=value) for value in (False, True, False, True, True)]
        logs = []
        with patch.object(gpu_queue, "resolve_gpu", return_value="GPU-test-4090"), patch.object(gpu_queue, "observation", side_effect=rows) as observe, patch.object(gpu_queue.time, "sleep") as sleep:
            result = gpu_queue.wait_for_idle(0, log=logs.append, poll_seconds=1)
        self.assertEqual(observe.call_count, 5)
        self.assertEqual(sleep.call_count, 4)
        self.assertEqual(len(logs), 5)
        self.assertFalse(result["cuda_context_created"])
        self.assertFalse(result["exclusive_reservation"])

    def test_query_failure_propagates_without_retry_or_fallback(self):
        failure = subprocess.CalledProcessError(9, ["nvidia-smi"])
        with patch.object(gpu_queue, "resolve_gpu", side_effect=failure), patch.object(gpu_queue, "observation") as observe, patch.object(gpu_queue.time, "sleep") as sleep:
            with self.assertRaises(subprocess.CalledProcessError):
                gpu_queue.wait_for_idle(0)
        observe.assert_not_called()
        sleep.assert_not_called()

    def test_smi_failure_stops_pipeline_before_any_model_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "protocol.json"
            source.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(output=str(folder / "new_run"), repo_root=str(folder / "repo"),
                protocol=str(source), python=sys.executable, workers=0, amp="bf16")
            with patch.object(pipeline, "_validate_protocol", return_value=({"runtime_settings": {}}, 0)), patch.object(pipeline, "snapshot_sources", return_value={}), patch.object(gpu_queue, "wait_for_idle", side_effect=FileNotFoundError("nvidia-smi unavailable")), patch.object(pipeline, "run_child") as child:
                with self.assertRaises(FileNotFoundError):
                    pipeline.run(args)
            child.assert_not_called()
            state = json.loads((folder / "new_run/workflow_state.json").read_text())
            self.assertFalse(state["complete"])
            self.assertFalse(state["full_validation_evidence"])
            self.assertEqual(state["stages"], [])
            self.assertIn("nvidia-smi unavailable", state["failure"])

    def test_import_and_mocked_wait_do_not_import_torch_or_initialize_cuda(self):
        # A fresh interpreter avoids already-imported torch from unrelated tests.
        code = """
import builtins, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def forbid_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('queue attempted torch import / CUDA initialization')
    return original(name, *args, **kwargs)
builtins.__import__ = forbid_torch
import gpu_queue
with patch.object(gpu_queue, 'resolve_gpu', return_value='GPU-fake'), patch.object(gpu_queue, 'observation', return_value={'idle': True}), patch.object(gpu_queue.time, 'sleep'):
    result = gpu_queue.wait_for_idle(0, log=lambda row: None, poll_seconds=1)
assert result['cuda_context_created'] is False
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-I", "-c", code, str(Path(__file__).resolve().parent)], check=True,
                       capture_output=True, text=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
