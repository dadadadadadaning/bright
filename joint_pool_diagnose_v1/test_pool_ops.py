"""Synthetic CPU tests only; no BRIGHT data or CUDA claims."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import pool_ops as p
import replay_worker as worker


def fixture():
    torch.manual_seed(31)
    content = torch.randn(2, 5, 3, 4)
    weights = torch.rand(2, 9, 12) * 0.03  # deliberately not normalized
    ids = torch.randint(0, 7, (9, 12))
    ids[1] = ids[0]  # duplicate routes must contribute independently
    return {"format": "native_pool_input_v1", "content": content,
            "weights": weights, "ids": ids, "token_count": 9,
            "layout": {k: p.tensor_layout(t) for k, t in
                       (("content", content), ("weights", weights), ("ids", ids))}}


def slow_reference(s):
    content, weights, ids = (s[k] for k in ("content", "weights", "ids"))
    b, c, h, w = content.shape
    num = content.new_zeros((b, s["token_count"], c))
    mass = content.new_zeros((b, s["token_count"]))
    for route in range(9):
        for pixel in range(h*w):
            token = ids[route, pixel]
            ww = weights[:, route, pixel]
            num[:, token] += content.flatten(2)[:, :, pixel] * ww[:, None]
            mass[:, token] += ww
    tokens = num / mass.clamp_min(1e-6)[..., None]
    reconstruction = content.new_zeros(content.shape)
    for route in range(9):
        for pixel in range(h*w):
            reconstruction.flatten(2)[:, :, pixel] += tokens[:, ids[route, pixel]] * weights[:, route, pixel, None]
    return {"tokens": tokens, "mass": mass, "reconstruction": reconstruction}


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(min(cls.original_threads, 4))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_matches_scalar_loop_with_duplicate_routes(self):
        s = fixture()
        result = p.cpu_reference(s)
        self.assertTrue(p.compare_results(result, slow_reference(s))["passed"])

    def test_contiguous_candidate_exact_cpu(self):
        s = fixture()
        args = [s[k] for k in ("content", "weights", "ids", "token_count")]
        a, b = p.native_original_algorithm(*args), p.contiguous_index_algorithm(*args)
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in p.OUTPUT_KEYS))

    def test_empty_tokens_and_tiny_mass_keep_clamp(self):
        s = fixture()
        s["weights"].fill_(1e-10)
        out = p.cpu_reference(s)
        self.assertTrue(torch.equal(out["tokens"][:, 7:], torch.zeros_like(out["tokens"][:, 7:])))
        self.assertTrue(p.compare_results(out, slow_reference(s))["passed"])

    def test_attention_weights_are_not_renormalized(self):
        s = fixture()
        s["weights"].fill_(0.02)
        s["content"].fill_(1)
        out = p.cpu_reference(s)
        self.assertTrue(torch.allclose(out["reconstruction"], torch.full_like(s["content"], 0.18)))
        self.assertTrue(p.validate_cpu_snapshot(s)["passed"])

    def test_zero_weights(self):
        s = fixture()
        s["weights"].zero_()
        out = p.cpu_reference(s)
        self.assertTrue(all(torch.count_nonzero(v) == 0 for v in out.values()))

    def test_noncontiguous_inputs(self):
        s = fixture()
        s["content"] = s["content"].transpose(2, 3)
        s["weights"] = s["weights"].transpose(1, 2).contiguous().transpose(1, 2)
        s["ids"] = s["ids"].t().contiguous().t()
        a = p.cpu_reference(s)
        b = p.contiguous_index_algorithm(*(s[k] for k in ("content", "weights", "ids", "token_count")))
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in p.OUTPUT_KEYS))

    def test_reject_bad_ids(self):
        for value in (-1, 9):
            s = fixture()
            s["ids"][0, 0] = value
            with self.assertRaisesRegex(ValueError, "Out-of-range"):
                p.validate_cpu_snapshot(s)

    def test_reject_fractional_id_dtype(self):
        s = fixture()
        s["ids"] = s["ids"].float()
        with self.assertRaises(TypeError):
            p.validate_cpu_snapshot(s)

    def test_reject_nonfinite_and_negative_weights(self):
        for key, value in (("content", float("nan")), ("weights", float("inf")), ("weights", -1)):
            s = fixture()
            s[key].flatten()[0] = value
            with self.assertRaises(ValueError):
                p.validate_cpu_snapshot(s)

    def test_reject_wrong_shape_and_count(self):
        s = fixture()
        s["token_count"] = True
        with self.assertRaises(ValueError):
            p.validate_cpu_snapshot(s)
        s = fixture()
        s["weights"] = s["weights"][:, :8]
        with self.assertRaises(ValueError):
            p.validate_cpu_snapshot(s)

    def test_restore_holes_offset_and_transpose(self):
        original = torch.arange(120, dtype=torch.float32).reshape(10, 12)[1:7:2, 2:10:2].t()
        restored = p.restore_layout(original.clone(), p.tensor_layout(original), torch.device("cpu"))
        self.assertTrue(torch.equal(restored, original))
        self.assertEqual(restored.stride(), original.stride())
        self.assertEqual(restored.storage_offset(), original.storage_offset())

    def test_restore_consistent_zero_stride(self):
        original = torch.arange(6).reshape(1, 6).expand(9, 6)
        restored = p.restore_layout(original.clone(), p.tensor_layout(original), torch.device("cpu"))
        self.assertEqual(restored.stride(), original.stride())
        self.assertTrue(torch.equal(restored, original))

    def test_reject_inconsistent_broadcast_values(self):
        original = torch.arange(6).reshape(1, 6).expand(9, 6)
        stored = original.clone()
        stored[1, 1] = 99
        with self.assertRaisesRegex(ValueError, "Broadcast"):
            p.restore_layout(stored, p.tensor_layout(original), torch.device("cpu"))

    def test_reject_negative_and_ambiguous_layout(self):
        tensor = torch.arange(6).reshape(2, 3)
        for strides in ((1, 1), (-3, 1)):
            layout = p.tensor_layout(tensor)
            layout["stride"] = strides
            with self.assertRaises(ValueError):
                p.restore_layout(tensor, layout, torch.device("cpu"))

    def test_comparison_fixed_tolerance_and_nonfinite(self):
        a = p.cpu_reference(fixture())
        b = {k: v.clone() for k, v in a.items()}
        b["tokens"][0, 0, 0] += .1
        self.assertFalse(p.compare_results(b, a)["passed"])
        b["tokens"][0, 0, 0] = float("nan")
        report = p.compare_results(b, a)
        self.assertFalse(report["passed"])
        self.assertIsNone(report["tokens"]["max_abs"])

    def test_cpu_thread_scope_restores(self):
        original = torch.get_num_threads()
        with p.limited_cpu_threads():
            self.assertLessEqual(torch.get_num_threads(), 4)
        self.assertEqual(torch.get_num_threads(), original)

    def test_runtime_version_mismatch_rejected_before_apply(self):
        runtime = worker.current_runtime()
        runtime["torch_version"] = "wrong"
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            worker.apply_runtime(runtime)

    def test_cpu_worker_and_existing_output_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.pth"
            torch.save(fixture(), source)
            args = argparse.Namespace(input=str(source), output=str(root / "out"), backend="cpu",
                                      repeats=2, device="cuda:0", runtime_json=None)
            result = worker.run(args)
            self.assertTrue(result["complete"])
            self.assertFalse(result["full_validation_evidence"])
            self.assertEqual(len(result["repeats"]), 2)
            self.assertTrue(json.loads((root / "out/manifest.json").read_text())["source_layout_recorded"])
            with self.assertRaises(FileExistsError):
                worker.run(args)

    def test_gpu_worker_without_runtime_preserves_cpu_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.pth"
            torch.save(fixture(), source)
            args = argparse.Namespace(input=str(source), output=str(root / "out"), backend="native",
                                      repeats=1, device="cuda:0", runtime_json=None)
            with self.assertRaisesRegex(ValueError, "runtime-json"):
                worker.run(args)
            result = json.loads((root / "out/summary.json").read_text())
            self.assertFalse(result["complete"])
            self.assertTrue(result["cpu_reference_check"]["passed"])
            self.assertEqual(result["failure"]["phase"], "runtime_configuration")

    def test_simulated_cuda_assert_does_not_retry_or_probe_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.pth"
            torch.save(fixture(), source)
            runtime = root / "runtime.json"
            runtime.write_text(json.dumps(worker.current_runtime()))
            args = argparse.Namespace(input=str(source), output=str(root / "out"), backend="contiguous",
                                      repeats=3, device="cuda:0", runtime_json=str(runtime))
            with patch.object(worker, "apply_runtime", return_value={}), \
                 patch.object(worker, "restore_layout", side_effect=lambda tensor, layout, device: tensor), \
                 patch.object(worker, "contiguous_index_algorithm", side_effect=RuntimeError("simulated CUDA assert")) as op, \
                 patch.object(torch.cuda, "synchronize") as sync, \
                 patch.object(torch.cuda, "empty_cache") as empty:
                with self.assertRaisesRegex(RuntimeError, "simulated CUDA assert"):
                    worker.run(args)
                self.assertEqual(op.call_count, 1)
                sync.assert_not_called()
                empty.assert_not_called()
            report = json.loads((root / "out/summary.json").read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["failure"]["phase"], "pool_contiguous_repeat_1")
            self.assertTrue(report["cpu_reference_check"]["passed"])


if __name__ == "__main__":
    unittest.main()
