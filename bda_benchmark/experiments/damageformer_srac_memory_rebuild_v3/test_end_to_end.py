"""Synthetic CPU build -> verified reload -> diagnostic -> report integration.

Only frozen-source loading, source immutability and dataset decoding are mocked.
The new bank, member/split/protocol/journal/source-image and saved-prediction
checks all run normally. Even the 450-image case uses tiny synthetic tensors;
it tests the formal claim gate and is not real BRIGHT Val450 evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import build_keys, diagnose, report, shared
from .test_build_keys import _TinyFront
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import sample_paths


AUDIT = "bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit"


class _PredictionFront(_TinyFront):
    def forward(self, pre, post):
        result = super().forward(pre, post)
        result["prob"] = result["old_pixel"].softmax(1)
        return result


class _ForbiddenReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("This retrieval experiment must never run a Reader")


class EndToEndSyntheticTests(unittest.TestCase):
    def setUp(self):
        # The existing loader module supplies hooks; this does not open data.
        import bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dataset = self.root / "synthetic_dataset"
        self.train = ["alpha_train_0", "beta_train_1"] + [
            f"unused_train_{index}" for index in range(2, 2918)]
        self.val = [f"gamma_val_{index}" for index in range(450)]
        self.front = _PredictionFront().eval().requires_grad_(False)
        self.reader = _ForbiddenReader().eval().requires_grad_(False)
        self.metric = _ForbiddenReader().eval().requires_grad_(False)
        channels = torch.arange(256).float().view(1, 256, 1, 1) / 4
        pixels = torch.tensor([[0., 4.], [8., 12.]]).view(1, 1, 2, 2)
        self.features = [channels + pixels, channels + pixels + 32]
        self.targets = [torch.tensor([[[0, 1], [2, 3]]]),
                        torch.tensor([[[3, 2], [1, 0]]])]
        images = torch.tensor([1, 0, 1, 0])
        self.original = dict(
            tokens=torch.full((4, 128), -999., dtype=torch.float16),
            distribution=torch.tensor([[0., 0., 0., 1.]] * 4),
            images=images, events=images.clone(), regions=torch.tensor([2, 1, 0, 2]),
            event_names=["alpha_train", "beta_train"])
        self.anchor = self.root / "synthetic_frozen_source.json"
        self.anchor.write_text(json.dumps({"synthetic": True, "test_accessed": False}))
        self.anchor_before = self.anchor.read_bytes()
        self.bundle = dict(
            args=argparse.Namespace(dataset=str(self.dataset), workers=0, seed=3407,
                                    pool_mode="native", profile="legacy"),
            bank_cpu=self.original, front=self.front, decoders={"j1": self.reader}, metric=self.metric,
            train_names=self.train, val_names=self.val, device=torch.device("cpu"),
            protocol_sha256=digest(self.anchor), protocol_path=str(self.anchor),
            hashes={"bank_sha256": "b" * 64, "front_sha256": "c" * 64},
            head_hashes={"j1": "d" * 64}, reference_hashes={"reference": "e" * 64},
            own_source_sha256="f" * 64, runtime_settings={"profile": "legacy", "pool_mode": "native"})
        self.loader_calls = []
        self.keys = self.root / "fresh_keys"
        self._make_source_files(self.train[:2])

    def _make_source_files(self, names):
        for name in names:
            for kind, path in sample_paths(self.dataset, name).items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("synthetic input " + name + "/" + kind).encode())

    def _source_loader(self, dataset, names, augment, workers, seed):
        self.assertEqual(dataset, str(self.dataset))
        self.assertFalse(augment)
        self.loader_calls.append(list(names))
        batches = []
        for index, name in enumerate(names):
            if name in self.train[:2]:
                tensor_index = self.train.index(name)
            else:
                self.assertIn(name, self.val)
                tensor_index = index % 2
            batches.append((self.features[tensor_index], torch.zeros(1, 1, 2, 2),
                            torch.zeros(1, 2, 2), self.targets[tensor_index], [name]))
        return batches

    @contextlib.contextmanager
    def _mock_source_only(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(build_keys, "load_bundle", return_value=self.bundle))
            stack.enter_context(patch.object(shared, "load_bundle", return_value=self.bundle))
            stack.enter_context(patch(
                "bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils.verify_unchanged",
                return_value={"synthetic_frozen_source": True}))
            stack.enter_context(patch(AUDIT + ".full_image_loader", side_effect=self._source_loader))
            stack.enter_context(patch(AUDIT + ".parse_batch", side_effect=lambda batch, _device: batch))
            # Output suppression does not alter any journal or saved report.
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield

    def _build(self):
        return build_keys.run(argparse.Namespace(protocol=str(self.anchor),
            output=str(self.keys), limit_images=0))

    def _diagnose(self, *, limit=2, output="diagnostic"):
        self._make_source_files(self.val[:limit or len(self.val)])
        return diagnose.run(argparse.Namespace(protocol=str(self.anchor), keys=str(self.keys),
            output=str(self.root / output), limit=limit, temperature=.1))

    def _assert_no_reader_or_training(self, result):
        self.assertFalse(result["training"])
        self.assertFalse(result["test_accessed"])
        self.assertEqual(self.reader.calls, 0)
        self.assertEqual(self.metric.calls, 0)
        self.assertFalse(any(model.training for model in (self.front, self.reader, self.metric)))
        self.assertFalse(any(parameter.requires_grad for parameter in self.front.parameters()))
        self.assertEqual(self.anchor.read_bytes(), self.anchor_before)

    def test_new_bank_verified_reload_two_val_images_and_real_report(self):
        with self._mock_source_only():
            built = self._build()
            bank, manifest = shared.load_paired_bank(self.keys, self.bundle)
            result = self._diagnose()
            text = report.render(self.root / "diagnostic")
        self.assertTrue(built["complete"])
        self.assertFalse(manifest["smoke"])
        self.assertEqual(len(bank["t128"]), 4)
        self.assertEqual(self.front.calls, 4)  # two source images + two validation images
        for field in ("t128", "f256", "distribution", "mass", "label_mass"):
            self.assertEqual(bank[field].dtype, torch.float32)
        for field in ("images", "events", "regions"):
            self.assertTrue(torch.equal(bank[field], self.original[field]))
        protocol = json.loads((self.keys / "protocol.json").read_text())
        self.assertEqual(protocol["format"], shared.PROTOCOL_FORMAT)
        self.assertEqual(protocol["source_e0_sha256"], digest(self.anchor))
        self.assertEqual(protocol["split_identity"]["train_count"], 2918)
        self.assertEqual(protocol["split_identity"]["val_count"], 450)
        self.assertFalse(protocol["historical_replay_required"])
        self.assertFalse(protocol["historical_fp32_reproduction_claim"])
        self.assertEqual(manifest["images_journal_sha256"], digest(self.keys / "images.jsonl"))
        self.assertTrue(result["execution_complete"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertEqual(result["image_count"], 2)
        self.assertTrue(all(value is True for value in result["validation"].values()))
        self.assertEqual(self.loader_calls, [self.train[:2], self.val[:2]])
        for image_index in range(2):
            with np.load(self.root / "diagnostic" / "predictions" / f"{image_index:04d}.npz",
                         allow_pickle=False) as saved:
                for key in ("t128", "f256"):
                    self.assertEqual(saved["query_" + key].dtype, np.float32)
                    for prefix in ("support_", "scores_"):
                        self.assertTrue(np.array_equal(saved[f"{prefix}{key}_top8"],
                                                       saved[f"{prefix}{key}_top64"][:, :8]))
                self.assertEqual(set(saved["support_t128_top8"][0, :4]), {0, 1, 2, 3})
        self.assertIn("T128/F256", text)
        self.assertIn("当前为未训练的 label vote", text)
        self.assertIn("smoke", text)
        self._assert_no_reader_or_training(result)

    def test_train_source_mutation_after_build_blocks_diagnostic_before_forward(self):
        with self._mock_source_only():
            self._build()
            path = sample_paths(self.dataset, self.train[0])["label"]
            path.write_bytes(path.read_bytes() + b"changed after extraction")
            before = self.front.calls
            with self.assertRaisesRegex(ValueError, "Source Train image changed since extraction"):
                self._diagnose()
        self.assertEqual(self.front.calls, before)
        summary = json.loads((self.root / "diagnostic" / "summary.json").read_text())
        self.assertEqual(summary["image_count"], 0)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["execution_complete"])
        self.assertFalse(summary["full_validation_evidence"])
        self.assertEqual(summary["phase"], "load")
        self.assertTrue((self.root / "diagnostic" / "traceback.txt").is_file())
        self._assert_no_reader_or_training(summary)

    def test_swapped_source_protocol_binding_blocks_diagnostic_before_forward(self):
        with self._mock_source_only():
            self._build()
            self.bundle["protocol_sha256"] = "1" * 64
            before = self.front.calls
            with self.assertRaisesRegex(ValueError, "provenance differs: protocol_sha256"):
                self._diagnose()
        self.assertEqual(self.front.calls, before)
        summary = json.loads((self.root / "diagnostic" / "summary.json").read_text())
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["image_count"], 0)
        self._assert_no_reader_or_training(summary)

    def test_synthetic_450_actual_loop_passes_formal_gate_without_bright_claim(self):
        with self._mock_source_only():
            self._build()
            result = self._diagnose(limit=0, output="synthetic_formal_450")
            summary, reopened = report.read_report(self.root / "synthetic_formal_450")
            text = report.render(self.root / "synthetic_formal_450")
        self.assertEqual(result["image_count"], 450)
        self.assertEqual(self.front.calls, 452)
        self.assertTrue(result["complete"])
        self.assertTrue(result["execution_complete"])
        self.assertTrue(result["full_validation_evidence"])
        self.assertFalse(result["smoke"])
        self.assertFalse(result["paired_bank_smoke"])
        self.assertTrue(summary["complete"])
        self.assertEqual(reopened, result)
        self.assertTrue(all(value is True for value in result["validation"].values()))
        self.assertEqual(len((self.root / "synthetic_formal_450" / "images.jsonl").read_text().splitlines()), 450)
        for arm, values in result["metrics"].items():
            self.assertEqual(values["count"], 1800)
            self.assertEqual(values["confusion"], result["confusion"][arm])
        self.assertIn("完整 Val450 新库配对诊断通过", text)
        self._assert_no_reader_or_training(result)


if __name__ == "__main__":
    unittest.main()
