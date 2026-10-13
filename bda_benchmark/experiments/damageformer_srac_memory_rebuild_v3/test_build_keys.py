"""Synthetic CPU integration of fresh paired extraction and saved artifacts.

The real feature/classifier hooks, same-pass A1 gate, Q pooling, labels and
output serialization run here. Frozen source loading and dataset image I/O
are replaced, so these tests make no GPU or BRIGHT Val450 claim.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from . import build_keys, shared
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest


AUDIT = "bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit"


class _TinyFront(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.clf_2 = nn.Conv2d(256, 4, 1)
        with torch.no_grad():
            self.base.clf_2.weight.zero_()
            self.base.clf_2.weight[:, :4, 0, 0] = torch.eye(4) * .125
            self.base.clf_2.bias.copy_(torch.tensor([0., .5, -.25, .125]))
        self.register_buffer("ids", torch.tensor(
            [[0, 0, 1, 1], [1, 2, 2, 0]] + [[0, 0, 1, 1]] * 7, dtype=torch.long))
        q = torch.zeros(1, 9, 4)
        q[:, 0] = torch.tensor([.75, .25, .5, .75])
        q[:, 1] = 1 - q[:, 0]
        self.register_buffer("q", q)
        self.calls = 0
        self.token_dtype = torch.float32
        self.zero_keys = False
        self.broken_native = False
        self.broken_valid = False

    def forward(self, pre, _post):
        self.calls += 1
        logits = self.base.clf_2(pre)
        old = logits.clone()
        if self.broken_native:
            old[:, 1] += 1
        tokens = (torch.arange(128).float()[None, None] / 32
                  + torch.arange(1, 4).float()[None, :, None] / 4 + pre[0, 0, 0, 0])
        if self.zero_keys:
            tokens.zero_()
        valid = torch.ones(1, 3, dtype=torch.bool)
        if self.broken_valid:
            valid[:, 1] = False
        return dict(q=self.q, ids=self.ids, count=3, shape=(2, 2),
                    old_pixel=old, tokens=tokens.to(self.token_dtype), valid_regions=valid,
                    mass=torch.tensor([[1.25, 1.5, 1.25]]))


class _IdentityOnlyBank(dict):
    def __getitem__(self, key):
        if key in ("tokens", "distribution"):
            raise AssertionError("Historical numeric payload was read: " + key)
        return super().__getitem__(key)


class FreshBuildIntegrationTests(unittest.TestCase):
    def setUp(self):
        import bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.front = _TinyFront().eval().requires_grad_(False)
        self.names = ["alpha_train_0", "beta_train_1"]
        channels = torch.arange(256).float().view(1, 256, 1, 1) / 4
        pixels = torch.tensor([[0., 4.], [8., 12.]]).view(1, 1, 2, 2)
        self.features = [channels + pixels, channels + pixels + 32]
        self.targets = [torch.tensor([[[0, 1], [2, 3]]]),
                        torch.tensor([[[3, 2], [1, 0]]])]
        self.batches = {
            name: (self.features[i], torch.zeros(1, 1, 2, 2),
                   torch.zeros(1, 2, 2), self.targets[i], [name])
            for i, name in enumerate(self.names)
        }
        images = torch.tensor([1, 0, 1, 0])
        self.original = dict(tokens=torch.full((4, 128), -999., dtype=torch.float16),
                             distribution=torch.tensor([[0., 0., 0., 1.]] * 4),
                             events=images.clone(), images=images,
                             regions=torch.tensor([2, 1, 0, 2]), event_names=["alpha_train", "beta_train"])
        self.dataset = self.root / "dataset"
        from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import sample_paths
        self.source_files = {}
        for name in self.names:
            for kind, path in sample_paths(self.dataset, name).items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("synthetic " + name + "/" + kind).encode())
                self.source_files[str(path)] = digest(path)
        self.bundle = dict(args=argparse.Namespace(dataset=str(self.dataset), workers=0,
                                                 seed=3407, pool_mode="native", profile="legacy"),
                           bank_cpu=self.original, front=self.front,
                           train_names=self.names, val_names=["gamma_val_0"], device=torch.device("cpu"),
                           protocol_sha256="a" * 64, protocol_path=str(self.root / "source.json"),
                           hashes={"bank_sha256": "b" * 64, "front_sha256": "c" * 64},
                           head_hashes={"j1": "d" * 64}, reference_hashes={"reference": "e" * 64},
                           own_source_sha256="f" * 64,
                           runtime_settings={"profile": "legacy", "pool_mode": "native"})
        self.loader_calls = []

    def source_loader(self, dataset, names, augment, workers, seed):
        self.loader_calls.append((dataset, list(names), augment, workers, seed))
        return [self.batches[name] for name in names]

    @contextlib.contextmanager
    def mocked_inputs(self, loader=None, verification_error=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(build_keys, "load_bundle", return_value=self.bundle))
            stack.enter_context(patch.object(build_keys, "source_hash", return_value="0" * 64))
            stack.enter_context(patch.object(shared, "source_hash", return_value="0" * 64))
            stack.enter_context(patch("bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils.verify_unchanged",
                                      return_value={"frozen_source": True}))
            verify = stack.enter_context(patch.object(build_keys, "verify_all", side_effect=verification_error,
                                                     wraps=None if verification_error else shared.verify_all))
            stack.enter_context(patch(AUDIT + ".full_image_loader", side_effect=loader or self.source_loader))
            stack.enter_context(patch(AUDIT + ".parse_batch", side_effect=lambda batch, _device: batch))
            yield verify

    def arguments(self, name="fresh", limit=0):
        return argparse.Namespace(protocol="source-anchor", output=str(self.root / name), limit_images=limit)

    def payload(self, cli):
        return torch.load(Path(cli.output) / "paired_bank.pt", map_location="cpu", weights_only=False)

    def record(self, cli, filename="summary.json"):
        return json.loads((Path(cli.output) / filename).read_text(encoding="utf-8"))

    def assert_hooks_removed(self):
        self.assertEqual(len(self.front._forward_hooks), 0)
        self.assertEqual(len(self.front.base.clf_2._forward_pre_hooks), 0)
        self.assertEqual(len(self.front.base.clf_2._forward_hooks), 0)

    def test_one_forward_paired_fp32_labels_and_exact_member_order(self):
        cli = self.arguments()
        with self.mocked_inputs() as verify:
            result = build_keys.run(cli)
        bank = self.payload(cli)
        self.assertTrue(result["complete"])
        self.assertFalse(result["training"])
        self.assertFalse(result["test_accessed"])
        self.assertFalse(result["archived_replay_required"])
        self.assertFalse(result["historical_fp32_reproduction_claim"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertEqual(self.front.calls, 2)
        self.assertEqual(result["extraction_forward_count"], 2)
        self.assertEqual(bank["original_rows"].tolist(), [0, 1, 2, 3])
        self.assertEqual(bank["member_ids"].tolist(), [0, 1, 2, 3])
        for field in ("images", "regions", "events"):
            self.assertTrue(torch.equal(bank[field], self.original[field]))
        for field in ("t128", "f256", "distribution", "mass", "label_mass"):
            self.assertEqual(bank[field].dtype, torch.float32)
        means = torch.tensor([3.2, 26 / 3, 5.6])
        expected = torch.stack([
            torch.arange(256).float() / 4 + means[int(region)] + int(image) * 32
            for image, region in zip(bank["images"], bank["regions"])
        ])
        torch.testing.assert_close(bank["f256"], expected, atol=2e-5, rtol=2e-5)
        expected_t128 = torch.stack([
            torch.arange(128).float() / 32 + (int(region) + 1) / 4 + int(image) * 32
            for image, region in zip(bank["images"], bank["regions"])
        ])
        torch.testing.assert_close(bank["t128"], expected_t128, rtol=0, atol=0)
        torch.testing.assert_close(bank["distribution"].sum(-1), torch.ones(4), rtol=2e-5, atol=2e-5)
        self.assertFalse(torch.equal(bank["distribution"], self.original["distribution"]))
        manifest = self.record(cli, "manifest.json")
        self.assertEqual(manifest["source_image_fingerprints"], self.source_files)
        self.assertEqual(manifest["paired_bank_sha256"], digest(Path(cli.output) / "paired_bank.pt"))
        self.assertEqual(manifest["rebuild_protocol_sha256"], digest(Path(cli.output) / "protocol.json"))
        self.assertTrue(manifest["saved_fp32_roundtrip_exact"])
        frames = [json.loads(line) for line in (Path(cli.output) / "images.jsonl").read_text().splitlines()]
        self.assertEqual([frame["forward_count"] for frame in frames], [1, 2])
        self.assertTrue(all(frame["q_check"]["passed"] and frame["label_check"]["passed"] for frame in frames))
        self.assertEqual(self.loader_calls[0][1], self.names)
        self.assertFalse(self.loader_calls[0][2])
        verify.assert_called_once()
        self.assert_hooks_removed()

    def test_radically_changed_archived_values_do_not_change_fresh_values(self):
        first, second = self.arguments("first"), self.arguments("second")
        with self.mocked_inputs():
            build_keys.run(first)
            self.original["tokens"].fill_(float("nan"))
            self.original["distribution"].fill_(-100000.)
            build_keys.run(second)
        for field in ("t128", "f256", "distribution", "mass", "label_mass", "eligible"):
            self.assertTrue(torch.equal(self.payload(first)[field], self.payload(second)[field]))
        self.assert_hooks_removed()

    def test_archived_numeric_payload_is_never_read(self):
        self.bundle["bank_cpu"] = _IdentityOnlyBank(self.original)
        with self.mocked_inputs():
            self.assertTrue(build_keys.run(self.arguments())["complete"])

    def test_integer_archived_identity_storage_becomes_canonical_int64(self):
        for field in ("images", "events", "regions"):
            self.original[field] = self.original[field].int()
        cli = self.arguments()
        with self.mocked_inputs():
            build_keys.run(cli)
        bank = self.payload(cli)
        for field in ("images", "events", "regions", "original_rows", "member_ids"):
            self.assertEqual(bank[field].dtype, torch.int64)

    def test_zero_norm_t128_retains_member_and_joint_ineligible(self):
        self.front.zero_keys = True
        cli = self.arguments()
        with self.mocked_inputs():
            summary = build_keys.run(cli)
        bank = self.payload(cli)
        self.assertEqual(len(bank["t128"]), 4)
        self.assertEqual(bank["original_rows"].tolist(), [0, 1, 2, 3])
        self.assertEqual(summary["joint_ineligible_count"], 4)
        self.assertFalse(bool(bank["eligible"].any()))
        self.assertEqual(self.record(cli, "manifest.json")["zero_norm_t128_count"], 4)

    def test_smoke_keeps_only_original_order_members_of_first_image(self):
        cli = self.arguments(limit=1)
        with self.mocked_inputs():
            summary = build_keys.run(cli)
        self.assertTrue(summary["smoke"])
        self.assertEqual(self.payload(cli)["original_rows"].tolist(), [1, 3])
        self.assertEqual(self.front.calls, 1)

    def test_fp16_current_tokens_are_rejected_without_upcasting(self):
        self.front.token_dtype = torch.float16
        cli = self.arguments()
        with self.mocked_inputs():
            with self.assertRaisesRegex(ValueError, "refusing to upcast"):
                build_keys.run(cli)
        self.assertFalse(self.record(cli)["complete"])
        self.assertFalse((Path(cli.output) / "paired_bank.pt").exists())
        self.assert_hooks_removed()

    def test_invalid_selected_current_region_blocks_completed_bank(self):
        self.front.broken_valid = True
        cli = self.arguments()
        with self.mocked_inputs():
            with self.assertRaisesRegex(ValueError, "valid current region"):
                build_keys.run(cli)
        self.assertFalse(self.record(cli)["complete"])
        self.assert_hooks_removed()

    def test_zero_valid_label_mass_does_not_use_archived_labels(self):
        name = self.names[0]
        pre, post, loc, target, names = self.batches[name]
        self.batches[name] = (pre, post, loc, torch.full_like(target, 255), names)
        cli = self.arguments()
        with self.mocked_inputs():
            with self.assertRaisesRegex(ValueError, "label mass"):
                build_keys.run(cli)
        self.assertFalse(self.record(cli)["complete"])
        self.assert_hooks_removed()

    def test_native_current_forward_failure_is_retained_with_traceback(self):
        self.front.broken_native = True
        cli = self.arguments()
        with self.mocked_inputs():
            with self.assertRaisesRegex(ValueError, "Same-pass native A1"):
                build_keys.run(cli)
        self.assertFalse(self.record(cli)["complete"])
        self.assertTrue((Path(cli.output) / "traceback.txt").is_file())
        self.assert_hooks_removed()

    def test_short_loader_rejects_unwritten_members(self):
        cli = self.arguments()
        with self.mocked_inputs(loader=lambda *_args: [self.batches[self.names[0]]]):
            with self.assertRaisesRegex(ValueError, "Not every fixed support"):
                build_keys.run(cli)
        self.assertFalse(self.record(cli)["complete"])
        self.assertEqual(self.record(cli)["image_count"], 1)
        self.assertFalse((Path(cli.output) / "paired_bank.pt").exists())
        self.assert_hooks_removed()

    def test_input_file_change_blocks_bank_publication(self):
        source = Path(next(iter(self.source_files)))
        def changed_loader(*_args):
            source.write_bytes(source.read_bytes() + b"mutated")
            return [self.batches[name] for name in self.names]
        cli = self.arguments()
        with self.mocked_inputs(loader=changed_loader):
            with self.assertRaisesRegex(RuntimeError, "immutability"):
                build_keys.run(cli)
        self.assertFalse(self.record(cli)["complete"])
        self.assertFalse((Path(cli.output) / "paired_bank.pt").exists())
        self.assert_hooks_removed()

    def test_immutable_source_change_blocks_bank_publication(self):
        cli = self.arguments()
        with self.mocked_inputs(verification_error=RuntimeError("source changed")):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                build_keys.run(cli)
        self.assertFalse(self.record(cli)["complete"])
        self.assertFalse((Path(cli.output) / "paired_bank.pt").exists())
        self.assert_hooks_removed()

    def test_missing_current_region_identity_is_rejected(self):
        self.original["regions"][1] = 30
        cli = self.arguments()
        with self.mocked_inputs():
            with self.assertRaisesRegex(ValueError, "region.*grid"):
                build_keys.run(cli)
        self.assertFalse(self.record(cli)["complete"])
        self.assert_hooks_removed()

    def test_existing_result_directory_is_never_overwritten(self):
        cli = self.arguments()
        output = Path(cli.output)
        output.mkdir()
        marker = output / "user-file.txt"
        marker.write_text("preserve me")
        with self.mocked_inputs():
            with self.assertRaises(FileExistsError):
                build_keys.run(cli)
        self.assertEqual(marker.read_text(), "preserve me")
        self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
