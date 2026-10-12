"""Synthetic integration of real Q pooling, classifier capture and key files.

Only the verified-source loader and image I/O are replaced: no GPU, Train2918
dataset or archived checkpoint is needed. These tests do not claim formal
model or split validation. Native A1 capture, Q pooling, region labels, output
checksums and artifact identity checks use the actual implementations.
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
    """One native clf_2 invocation, a 256-channel feature, and nine Q routes."""

    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.clf_2 = nn.Conv2d(256, 4, 1)
        with torch.no_grad():
            self.base.clf_2.weight.zero_()
            self.base.clf_2.weight[:, :4, 0, 0] = torch.eye(4) * .125
            self.base.clf_2.bias.copy_(torch.tensor([0., .5, -.25, .125]))
        ids = torch.tensor([[0, 0, 1, 1], [1, 2, 2, 0]] +
                           [[0, 0, 1, 1]] * 7, dtype=torch.long)
        q = torch.zeros(1, 9, 4)
        q[:, 0] = torch.tensor([.75, .25, .5, .75])
        q[:, 1] = 1 - q[:, 0]
        self.register_buffer("ids", ids)
        self.register_buffer("q", q)
        self.calls = 0
        self.broken_replay = False
        self.token_drift = 0.0
        self.token_override = None
        self.token_dtype = torch.float32

    def forward(self, pre, _post):
        self.calls += 1
        logits = self.base.clf_2(pre)
        old = logits.clone()
        if self.broken_replay:
            old[:, 1] += 1
        tokens = (torch.arange(128).float()[None, None, :] / 32
                  + torch.arange(1, 4).float()[None, :, None] / 4
                  + pre[0, 0, 0, 0] + self.token_drift)
        if self.token_override is not None:
            tokens = torch.full_like(tokens, self.token_override)
        tokens = tokens.to(self.token_dtype)
        return {"q": self.q, "ids": self.ids, "count": 3,
                "shape": (2, 2), "old_pixel": old,
                "tokens": tokens}


class BuildKeysIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Import before patching: tests exercise these actual helper modules.
        from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import region_targets
        import bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.front = _TinyFront().eval().requires_grad_(False)
        self.names = ["alpha_train_0", "beta_train_1"]
        channel_offset = torch.arange(256, dtype=torch.float32).view(1, 256, 1, 1) / 4
        pixels = torch.tensor([[0., 4.], [8., 12.]]).view(1, 1, 2, 2)
        self.features = [channel_offset + pixels, channel_offset + pixels + 32]
        self.targets = [torch.tensor([[[0, 1], [2, 3]]]),
                        torch.tensor([[[3, 2], [1, 0]]])]
        self.batches = {
            name: (self.features[index], torch.zeros(1, 1, 2, 2),
                   torch.zeros(1, 2, 2), self.targets[index], [name])
            for index, name in enumerate(self.names)
        }
        source_labels, source_tokens = [], []
        with torch.no_grad():
            for feature, target in zip(self.features, self.targets):
                result = self.front(feature, None)
                labels, _ = region_targets(target, result)
                source_labels.append(labels[0])
                source_tokens.append(result["tokens"][0].half())
        self.front.calls = 0
        # Deliberately interleave image identities and use non-sorted regions.
        # Extraction visits image 0 then 1 and must still preserve these rows.
        images, regions = torch.tensor([1, 0, 1, 0]), torch.tensor([2, 1, 0, 2])
        distributions = torch.stack([source_labels[int(image)][int(region)]
                                     for image, region in zip(images, regions)])
        self.original = {
            "tokens": torch.stack([source_tokens[int(image)][int(region)]
                                    for image, region in zip(images, regions)]),
            "distribution": distributions,
            "events": images.clone(), "images": images, "regions": regions,
            "event_names": ["alpha", "beta"],
        }
        self.dataset = self.root / "synthetic-dataset"
        from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import sample_paths
        self.source_files = {}
        for name in self.names:
            for kind, path in sample_paths(self.dataset, name).items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("synthetic image bytes: " + name + "/" + kind).encode())
                self.source_files[str(path)] = digest(path)
        self.bundle = {
            "args": argparse.Namespace(dataset=str(self.dataset), workers=0,
                                       seed=3407, pool_mode="native"),
            "bank_cpu": self.original, "front": self.front,
            "train_names": self.names, "val_names": [],
            "device": torch.device("cpu"),
            "protocol_sha256": "synthetic-protocol",
            "hashes": {"bank_sha256": "synthetic-original-bank", "front_sha256": "synthetic-front"},
            "head_hashes": {"j1": "synthetic-j1"},
            "reference_hashes": {"reference": "synthetic-reference"},
            "own_source_sha256": "synthetic-inherited-code",
        }
        self.loader_calls = []

    def source_loader(self, dataset, names, augment, workers, seed):
        self.loader_calls.append((dataset, list(names), augment, workers, seed))
        return [self.batches[name] for name in names]

    @contextlib.contextmanager
    def mocked_inputs(self, *, loader=None, verification_error=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(build_keys, "load_bundle", return_value=self.bundle))
            stack.enter_context(patch.object(build_keys, "source_hash", return_value="synthetic-paired-code"))
            stack.enter_context(patch.object(shared, "source_hash", return_value="synthetic-paired-code"))
            stack.enter_context(patch(
                "bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils.verify_unchanged",
                return_value={"frozen_source": True}))
            verify = stack.enter_context(patch.object(
                build_keys, "verify_all", side_effect=verification_error,
                wraps=None if verification_error else shared.verify_all))
            stack.enter_context(patch(AUDIT + ".full_image_loader",
                                      side_effect=loader or self.source_loader))
            stack.enter_context(patch(AUDIT + ".parse_batch", side_effect=lambda batch, _device: batch))
            yield verify

    def arguments(self, name="paired", limit_images=0):
        return argparse.Namespace(protocol="synthetic-e0", output=str(self.root / name),
                                  limit_images=limit_images)

    @staticmethod
    def json_file(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def assert_no_capture_hooks(self):
        self.assertEqual(len(self.front.base.clf_2._forward_pre_hooks), 0)
        self.assertEqual(len(self.front.base.clf_2._forward_hooks), 0)

    def test_full_paired_keys_keep_original_rows_and_manual_weighted_means(self):
        cli = self.arguments()
        with self.mocked_inputs() as verify:
            result = build_keys.run(cli)
            bank, manifest = shared.load_paired_bank(cli.output, self.bundle)
        output = Path(cli.output)
        self.assertTrue(result["complete"])
        self.assertFalse(result["training"])
        self.assertFalse(result["test_accessed"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertEqual(self.front.calls, 2)
        self.assertEqual(bank["original_rows"].tolist(), [0, 1, 2, 3])
        for new, old in (("t128", "tokens"), ("distribution", "distribution"),
                         ("images", "images"), ("regions", "regions"), ("events", "events")):
            self.assertEqual(bank[new].dtype, self.original[old].dtype)
            self.assertTrue(torch.equal(bank[new], self.original[old]))
        self.assertEqual(bank["f256"].dtype, torch.float16)
        # Q region-0/1/2 means for pixel [0,4,8,12]:
        # (0*.75+4*.25+12*.25)/1.25 = 3.2
        # (0*.25+8*.5+12*.75)/1.5 = 26/3
        # (4*.75+8*.5)/1.25 = 5.6
        means = torch.tensor([3.2, 26 / 3, 5.6])
        expected = torch.stack([
            torch.arange(256).float() / 4 + means[int(region)] + int(image) * 32
            for image, region in zip(self.original["images"], self.original["regions"])
        ]).half()
        torch.testing.assert_close(bank["f256"], expected, rtol=0, atol=0)
        self.assertEqual(manifest["paired_bank_sha256"], digest(output / "paired_bank.pt"))
        self.assertEqual(manifest["support_count"], 4)
        self.assertEqual(manifest["image_count"], 2)
        self.assertFalse(manifest["smoke"])
        rows = [json.loads(line) for line in (output / "images.jsonl").read_text().splitlines()]
        self.assertEqual([row["train_image_id"] for row in rows], [0, 1])
        self.assertEqual([row["rows"] for row in rows], [2, 2])
        self.assertTrue(all(row["native_a1"]["passed"] for row in rows))
        self.assertEqual(self.loader_calls[0][1], self.names)
        self.assertFalse(self.loader_calls[0][2])
        self.assertEqual(manifest["source_image_fingerprints"], self.source_files)
        verify.assert_called_once()
        self.assertEqual(verify.call_args.args[:2], (self.bundle, "synthetic-paired-code"))
        self.assert_no_capture_hooks()

    def test_smoke_subset_is_explicit_and_cannot_enter_formal_reader(self):
        cli = self.arguments(limit_images=1)
        with self.mocked_inputs():
            result = build_keys.run(cli)
            with self.assertRaisesRegex(ValueError, "Smoke key subset"):
                shared.load_paired_bank(cli.output, self.bundle)
            bank, manifest = shared.load_paired_bank(cli.output, self.bundle, allow_smoke=True)
        self.assertTrue(result["complete"])
        self.assertTrue(result["smoke"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertTrue(manifest["smoke"])
        self.assertEqual(bank["original_rows"].tolist(), [1, 3])
        self.assertEqual(bank["images"].tolist(), [0, 0])
        self.assertEqual(self.front.calls, 1)
        self.assert_no_capture_hooks()

    def test_changed_source_labels_fail_before_completed_bank(self):
        self.original["distribution"][1] = torch.tensor([1., 0., 0., 0.])
        cli = self.arguments()
        with self.mocked_inputs() as verify:
            with self.assertRaisesRegex(ValueError, "support labels differ"):
                build_keys.run(cli)
        output = Path(cli.output)
        summary = self.json_file(output / "summary.json")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["phase"], "Train_support_extraction")
        self.assertEqual(summary["image"], self.names[0])
        self.assertIn("support labels differ", summary["error"])
        self.assertFalse((output / "paired_bank.pt").exists())
        self.assertFalse(self.json_file(output / "manifest.json")["complete"])
        verify.assert_not_called()
        self.assert_no_capture_hooks()

    def test_native_same_pass_gate_failure_stops_extraction(self):
        self.front.broken_replay = True
        cli = self.arguments()
        with self.mocked_inputs():
            with self.assertRaisesRegex(ValueError, "Same-pass native A1 gate"):
                build_keys.run(cli)
        summary = self.json_file(Path(cli.output) / "summary.json")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["phase"], "Train_support_extraction")
        self.assertFalse((Path(cli.output) / "paired_bank.pt").exists())
        self.assert_no_capture_hooks()

    def test_same_q_labels_but_different_t128_replay_is_rejected(self):
        self.front.token_drift = .125
        cli = self.arguments()
        with self.mocked_inputs():
            with self.assertRaisesRegex(ValueError, "T128|token|support.*differ"):
                build_keys.run(cli)
        summary = self.json_file(Path(cli.output) / "summary.json")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["phase"], "Train_support_extraction")
        self.assertFalse((Path(cli.output) / "paired_bank.pt").exists())
        evidence = Path(cli.output) / summary["last_replay_evidence"]["path"]
        self.assertEqual(digest(evidence), summary["last_replay_evidence"]["sha256"])
        recorded = torch.load(evidence, map_location="cpu", weights_only=False)
        self.assertEqual(recorded["replayed_fp32"].dtype, torch.float32)
        self.assertEqual(recorded["archived_tokens"].dtype, torch.float16)
        self.assertGreater(recorded["report"]["definite_drift_count"], 0)
        self.assert_no_capture_hooks()

    def test_half_boundary_replay_preserves_archive_and_saves_uncertainty(self):
        archived = torch.tensor(.04, dtype=torch.float16)
        next_half = torch.nextafter(archived, torch.tensor(float("inf"), dtype=torch.float16))
        midpoint = (float(archived) + float(next_half)) / 2
        self.original["tokens"].fill_(float(archived))
        self.front.token_override = midpoint + 1e-7
        cli = self.arguments()
        with self.mocked_inputs():
            result = build_keys.run(cli)
        self.assertTrue(result["complete"])
        output = Path(cli.output)
        manifest = self.json_file(output / "manifest.json")
        self.assertGreater(manifest["t128_storage_uncertain_elements"], 0)
        self.assertEqual(manifest["t128_strict_rounded_replay_failed_images"], 2)
        bank = torch.load(output / "paired_bank.pt", map_location="cpu", weights_only=False)
        self.assertTrue(torch.equal(bank["t128"], self.original["tokens"]))
        evidence = output / result["last_replay_evidence"]["path"]
        recorded = torch.load(evidence, map_location="cpu", weights_only=False)
        self.assertTrue(recorded["report"]["passed"])
        self.assertFalse(recorded["report"]["legacy_report"]["passed"])
        self.assertEqual(recorded["report"]["definite_drift_count"], 0)
        self.assertEqual(recorded["report"]["atol"], 2e-5)

    def test_half_front_is_not_mislabeled_as_raw_fp32(self):
        self.front.token_dtype = torch.float16
        cli = self.arguments()
        with self.mocked_inputs():
            with self.assertRaisesRegex(ValueError, "refusing to upcast"):
                build_keys.run(cli)
        output = Path(cli.output)
        summary = self.json_file(output / "summary.json")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["raw_token_dtype"], "torch.float16")
        self.assertFalse((output / "paired_bank.pt").exists())

    def test_input_image_mutation_during_extraction_blocks_publication(self):
        cli = self.arguments()
        source = Path(next(iter(self.source_files)))

        def changing_loader(*_args):
            def batches():
                source.write_bytes(source.read_bytes() + b"changed during extraction")
                for name in self.names:
                    yield self.batches[name]
            return batches()

        with self.mocked_inputs(loader=changing_loader):
            with self.assertRaisesRegex(RuntimeError, "immutability"):
                build_keys.run(cli)
        summary = self.json_file(Path(cli.output) / "summary.json")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["phase"], "final_checks")
        self.assertFalse((Path(cli.output) / "paired_bank.pt").exists())
        self.assert_no_capture_hooks()

    def test_short_loader_does_not_claim_all_members_written(self):
        cli = self.arguments()
        with self.mocked_inputs(loader=lambda *args: [self.batches[self.names[0]]]):
            with self.assertRaisesRegex(ValueError, "Not every original support"):
                build_keys.run(cli)
        summary = self.json_file(Path(cli.output) / "summary.json")
        self.assertEqual(summary["phase"], "final_checks")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["image_count"], 1)
        self.assertFalse((Path(cli.output) / "paired_bank.pt").exists())
        self.assert_no_capture_hooks()

    def test_immutable_source_failure_prevents_bank_publication(self):
        cli = self.arguments()
        with self.mocked_inputs(verification_error=RuntimeError("source changed")):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                build_keys.run(cli)
        summary = self.json_file(Path(cli.output) / "summary.json")
        self.assertEqual(summary["phase"], "final_checks")
        self.assertFalse(summary["complete"])
        self.assertFalse((Path(cli.output) / "paired_bank.pt").exists())
        self.assert_no_capture_hooks()

    def test_reader_rejects_checksum_and_provenance_mutation(self):
        cli = self.arguments()
        with self.mocked_inputs():
            build_keys.run(cli)
            output = Path(cli.output)
            manifest_path = output / "manifest.json"
            manifest = self.json_file(manifest_path)
            wrong = dict(manifest, original_bank_sha256="another-source")
            manifest_path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                shared.load_paired_bank(output, self.bundle)
            manifest_path.write_text(json.dumps(manifest))
            with (output / "paired_bank.pt").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                shared.load_paired_bank(output, self.bundle)

    def test_even_rehashed_payload_cannot_change_original_member_labels(self):
        cli = self.arguments()
        with self.mocked_inputs():
            build_keys.run(cli)
            output = Path(cli.output)
            payload = torch.load(output / "paired_bank.pt", map_location="cpu", weights_only=False)
            payload["distribution"][0] = torch.tensor([1., 0., 0., 0.])
            torch.save(payload, output / "paired_bank.pt")
            manifest_path = output / "manifest.json"
            manifest = self.json_file(manifest_path)
            manifest["paired_bank_sha256"] = digest(output / "paired_bank.pt")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Original support changed.*distribution"):
                shared.load_paired_bank(output, self.bundle)

    def test_existing_output_is_preserved(self):
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
