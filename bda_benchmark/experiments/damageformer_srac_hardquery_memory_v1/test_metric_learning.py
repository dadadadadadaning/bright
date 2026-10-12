"""CPU checks for E2 supervision, exclusions, sampling, and learned gradients."""
from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch

from .metric_learning import HardQuerySampler, RetrievalProjector, query_hardness, soft_cross_event_loss


class MetricLearningTests(unittest.TestCase):
    def test_soft_targets_equal_manual_overlap_ce(self):
        query = torch.tensor([[1., .2]])
        support = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        distribution = torch.tensor([[0., .7, .3, 0.]])
        labels = torch.tensor([[0., 1., 0., 0.], [0., 0., 1., 0.], [1., 0., 0., 0.]])
        projector = RetrievalProjector(2)
        loss, stats = soft_cross_event_loss(
            query, distribution, torch.tensor([0]), torch.tensor([0]),
            support, labels, torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), projector, 1.)
        logits = projector(query) @ projector(support).T
        expected = -(torch.tensor([[.7, .3, 0.]]) * logits.log_softmax(-1)).sum()
        hard_label_loss = -logits.log_softmax(-1)[0, 0]
        self.assertTrue(torch.allclose(loss, expected, atol=1e-7))
        self.assertGreater(abs(float(loss - hard_label_loss)), .1)
        self.assertEqual(stats["mixed_used"], 1)

    def test_same_event_and_same_image_excluded_before_normalization(self):
        query = torch.tensor([[1., 1.]])
        support = torch.tensor([[100., 0.], [0., 100.], [1., 0.], [0., 1.]])
        labels = torch.tensor([[0., 1., 0., 0.]]).repeat(4, 1)
        projector = RetrievalProjector(2)
        loss, stats = soft_cross_event_loss(
            query, labels[:1], torch.tensor([5]), torch.tensor([9]),
            support, labels, torch.tensor([5, 6, 6, 7]), torch.tensor([1, 9, 2, 3]), projector, .3)
        expected = -(projector(query) @ projector(support[2:]).T / .3).log_softmax(-1).mean()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        self.assertEqual(stats["allowed_pair_count"], 2)
        self.assertEqual(stats["positive_overlap_pairs"], 2)

    def test_absent_positive_returns_connected_zero(self):
        query = torch.randn(2, 3, requires_grad=True)
        support = torch.randn(3, 3, requires_grad=True)
        projector = RetrievalProjector(3)
        loss, stats = soft_cross_event_loss(
            query, torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1), torch.tensor([0, 0]), torch.tensor([0, 0]),
            support, torch.tensor([[0., 1., 0., 0.]]).repeat(3, 1),
            torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), projector)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(stats["skipped_selected_no_positive"], 2)
        self.assertEqual(stats["used_anchor_count"], 0)
        self.assertIsNone(query.grad)
        self.assertIsNone(support.grad)
        self.assertTrue(torch.equal(projector.linear.weight.grad, torch.zeros(3, 3)))

    def test_all_excluded_is_finite_zero(self):
        projector = RetrievalProjector(2)
        loss, stats = soft_cross_event_loss(
            torch.ones(1, 2), torch.tensor([[1., 0., 0., 0.]]), torch.tensor([0]), torch.tensor([0]),
            torch.ones(2, 2), torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
            torch.tensor([0, 0]), torch.tensor([1, 2]), projector)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(stats["allowed_pair_count"], 0)

    def test_learning_reduces_loss_and_keeps_features_frozen(self):
        torch.manual_seed(12)
        query = torch.tensor([[1., .6]], requires_grad=True)
        support = torch.tensor([[.7, .7], [-.2, 1.], [-1., 0.]], requires_grad=True)
        distribution = torch.tensor([[0., .8, .2, 0.]])
        labels = torch.tensor([[0., 1., 0., 0.], [0., 0., 1., 0.], [1., 0., 0., 0.]])
        projector = RetrievalProjector(2)
        optimizer = torch.optim.AdamW(projector.parameters(), lr=.03, weight_decay=0.)
        initial = None
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = soft_cross_event_loss(query, distribution, torch.tensor([0]), torch.tensor([0]),
                support, labels, torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), projector, .5)
            if initial is None:
                initial = float(loss)
            loss.backward()
            self.assertTrue(torch.isfinite(projector.linear.weight.grad).all())
            optimizer.step()
        self.assertLess(float(loss), initial)
        self.assertIsNone(query.grad)
        self.assertIsNone(support.grad)
        self.assertFalse(torch.equal(projector.linear.weight.detach(), torch.eye(2)))

    def test_sampler_preserves_difficult_and_ordinary_per_class(self):
        target = torch.eye(4).repeat_interleave(8, 0)
        confidence = torch.tensor([.1, .2, .3, .4, .6, .7, .8, .9]).repeat(4)
        probability = (1 - confidence[:, None]) / 3 * (1 - target) + confidence[:, None] * target
        sampler = HardQuerySampler(target, probability)
        indices, stats = sampler.sample(6, torch.Generator().manual_seed(3))
        self.assertEqual(len(indices), 24)
        for class_id in range(4):
            self.assertEqual(stats["by_class"][str(class_id)]["ordinary"], 3)
            self.assertEqual(stats["by_class"][str(class_id)]["difficult"], 3)
            selected = indices[sampler.classes[indices] == class_id]
            self.assertEqual(int((sampler.hardness[selected] >= .8).sum()), 3)
            self.assertEqual(sampler.inventory[str(class_id)]["difficult_count"], 2)

    def test_hardness_uses_all_target_mass(self):
        target = torch.tensor([[0., .6, .4, 0.]])
        prediction = torch.tensor([[0., .2, .8, 0.]])
        self.assertAlmostEqual(float(query_hardness(target, prediction)), .56, places=6)

    def test_empty_sampler_and_invalid_distribution(self):
        sampler = HardQuerySampler(torch.empty(0, 4), torch.empty(0, 4))
        indices, _ = sampler.sample(2, torch.Generator().manual_seed(1))
        self.assertEqual(len(indices), 0)
        with self.assertRaises(ValueError):
            HardQuerySampler(torch.tensor([[.1, .1, .1, .1]]), torch.tensor([[.25] * 4]))

    def test_matched_twenty_update_training_arms(self):
        from .train_metric import train_arm
        torch.manual_seed(2)
        classes = torch.arange(4).repeat(2)
        target = torch.eye(4)[classes]
        tokens = torch.randn(8, 5)
        events = torch.tensor([0] * 4 + [1] * 4)
        images = torch.arange(8)
        support = dict(tokens=tokens, distribution=target, classes=classes, events=events, images=images)
        probability = .8 * target + .05
        sampler = HardQuerySampler(target, probability)

        class Store:
            manifest_sha256 = "synthetic"

            def batch(self, indices, device):
                return dict(tokens=tokens[indices].to(device), distribution=target[indices].to(device),
                            base_probability=probability[indices].to(device),
                            events=events[indices].to(device), images=images[indices].to(device))

        args = SimpleNamespace(lr=.001, seed=3407, steps=20, per_class=2,
                               temperature=.1, hard_fraction=.5, smoke=True)
        with tempfile.TemporaryDirectory() as temporary:
            for arm in ("bank", "hard"):
                payload, summary = train_arm(args, arm, support, Store(), sampler, Path(temporary), {}, {})
                self.assertEqual(payload["step"], 20)
                self.assertEqual(summary["optimizer_updates"], 20)
                self.assertGreater(summary["nonzero_gradient_updates"], 0)
                self.assertTrue(payload["complete"])
                self.assertEqual(payload["format"], "SRAC_HARDQUERY_METRIC_V1")
                self.assertFalse(torch.equal(payload["model"]["linear.weight"], torch.eye(5)))

    def test_query_cache_verifies_train_identity_hash_and_invalid_regions(self):
        from .io_utils import artifact_identity, digest, write_json
        from .train_metric import QueryStore
        bundle = dict(hashes={"front": "same"}, head_hashes={"j1": "same"}, reference_hashes={},
                      protocol_sha256="verified", train_names=[f"event{i}_0001" for i in range(4)],
                      val_names=["different_0001"], input_paths={}, inputs_before={},
                      bank_cpu=dict(tokens=torch.empty(1, 3), event_names=[f"event{i}" for i in range(4)]))
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils.code_hash",
            return_value="test-code"):
            root = Path(temporary)
            (root / "shards").mkdir()
            manifest = dict(format="SRAC_HARDQUERY_QUERIES_V1", complete=True, smoke=True,
                            train_only=True, test_accessed=False, shards=[], **artifact_identity(bundle))
            for index, name in enumerate(bundle["train_names"]):
                shard = dict(tokens=torch.ones(2, 3) * index,
                             distribution=torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 0.]]),
                             base_probability=torch.tensor([[.7, .1, .1, .1], [0., 0., 0., 0.]]),
                             weights=torch.tensor([1., 0.]), mass=torch.tensor([1., 0.]),
                             event=index, image=index, name=name, region_ids=torch.arange(2))
                relative = f"shards/{index}.pt"
                torch.save(shard, root / relative)
                manifest["shards"].append(dict(path=relative, sha256=digest(root / relative),
                                                name=name, event=index, image=index))
            write_json(root / "query_manifest.json", manifest)
            store = QueryStore(root, bundle, True)
            batch = store.batch(torch.tensor([3, 0, 2]), torch.device("cpu"))
            self.assertTrue(torch.equal(batch["images"], torch.tensor([3, 0, 2])))
            self.assertTrue(torch.equal(batch["tokens"][:, 0], torch.tensor([3., 0., 2.])))
            self.assertEqual(store.inventory["valid_regions"], 4)
            del store
            manifest["shards"][0]["sha256"] = "changed"
            write_json(root / "query_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                QueryStore(root, bundle, True)
            manifest["shards"][0]["sha256"] = digest(root / "shards/0.pt")
            manifest["shards"][0]["name"] = bundle["val_names"][0]
            write_json(root / "query_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                QueryStore(root, bundle, True)
            manifest["shards"][0]["path"] = "../outside.pt"
            write_json(root / "query_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "escapes"):
                QueryStore(root, bundle, True)

    def test_saved_metric_run_is_consumed_with_identity_budget_and_hash_checks(self):
        from . import train_metric
        from .io_utils import digest, read_json, write_json
        from .runtime import load_new_metrics
        torch.manual_seed(11)
        classes = torch.arange(4).repeat(2)
        distribution = torch.eye(4)[classes]
        probability = .8 * distribution + .05
        tokens = torch.randn(8, 5)
        events, images = torch.tensor([0] * 4 + [1] * 4), torch.arange(8)
        bundle = dict(hashes={"front": "frozen"}, head_hashes={"j1": "frozen"},
                      reference_hashes={"original_validation": "unchanged"},
                      protocol_sha256="synthetic-protocol", runtime_settings={"device": "cpu"},
                      device=torch.device("cpu"), bank_cpu=dict(
                          tokens=tokens, distribution=distribution, events=events, images=images))

        class Store:
            manifest_sha256 = "synthetic-Train-query-manifest"
            inventory = dict(images=8, valid_regions=8)
            fields = dict(distribution=distribution, base_probability=probability)

            def batch(self, indices, device):
                return dict(tokens=tokens[indices].to(device), distribution=distribution[indices].to(device),
                            base_probability=probability[indices].to(device),
                            events=events[indices].to(device), images=images[indices].to(device))

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            train_metric, "load_verified_bundle", return_value=bundle), patch.object(
            train_metric, "QueryStore", return_value=Store()), patch.object(
            train_metric, "verify_unchanged", return_value={"synthetic_frozen_sources": True}):
            output = Path(temporary) / "metric_run"
            args = SimpleNamespace(output=str(output), queries="synthetic-Train", smoke=True,
                                   steps=20, arms=["bank", "hard"], per_class=2, purity=.8,
                                   lr=.001, temperature=.1, hard_fraction=.5, hard_quantile=.75,
                                   seed=3407, device="cpu")
            train_metric.run(args)
            summary = read_json(output / "summary.json")
            self.assertEqual(summary["steps"], 20)
            self.assertEqual(set(summary["arms"]), {"bank", "hard"})
            for arm in ("bank", "hard"):
                self.assertEqual(summary["checkpoint_sha256"][arm], digest(output / f"metric_{arm}_last.pth"))
            projectors, ledger = load_new_metrics(output, bundle, smoke=True)
            self.assertEqual(set(projectors), {"bank", "hard"})
            self.assertEqual(ledger, summary["checkpoint_sha256"])
            self.assertTrue(all(not p.requires_grad for model in projectors.values() for p in model.parameters()))

            for field, wrong, message in (
                ("steps", 19, "equal-budget"),
                ("protocol_sha256", "another-protocol", "provenance mismatch"),
                ("checkpoint_sha256", {"bank": "changed", "hard": ledger["hard"]}, "hash mismatch"),
            ):
                changed = dict(summary)
                changed[field] = wrong
                write_json(output / "summary.json", changed)
                with self.assertRaisesRegex(ValueError, message):
                    load_new_metrics(output, bundle, smoke=True)
            write_json(output / "summary.json", summary)


if __name__ == "__main__":
    unittest.main()
