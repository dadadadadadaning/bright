"""Synthetic contracts for Train-only misleading-reference metric fitting."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ..damageformer_srac_hardquery_memory_v1.metric_learning import HardQuerySampler
from ..damageformer_srac_hardquery_memory_v1.io_utils import digest, write_json
from ..damageformer_srac_qmetric_v1.metric import RetrievalProjector
from .sampling import RetrievalMisleadSampler, mine_reference_risk


class SyntheticStore:
    def __init__(self, tokens, distribution, base_probability, events, images):
        self.tokens = tokens
        self.fields = dict(distribution=distribution, base_probability=base_probability,
                           events=events, images=images)
        self.manifest_sha256 = "synthetic-Train-cache"
        self.inventory = dict(images=3, valid_regions=len(tokens))

    def batch(self, indices, device):
        return dict(tokens=self.tokens[indices].to(device), **{
            key: value[indices].to(device) for key, value in self.fields.items()})


class RetrievalMiningTests(unittest.TestCase):
    def _fixture(self):
        # Query 0: true Intact, J1 Intact, cross-event Damaged support wins.
        # Query 1: true Intact, J1 Intact, cross-event Intact support wins.
        # Query 2: true Damaged, J1 incorrectly predicts Intact.
        target = torch.eye(4)[torch.tensor([1, 1, 2])]
        base = torch.tensor([[.02, .90, .07, .01], [.02, .90, .07, .01],
                             [.02, .90, .07, .01]])
        store = SyntheticStore(torch.tensor([[1., 0.], [0., 1.], [1., 0.]]),
                               target, base, torch.tensor([0, 0, 0]),
                               torch.tensor([0, 1, 2]))
        bank = dict(tokens=torch.tensor([[1., 0.], [.8, .6], [0., 1.]]),
                    distribution=torch.eye(4)[torch.tensor([2, 2, 1])],
                    events=torch.tensor([0, 1, 1]), images=torch.tensor([9, 10, 11]))
        return store, bank

    def test_original_topk_mining_respects_event_exclusion(self):
        store, bank = self._fixture()
        misleading, missed, stats = mine_reference_risk(
            store, bank, RetrievalProjector(2), torch.device("cpu"), topk=2,
            temperature=.1, batch_size=1)
        self.assertEqual(misleading.tolist(), [True, False, False])
        self.assertEqual(missed.tolist(), [False, False, True])
        self.assertEqual(stats["pure_intact_j1_correct"], 2)
        self.assertEqual(stats["misleading_intact"], 1)
        self.assertEqual(stats["pure_damaged_j1_missed"], 1)
        # If only the same-event most-similar Damaged row remains, no support
        # is available and it must never become a false risk diagnosis.
        same_event = {k: v[:1] for k, v in bank.items()}
        misleading, _, stats = mine_reference_risk(
            store, same_event, RetrievalProjector(2), torch.device("cpu"), topk=8)
        self.assertFalse(misleading.any())
        self.assertEqual(stats["pure_intact_with_support"], 0)
        # One finite candidate and one masked slot must not yield NaNs.
        partly_excluded = {k: v[:2] for k, v in bank.items()}
        misleading, _, stats = mine_reference_risk(
            store, partly_excluded, RetrievalProjector(2), torch.device("cpu"), topk=8)
        self.assertEqual(stats["pure_intact_with_support"], 2)
        self.assertTrue(misleading[0])

    def test_soft_label_vote_is_mining_diagnostic_not_readout(self):
        store, bank = self._fixture()
        _, _, stats = mine_reference_risk(store, bank, RetrievalProjector(2),
                                           torch.device("cpu"), risk_margin=.2)
        self.assertIn("not Reader", stats["interpretation"])
        with self.assertRaises(ValueError):
            mine_reference_risk(store, bank, RetrievalProjector(2),
                                torch.device("cpu"), temperature=0.)
        with self.assertRaises(ValueError):
            mine_reference_risk(store, bank, RetrievalProjector(2),
                                torch.device("cpu"), risk_margin=-.1)

    def test_risk_sampler_balances_both_target_cohorts(self):
        classes = torch.arange(4).repeat_interleave(6)
        target = torch.eye(4)[classes]
        base = .8 * target + .05
        misleading = torch.zeros(24, dtype=torch.bool)
        missed = torch.zeros(24, dtype=torch.bool)
        misleading[6:9] = True
        missed[12:15] = True
        sampler = RetrievalMisleadSampler(target, base, misleading, missed)
        indices, stats = sampler.sample(4, torch.Generator().manual_seed(4), .5)
        self.assertEqual(len(indices), 16)
        for class_id, risk in ((1, misleading), (2, missed)):
            selected = indices[sampler.classes[indices] == class_id]
            self.assertEqual(int(risk[selected].sum()), 2)
            self.assertEqual(stats["by_class"][str(class_id)]["ordinary"], 2)
            self.assertEqual(stats["by_class"][str(class_id)]["target_risk"], 2)
        with self.assertRaisesRegex(ValueError, "lacks an ordinary or target-risk pool"):
            RetrievalMisleadSampler(target, base, torch.zeros_like(misleading), missed)
        with self.assertRaisesRegex(ValueError, "non-Intact"):
            RetrievalMisleadSampler(target, base, misleading | (classes == 0), missed)

    def test_matched_training_updates_only_search_projection(self):
        from .train import _train_arm
        torch.manual_seed(12)
        classes = torch.arange(4).repeat(3)
        target = torch.eye(4)[classes]
        tokens = torch.randn(12, 5)
        events = torch.arange(3).repeat_interleave(4)
        images = torch.arange(12)
        base = .8 * target + .05
        store = SyntheticStore(tokens, target, base, events, images)
        support = dict(tokens=tokens.clone(), distribution=target, events=events, images=images)
        misleading = torch.zeros(12, dtype=torch.bool)
        missed = torch.zeros(12, dtype=torch.bool)
        misleading[1] = True
        missed[2] = True
        samplers = dict(j1_hard=HardQuerySampler(target, base),
                        mislead=RetrievalMisleadSampler(target, base, misleading, missed))
        args = SimpleNamespace(lr=.001, seed=3407, steps=20, per_class=2, hard_fraction=.5,
                               temperature=.1, smoke=True)
        with tempfile.TemporaryDirectory() as temporary:
            for arm, sampler in samplers.items():
                with contextlib.redirect_stdout(io.StringIO()):
                    checkpoint, result = _train_arm(args, arm, store, support, sampler,
                                                     {}, Path(temporary), {}, "source")
                self.assertEqual(checkpoint["format"], "SRAC_MEMORY_METRIC_V2")
                self.assertEqual(checkpoint["steps"], 20)
                self.assertGreater(result["nonzero_gradient_updates"], 0)
                self.assertFalse(torch.equal(checkpoint["model"]["linear.weight"], torch.eye(5)))
        self.assertFalse(tokens.requires_grad)

    def test_checkpoint_loader_rejects_hash_change(self):
        from .train import load_metric_checkpoint
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bda_benchmark.experiments.damageformer_srac_memory_metric_v2.train.artifact_identity",
            return_value={"source_hashes": {"front": "same"}}), patch(
            "bda_benchmark.experiments.damageformer_srac_memory_metric_v2.train.experiment_source_hash",
            return_value="new-source"):
            root = Path(temporary)
            path = root / "metric_mislead_last.pth"
            checkpoint = dict(format="SRAC_MEMORY_METRIC_V2", arm="mislead", complete=True,
                              smoke=True, train_only=True, test_accessed=False, steps=20,
                              source_experiment_sha256="new-source",
                              query_manifest_sha256="Train-cache", source_hashes={"front": "same"},
                              dim=3, model=RetrievalProjector(3).state_dict(),
                              immutable_checks={"source": True})
            torch.save(checkpoint, path)
            other = root / "metric_j1_hard_last.pth"
            torch.save(dict(checkpoint, arm="j1_hard"), other)
            summary = dict(format="SRAC_MEMORY_METRIC_V2", complete=True, smoke=True,
                           train_only=True, test_accessed=False, steps=20,
                           protocol_sha256="E0", source_experiment_sha256="new-source",
                           query_manifest_sha256="Train-cache", source_hashes={"front": "same"},
                           immutable_checks={"source": True},
                           checkpoint_sha256={"mislead": digest(path), "j1_hard": digest(other)})
            write_json(root / "summary.json", summary)
            write_json(root / "manifest.json", dict(format="SRAC_MEMORY_METRIC_V2",
                       complete=True, smoke=True, train_only=True, test_accessed=False,
                       source_experiment_sha256="new-source",
                       query_manifest_sha256="Train-cache", source_hashes={"front": "same"}))
            bundle = dict(protocol_sha256="E0", bank_cpu={"tokens": torch.empty(1, 3)},
                          device=torch.device("cpu"))
            model = load_metric_checkpoint(path, bundle, "mislead", smoke=True)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            summary["checkpoint_sha256"]["mislead"] = "changed"
            write_json(root / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "mismatched or changed"):
                load_metric_checkpoint(path, bundle, "mislead", smoke=True)

    def test_full_smoke_run_writes_matched_hashed_checkpoints(self):
        from . import train
        torch.manual_seed(13)
        classes = torch.arange(4).repeat(3)
        target = torch.eye(4)[classes]
        tokens = torch.randn(12, 5)
        events = torch.arange(3).repeat_interleave(4)
        images = torch.arange(12)
        base = .8 * target + .05
        store = SyntheticStore(tokens, target, base, events, images)
        bank = dict(tokens=tokens.clone(), distribution=target, events=events, images=images)
        bundle = dict(bank_cpu=bank, bank=bank, metric=RetrievalProjector(5), topk=8,
                      runtime_settings={}, protocol_sha256="synthetic-E0",
                      device=torch.device("cpu"))
        misleading = torch.zeros(12, dtype=torch.bool)
        missed = torch.zeros(12, dtype=torch.bool)
        misleading[1], missed[2] = True, True
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store.path = root / "queries.json"
            store.path.write_text("Train only", encoding="utf-8")
            store.manifest_sha256 = digest(store.path)
            args = SimpleNamespace(output=str(root / "fit"), queries=str(store.path),
                                   smoke=True, steps=20, per_class=2, purity=.8,
                                   temperature=.1, risk_temperature=.1, risk_margin=0.,
                                   lr=.001, hard_fraction=.5, hard_quantile=.75,
                                   mine_batch=2, seed=3407, device="cpu")
            with patch.object(train, "load_verified_bundle", return_value=bundle), patch.object(
                    train, "QueryStore", return_value=store), patch.object(
                    train, "mine_reference_risk", return_value=(misleading, missed, {"Train_only": True})), patch.object(
                    train, "verify_unchanged", return_value={"frozen_source": True}), patch.object(
                    train, "artifact_identity", return_value={
                        "source_hashes": {"front": "same"},
                        "protocol_sha256": "synthetic-E0",
                    }), \
                    contextlib.redirect_stdout(io.StringIO()):
                train.run(args)
                summary = train.read_json(root / "fit" / "summary.json")
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["protocol_sha256"], "synthetic-E0")
                self.assertEqual(set(summary["checkpoint_sha256"]), {"j1_hard", "mislead"})
                self.assertTrue(all(summary["immutable_checks"].values()))
                for arm in ("j1_hard", "mislead"):
                    path = root / "fit" / f"metric_{arm}_last.pth"
                    self.assertEqual(summary["checkpoint_sha256"][arm], digest(path))
                    model = train.load_metric_checkpoint(path, bundle, arm, smoke=True)
                    self.assertFalse(any(p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
