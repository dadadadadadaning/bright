"""Fast artifact/pipeline contracts; no BRIGHT data or checkpoint is required."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from . import common, train_metric
from .metric import RetrievalProjector
from .model import FRONT_FORMAT


class _FakeAnnealedFront(nn.Module):
    """Small stand-in that exposes only the checkpoint-loader contract."""

    def __init__(self, dim: int = 3):
        super().__init__()
        self.base = nn.Linear(dim, dim)
        self.marker = None
        self._beta = 1.0

    @property
    def current_beta(self) -> float:
        return self._beta

    def load_adapter_state_dict(self, state):
        self.marker = state["marker"].detach().clone()
        self._beta = float(state["beta"])


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(211)
        torch.set_num_threads(2)

    @staticmethod
    def _metric_bank(dim: int = 8):
        # Every class has a same-class positive in another event and negatives
        # in another event, so even a two-update smoke exercises real gradients.
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        return {
            "tokens": torch.randn(len(labels), dim),
            "distribution": torch.eye(4)[labels],
            "events": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            "images": torch.arange(len(labels)),
        }

    def test_metric_smoke_writes_train_last_and_reloads_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            front = root / "front.pth"
            a1 = root / "a1.pth"
            cache_root = root / "cache"
            output = root / "metric"
            front.write_bytes(b"frozen-front")
            a1.write_bytes(b"audited-a1")
            cache_root.mkdir()
            (cache_root / "manifest.json").write_text("{}", encoding="utf-8")
            cache = {"smoke": True, "bank_sha256": "bank-sha"}
            args = SimpleNamespace(
                checkpoint=str(front),
                a1_checkpoint=str(a1),
                cache=str(cache_root),
                output=str(output),
                steps=2,
                per_class=1,
                purity=0.8,
                temperature=0.1,
                lr=1e-3,
                seed=3407,
                device="cpu",
                smoke=True,
            )
            with (
                mock.patch.object(
                    train_metric,
                    "read_cache",
                    return_value=(cache_root, cache, self._metric_bank()),
                ),
                mock.patch.object(
                    train_metric,
                    "load_front",
                    return_value=(nn.Linear(1, 1), {"smoke": True}),
                ),
                mock.patch.object(train_metric, "source_hash", return_value="source-v1"),
                mock.patch("builtins.print"),
            ):
                train_metric.run(args)

            checkpoint_path = output / "metric_last.pth"
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["step"], 2)
            self.assertEqual(payload["selection"], "fixed-budget last; no Val checkpoint or hyperparameter selection")
            self.assertFalse(
                torch.equal(payload["model"]["linear.weight"], torch.eye(8))
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["complete"])
            self.assertGreater(summary["nonzero_gradient_updates"], 0)

            with mock.patch.object(common, "source_hash", return_value="source-v1"):
                restored, restored_payload = common.load_metric(
                    checkpoint_path, cache, front, torch.device("cpu")
                )
            self.assertFalse(restored.training)
            self.assertFalse(any(parameter.requires_grad for parameter in restored.parameters()))
            self.assertEqual(restored_payload["step"], 2)
            torch.testing.assert_close(
                restored.state_dict()["linear.weight"],
                payload["model"]["linear.weight"],
                atol=0,
                rtol=0,
            )

    def test_metric_loader_rejects_incomplete_and_stale_artifacts(self):
        projector = RetrievalProjector(4)
        base = {
            "format": common.METRIC_FORMAT,
            "complete": True,
            "source_hash": "current-source",
            "front_sha256": "front-sha",
            "bank_sha256": "bank-sha",
            "smoke": False,
            "train_only": True,
            "test_accessed": False,
            "step": 2000,
            "arguments": {"steps": 2000},
            "additional_optimizer_updates": 2000,
            "dim": 4,
            "model": projector.state_dict(),
        }
        cases = {
            "incomplete": dict(base, complete=False),
            "stale_source": dict(base, source_hash="old-source"),
            "other_front": dict(base, front_sha256="other-front"),
            "other_bank": dict(base, bank_sha256="other-bank"),
            "smoke_mismatch": dict(base, smoke=True),
            "not_train_only": dict(base, train_only=False),
            "test_accessed": dict(base, test_accessed=True),
            "partial_step": dict(base, step=1999),
            "declared_budget_mismatch": dict(base, additional_optimizer_updates=1999),
            "nonpositive_budget": dict(
                base,
                step=0,
                arguments={"steps": 0},
                additional_optimizer_updates=0,
            ),
            "wrong_formal_budget": dict(
                base,
                step=20,
                arguments={"steps": 20},
                additional_optimizer_updates=20,
            ),
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                with (
                    mock.patch.object(common.torch, "load", return_value=payload),
                    mock.patch.object(common, "source_hash", return_value="current-source"),
                    mock.patch.object(common, "digest", return_value="front-sha"),
                ):
                    with self.assertRaises(ValueError):
                        common.load_metric(
                            "metric.pth",
                            {"bank_sha256": "bank-sha", "smoke": False},
                            "front.pth",
                            torch.device("cpu"),
                        )

    def test_metric_loader_preserves_global_rng_stream(self):
        projector = RetrievalProjector(4)
        payload = {
            "format": common.METRIC_FORMAT,
            "complete": True,
            "source_hash": "current-source",
            "front_sha256": "front-sha",
            "bank_sha256": "bank-sha",
            "smoke": False,
            "train_only": True,
            "test_accessed": False,
            "step": 2000,
            "arguments": {"steps": 2000},
            "additional_optimizer_updates": 2000,
            "dim": 4,
            "model": projector.state_dict(),
        }
        torch.manual_seed(9182)
        before = torch.random.get_rng_state().clone()
        with (
            mock.patch.object(common.torch, "load", return_value=payload),
            mock.patch.object(common, "source_hash", return_value="current-source"),
            mock.patch.object(common, "digest", return_value="front-sha"),
        ):
            loaded, _ = common.load_metric(
                "metric.pth",
                {"bank_sha256": "bank-sha", "smoke": False},
                "front.pth",
                torch.device("cpu"),
            )
        self.assertIsInstance(loaded, RetrievalProjector)
        torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)

    def test_q_front_loader_restores_adapter_then_freezes_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            a1 = Path(temporary) / "a1.pth"
            a1.write_bytes(b"a1")
            adapter = {"marker": torch.tensor([3.0, 4.0]), "beta": 2.0}
            payload = {
                "format": FRONT_FORMAT,
                "run_complete": True,
                "completed_steps": 7,
                "arguments": {"steps": 7},
                "source_hash": "current-source",
                "a1_checkpoint": str(a1),
                "a1_sha256": "a1-sha",
                "config": {"dim": 3},
                "adapter": adapter,
                "q_beta_current": 2.0,
            }
            with (
                mock.patch.object(common.torch, "load", return_value=payload),
                mock.patch.object(common, "source_hash", return_value="current-source"),
                mock.patch.object(common, "digest", return_value="a1-sha"),
                mock.patch.object(common, "load_a1_strict") as load_a1,
                mock.patch(
                    "bda_benchmark.experiments.damageformer_srac_qmetric_v1.model.AnnealedFront",
                    _FakeAnnealedFront,
                ),
            ):
                restored, returned = common.load_front(
                    "front.pth", torch.device("cpu"), str(a1)
                )
            load_a1.assert_called_once()
            self.assertIs(returned, payload)
            torch.testing.assert_close(restored.marker, adapter["marker"], atol=0, rtol=0)
            self.assertEqual(restored.current_beta, 2.0)
            self.assertFalse(restored.base.training)
            self.assertFalse(any(parameter.requires_grad for parameter in restored.base.parameters()))

    def test_q_front_loader_rejects_incomplete_checkpoint_before_model_build(self):
        payload = {
            "format": FRONT_FORMAT,
            "run_complete": False,
            "completed_steps": 6,
            "arguments": {"steps": 7},
        }
        with mock.patch.object(common.torch, "load", return_value=payload):
            with self.assertRaises(ValueError):
                common.load_front("front.pth", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
