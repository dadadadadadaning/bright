"""Read-only entry gates and isolated head optimization on synthetic tensors."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import JointContextDecoder
from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import RetrievalProjector
from . import precheck
from .joint_runtime import ARMS, decoder_forward, retrieve_pair


class EntryContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_completed_parent_gate_and_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "front").mkdir()
            (root / "train_memory").mkdir()
            for relative in ("front/best.pth", "a1.pth", "train_memory/manifest.json", "train_memory/bank.pt"):
                (root / relative).write_bytes(b"test-artifact")
            base = dict(format=precheck.parent.FRONT_FORMAT, run_complete=True,
                        completed_steps=50000, arguments={"steps": 50000}, step=45000,
                        source_hash="source", a1_sha256="digest", train_sha256="digest",
                        val_sha256="digest", smoke=False, test_accessed=False)
            cases = [({}, True), ({"run_complete": False}, False),
                     ({"completed_steps": 49999}, False), ({"source_hash": "stale"}, False),
                     ({"a1_sha256": "wrong"}, False), ({"val_sha256": "wrong"}, False),
                     ({"smoke": True}, False), ({"test_accessed": True}, False)]
            for changes, success in cases:
                with self.subTest(changes=changes), \
                     mock.patch.object(precheck, "lists", return_value=([], [])), \
                     mock.patch.object(precheck.torch, "load", return_value=dict(base, **changes)), \
                     mock.patch.object(precheck.parent, "source_hash", return_value="source"), \
                     mock.patch.object(precheck, "digest", return_value="digest"), \
                     mock.patch.object(precheck, "read_cache", return_value=(
                         root, {"smoke": False, "shards": [{}] * 2918}, {"tokens": torch.zeros(8, 128)})):
                    if success:
                        _, _, result = precheck.inspect_parent(root, root / "a1.pth", "train_set.txt", "val_set.txt")
                        self.assertEqual(result["train_images"], 2918)
                        self.assertEqual(result["front_step"], 45000)
                    else:
                        with self.assertRaises(ValueError):
                            precheck.inspect_parent(root, root / "a1.pth", "train_set.txt", "val_set.txt")

    def test_three_heads_update_without_changing_frozen_evidence(self):
        torch.manual_seed(137)
        a1 = nn.Conv2d(256, 4, 1).eval().requires_grad_(False)
        template = JointContextDecoder(a1, dim=8, num_events=2, slots=2)
        decoders = {arm: copy.deepcopy(template) for arm in ARMS}
        bank = dict(tokens=torch.randn(8, 8), distribution=torch.eye(4).repeat(2, 1),
                    events=torch.ones(8, dtype=torch.long), images=torch.arange(8),
                    geometry=torch.rand(8, 5), disasters=torch.zeros(8, dtype=torch.long))
        metric = RetrievalProjector(8).eval().requires_grad_(False)
        pixels = torch.randn(1, 256, 2, 2)
        result = dict(tokens=torch.randn(1, 2, 8), q=torch.tensor([[[0.8, 0.7, 0.2, 0.1], [0.2, 0.3, 0.8, 0.9]]]),
                      ids=torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]), shape=(2, 2))
        snapshots = {key: value.clone() for key, value in result.items() if torch.is_tensor(value)}
        before = a1.weight.detach().clone()
        _, cosine, _, learned = retrieve_pair(result["tokens"][0], bank, 0, -1, 2, metric)
        for arm, model in decoders.items():
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            old = model.classifier.pixel.weight.detach().clone()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                logits = decoder_forward(model, arm, pixels, result, cosine, learned)["logits"]
                loss = F.cross_entropy(logits, torch.tensor([[[0, 1], [2, 3]]]))
                loss.backward()
                optimizer.step()
            self.assertFalse(torch.equal(old, model.classifier.pixel.weight))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.reader.parameters()))
        torch.testing.assert_close(a1.weight, before, atol=0, rtol=0)
        for key, tensor in snapshots.items():
            torch.testing.assert_close(result[key], tensor, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in a1.parameters()))
        self.assertTrue(all(p.grad is None for p in metric.parameters()))


if __name__ == "__main__":
    unittest.main()
