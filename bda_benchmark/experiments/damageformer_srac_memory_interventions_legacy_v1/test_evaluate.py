"""CPU contracts for read-only support interventions."""
import unittest
import json
import tempfile
from pathlib import Path

import torch

from .evaluate import intervention_supports
from .launch import arguments_from_reference
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import ContextMemoryReader


class SupportInterventionTests(unittest.TestCase):
    def setUp(self):
        self.bank = {
            "tokens": torch.arange(40, dtype=torch.float32).reshape(5, 8),
            "distribution": torch.eye(4)[torch.tensor([0, 1, 2, 3, 1])],
            "geometry": torch.zeros(5, 5),
            "events": torch.arange(5),
            "images": torch.arange(5),
            "disasters": torch.zeros(5, dtype=torch.long),
        }
        self.selected = torch.tensor([[1, 2], [1, -1]], dtype=torch.long)

    def test_random_shuffle_off_preserve_source_and_exclusion(self):
        from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import gather_support

        original = gather_support(self.selected, self.bank)
        snapshot = {name: value.clone() for name, value in self.bank.items()}
        generator = torch.Generator().manual_seed(7)
        variants, counts = intervention_supports(
            self.selected, original, self.bank, 0, -1, generator)

        self.assertTrue(all(torch.equal(self.bank[name], value) for name, value in snapshot.items()))
        self.assertTrue(torch.equal(variants["random"][2], original[2]))
        self.assertFalse(variants["off"][2].any())
        self.assertTrue(torch.equal(variants["shuffle_labels"][0], original[0]))
        self.assertTrue(torch.equal(variants["shuffle_labels"][2], original[2]))
        self.assertTrue(torch.equal(variants["shuffle_labels"][3]["events"], original[3]["events"]))
        self.assertEqual(counts["random_valid_slots"], 3)
        self.assertTrue((variants["random"][3]["events"][original[2]] != 0).all())
        original_labels = original[1][original[2]].sort(dim=0).values
        shuffled_labels = variants["shuffle_labels"][1][original[2]].sort(dim=0).values
        self.assertTrue(torch.equal(original_labels, shuffled_labels))

    def test_off_is_own_reader_fallback(self):
        torch.manual_seed(17)
        reader = ContextMemoryReader(dim=8, num_events=5, heads=2, slots=2).eval()
        query = torch.randn(2, 8)
        support = torch.randn(2, 2, 8)
        distribution = torch.zeros(2, 2, 4)
        distribution[..., 2] = 1
        invalid = torch.zeros(2, 2, dtype=torch.bool)
        with torch.no_grad():
            off = reader(query, support, distribution, invalid, mode="memory")
            own_fallback = reader(query, mode="nomemory")
        self.assertTrue(torch.equal(off, own_fallback))

    def test_launcher_preserves_e0_source_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol").mkdir()
            original = dict(joint_run="/original/joint", a1_checkpoint="/original/a1.pth",
                            dataset="/original/dataset", train_list="/original/train.txt",
                            val_list="/original/val.txt", device="cuda:0", workers=4,
                            seed=3407, profile="legacy", pool_mode="native", smoke=False,
                            matmul_tf32=None, cudnn_tf32=None, deterministic=None)
            (root / "protocol" / "protocol.json").write_text(json.dumps(dict(
                format="SRAC_HARDQUERY_PROTOCOL_V1", complete=True, scope="all",
                arguments=original)), encoding="utf-8")
            args = arguments_from_reference(root, root / "intervention_output", 93)
            self.assertEqual(args.joint_run, original["joint_run"])
            self.assertEqual(args.a1_checkpoint, original["a1_checkpoint"])
            self.assertEqual(args.pool_mode, "native")
            self.assertEqual(args.intervention_seed, 93)


if __name__ == "__main__":
    unittest.main()
