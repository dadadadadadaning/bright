import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .shared import validate_member_identity, verify_all


class MemberIdentityTest(unittest.TestCase):
    def setUp(self):
        self.original = dict(tokens=torch.randn(4, 128).half(),
                             distribution=torch.eye(4), events=torch.tensor([0, 0, 1, 1]),
                             images=torch.tensor([0, 1, 2, 3]), regions=torch.tensor([1, 2, 3, 4]),
                             event_names=["a", "b"])
        self.bank = {"t128": self.original["tokens"].clone(), "f256": torch.randn(4, 256).half(),
                     "original_rows": torch.arange(4), "event_names": ["a", "b"]}
        self.bank.update({k: self.original[k].clone() for k in ("distribution", "events", "images", "regions")})
        self.manifest = {"smoke": False, "support_count": 4}

    def test_full_bank_exact_original_order(self):
        validate_member_identity(self.bank, self.manifest, self.original)

    def test_reordered_member_rejected(self):
        self.bank["original_rows"] = torch.tensor([1, 0, 2, 3])
        with self.assertRaisesRegex(ValueError, "original order"):
            validate_member_identity(self.bank, self.manifest, self.original)

    def test_soft_label_change_rejected(self):
        self.bank["distribution"][0] = torch.tensor([.9, .1, 0., 0.])
        with self.assertRaisesRegex(ValueError, "distribution"):
            validate_member_identity(self.bank, self.manifest, self.original)

    def test_feature_change_rejected(self):
        self.bank["t128"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "t128"):
            validate_member_identity(self.bank, self.manifest, self.original)

    def test_original_identity_field_change_rejected(self):
        self.bank["regions"][0] += 1
        with self.assertRaisesRegex(ValueError, "regions"):
            validate_member_identity(self.bank, self.manifest, self.original)

    def test_dtype_change_rejected(self):
        self.bank["t128"] = self.bank["t128"].float()
        with self.assertRaisesRegex(ValueError, "t128"):
            validate_member_identity(self.bank, self.manifest, self.original)

    def test_subset_requires_smoke(self):
        for key in ("t128", "f256", "distribution", "events", "images", "regions", "original_rows"):
            self.bank[key] = self.bank[key][:2]
        self.manifest["support_count"] = 2
        with self.assertRaisesRegex(ValueError, "every original support"):
            validate_member_identity(self.bank, self.manifest, self.original)
        self.manifest["smoke"] = True
        validate_member_identity(self.bank, self.manifest, self.original)

    def test_duplicate_member_rejected(self):
        self.bank["original_rows"] = torch.tensor([0, 0, 2, 3])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_member_identity(self.bank, self.manifest, self.original)


class LedgerTest(unittest.TestCase):
    def test_new_input_mutation_rejected(self):
        from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bank.pt"
            path.write_bytes(b"original")
            ledger = {path: digest(path)}
            with patch("bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils.verify_unchanged",
                       return_value={"frozen_models": True}), patch(__package__ + ".shared.source_hash", return_value="code"):
                self.assertTrue(verify_all({}, "code", ledger)["paired_input_files"])
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "immutability"):
                    verify_all({}, "code", ledger)


if __name__ == "__main__":
    unittest.main()
