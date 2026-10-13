import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import shared


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.original = dict(images=torch.tensor([0, 0, 1, 1]),
                             events=torch.tensor([0, 0, 1, 1]),
                             regions=torch.tensor([0, 1, 0, 1]), event_names=["a", "b"],
                             tokens=torch.full((4, 128), 99., dtype=torch.float16),
                             distribution=torch.tensor([[0., 0., 0., 1.]]).repeat(4, 1))
        self.bank = dict(t128=torch.ones(4, 128), f256=torch.ones(4, 256),
                         distribution=torch.eye(4), mass=torch.ones(4), label_mass=torch.ones(4),
                         member_ids=torch.arange(4), original_rows=torch.arange(4),
                         active=torch.ones(4, dtype=torch.bool), eligible=torch.ones(4, dtype=torch.bool),
                         **{key: copy.deepcopy(self.original[key])
                            for key in ("images", "events", "regions", "event_names")})
        self.manifest = dict(smoke=False, support_count=4,
                             membership_sha256=shared.membership_digest(self.bank))

    def test_fresh_values_are_independent_of_historical_tokens_and_labels(self):
        self.assertTrue(shared.validate_member_identity(self.bank, self.manifest, self.original))
        self.original["tokens"].fill_(float("nan"))
        self.original["distribution"].fill_(-12345)
        self.assertTrue(shared.validate_member_identity(self.bank, self.manifest, self.original))

    def test_member_plan_uses_train_identities_not_historical_values(self):
        plan = shared.member_plan(self.original, ["a_0", "b_0"], ["c_0"], 1)
        self.assertEqual(plan["images"], [0])
        self.assertEqual(plan["names"], ["a_0"])
        self.assertEqual(plan["rows"].tolist(), [0, 1])

    def test_formal_bank_cannot_omit_fixed_support_members(self):
        for key, value in list(self.bank.items()):
            if torch.is_tensor(value):
                self.bank[key] = value[:2]
        self.manifest.update(support_count=2, membership_sha256=shared.membership_digest(self.bank))
        with self.assertRaisesRegex(ValueError, "every original"):
            shared.validate_member_identity(self.bank, self.manifest, self.original)
        self.manifest["smoke"] = True
        self.assertTrue(shared.validate_member_identity(self.bank, self.manifest, self.original))

    def test_metadata_change_and_member_reordering_are_rejected(self):
        self.bank["regions"][0] = 5
        with self.assertRaisesRegex(ValueError, "identity changed"):
            shared.validate_member_identity(self.bank, self.manifest, self.original)
        self.bank["regions"][0] = 0
        self.bank["original_rows"] = torch.tensor([1, 0, 2, 3])
        with self.assertRaisesRegex(ValueError, "ordered"):
            shared.validate_member_identity(self.bank, self.manifest, self.original)

    def test_fp16_new_keys_and_noncanonical_ids_are_rejected(self):
        self.bank["t128"] = self.bank["t128"].half()
        with self.assertRaisesRegex(ValueError, "float32|FP32"):
            shared.validate_member_identity(self.bank, self.manifest, self.original)
        self.bank["t128"] = self.bank["t128"].float()
        self.bank["member_ids"] = torch.tensor([1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "canonical"):
            shared.validate_member_identity(self.bank, self.manifest, self.original)

    def test_joint_zero_key_is_retained_with_ineligible_mask(self):
        self.bank["f256"][0].zero_()
        self.bank["active"][0] = self.bank["eligible"][0] = False
        self.assertTrue(shared.validate_member_identity(self.bank, self.manifest, self.original))
        self.bank["eligible"][0] = True
        with self.assertRaisesRegex(ValueError, "eligibility differs"):
            shared.validate_member_identity(self.bank, self.manifest, self.original)

    def test_valid_label_mass_cannot_exceed_Q_mass(self):
        self.bank["label_mass"][0] = 2.
        with self.assertRaisesRegex(ValueError, "exceeds"):
            shared.validate_member_identity(self.bank, self.manifest, self.original)

    def test_duplicate_original_image_region_and_event_mismatch_rejected(self):
        self.original["regions"][1] = 0
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            shared.member_plan(self.original, ["a_0", "b_0"], ["c_0"])
        self.original["regions"][1] = 1
        self.original["events"][0] = 1
        with self.assertRaisesRegex(ValueError, "event/image"):
            shared.member_plan(self.original, ["a_0", "b_0"], ["c_0"])

    def test_split_overlap_duplicates_and_size_are_rejected(self):
        for train, val in ((["a_0"], ["a_0"]), (["a_0", "a_0"], ["b_0"]),
                           (["../a"], ["b_0"])):
            with self.assertRaises(ValueError):
                shared.validate_splits(train, val)
        with self.assertRaisesRegex(ValueError, "Train2918"):
            shared.validate_splits(["a_0"], ["b_0"], 2918, 450)

    def test_membership_digest_ignores_keys_but_tracks_identity(self):
        before = shared.membership_digest(self.bank)
        self.bank["t128"].mul_(2)
        self.assertEqual(before, shared.membership_digest(self.bank))
        self.bank["regions"][0] = 4
        self.assertNotEqual(before, shared.membership_digest(self.bank))

    def test_inventory_reports_fixed_member_coverage_without_changing_bank(self):
        self.bank["eligible"][0] = False
        before = copy.deepcopy(self.bank)
        result = shared.label_inventory(self.bank)
        self.assertEqual(result["all"]["dominant_counts"], [1, 1, 1, 1])
        self.assertEqual(result["eligible"]["dominant_counts"], [0, 1, 1, 1])
        self.assertEqual(result["eligible"]["soft_mass"], [0., 1., 1., 1.])
        self.assertEqual(result["by_event"]["a"]["all"]["dominant_counts"], [1, 1, 0, 0])
        for key, value in self.bank.items():
            if torch.is_tensor(value):
                self.assertTrue(torch.equal(value, before[key]))


class JournalTests(unittest.TestCase):
    def setUp(self):
        IdentityTests.setUp(self)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = dict(train_names=["a_0", "b_0"], val_names=["c_0"])
        self.frames = []
        for index in (0, 1):
            members = (self.bank["images"] == index).nonzero().flatten()
            self.frames.append(dict(index=index, frame_id=index, image=self.bundle["train_names"][index],
                train_image_id=index, extraction_forward_count=1, forward_count=index + 1,
                rows=len(members), member_ids=self.bank["member_ids"][members].tolist(),
                original_rows=self.bank["original_rows"][members].tolist(),
                region_ids=self.bank["regions"][members].tolist(),
                native_a1=dict(passed=True), q_check=dict(passed=True),
                label_check=dict(passed=True, reference_dtype="torch.float64", reference_device="cpu",
                                 atol=2e-5, rtol=2e-5),
                paired_sha256={key: shared.tensor_digest(self.bank[key][members])
                               for key in ("t128", "f256", "distribution", "mass", "label_mass")},
                joint_ineligible_rows=0))
        self.manifest.update(selected_source_images=["a_0", "b_0"], selected_train_image_ids=[0, 1])

    def save(self):
        path = self.root / "images.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in self.frames))
        self.manifest["images_journal_sha256"] = shared.digest(path)

    def test_journal_binds_current_values_and_exact_one_forward(self):
        self.save()
        self.assertTrue(shared.verify_bank_journal(self.root, self.bank, self.manifest, self.bundle))
        self.frames[0]["extraction_forward_count"] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, "one-forward"):
            shared.verify_bank_journal(self.root, self.bank, self.manifest, self.bundle)

    def test_saved_label_or_membership_change_is_detected_by_journal(self):
        self.save()
        self.bank["distribution"][0] = torch.tensor([.5, .5, 0., 0.])
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            shared.verify_bank_journal(self.root, self.bank, self.manifest, self.bundle)

    def test_failed_reference_cannot_become_valid_by_rehashing_journal(self):
        self.frames[0]["label_check"]["passed"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "failed current"):
            shared.verify_bank_journal(self.root, self.bank, self.manifest, self.bundle)

    def test_changed_journal_or_reference_settings_are_rejected(self):
        self.save()
        path = self.root / "images.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            shared.verify_bank_journal(self.root, self.bank, self.manifest, self.bundle)
        self.frames[0]["label_check"]["atol"] = 1.
        self.save()
        with self.assertRaisesRegex(ValueError, "settings"):
            shared.verify_bank_journal(self.root, self.bank, self.manifest, self.bundle)


class LedgerTests(unittest.TestCase):
    def test_input_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.bin"
            path.write_bytes(b"original")
            ledger = {path: shared.digest(path)}
            with patch("bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils.verify_unchanged",
                       return_value={"frozen_models": True}), patch.object(shared, "source_hash", return_value="code"):
                self.assertTrue(shared.verify_all({}, "code", ledger)["rebuild_input_files"])
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "immutability"):
                    shared.verify_all({}, "code", ledger)


if __name__ == "__main__":
    unittest.main()
