"""CPU checks for V3 identity-only selection and current-Q parent extraction."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

try:
    from . import fresh_parent_bank as parent
    from .common import digest, write_json
except ImportError:
    import fresh_parent_bank as parent
    from common import digest, write_json


class TinyFront(nn.Module):
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
        self.invalid_for_offset = None

    def forward(self, pre, _post):
        self.calls += 1
        old = self.base.clf_2(pre)
        tokens = (torch.arange(128).float()[None, None] / 32
                  + torch.arange(1, 4).float()[None, :, None] / 4 + pre[0, 0, 0, 0])
        valid = torch.ones(1, 3, dtype=torch.bool)
        if self.invalid_for_offset is not None and float(pre[0, 0, 0, 0]) == self.invalid_for_offset:
            valid[0, 1] = False
        return dict(q=self.q, ids=self.ids, count=3, shape=(2, 2), old_pixel=old,
                    tokens=tokens, valid_regions=valid,
                    mass=torch.tensor([[1.25, 1.5, 1.25]]))


class FreshParentBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = Path(__file__).resolve().parents[2] / "bright"
        self.front = TinyFront().eval().requires_grad_(False)
        # The first four source images share one event; smoke must reach image 4.
        self.names = [f"bata_train_{i}" for i in range(4)] + ["alpha_train_0"]
        self.val_names = ["heldout_val_0"]
        self.dataset = self.root / "dataset"
        self.batches = {}
        self.image_hashes = {}
        from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import sample_paths
        for i, name in enumerate(self.names):
            for kind, path in sample_paths(self.dataset, name).items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((name + kind).encode())
                self.image_hashes[str(path)] = digest(path)
            feature = (torch.arange(256).float().view(1, 256, 1, 1) / 4
                       + torch.tensor([[0., 4.], [8., 12.]]).view(1, 1, 2, 2) + i * 32)
            target = torch.tensor([[[0, 1], [2, 3]]])
            self.batches[name] = (feature, torch.zeros(1, 1, 2, 2),
                                  torch.zeros(1, 2, 2), target, [name])
        self.input_file = self.root / "q_source.json"
        self.input_file.write_text("source", encoding="utf-8")
        self.bundle = dict(front=self.front, args=argparse.Namespace(dataset=str(self.dataset), pool_mode="native"),
                           train_names=self.names, val_names=self.val_names, device=torch.device("cpu"),
                           original_front_sha256="a" * 64, a1_sha256="b" * 64,
                           source_protocol_sha256="c" * 64, source_checkpoint_sha256="d" * 64,
                           source_provenance={"historical_e0_verified": False},
                           input_paths={"source": self.input_file},
                           input_fingerprints={"source": digest(self.input_file)})
        self.archive = self.root / "archive" / "keys"
        self.archive.mkdir(parents=True)
        self._make_archive()

    def _make_archive(self):
        # Deliberately poison the archived values: the fresh bank must ignore them.
        images = torch.tensor([1, 0, 1, 0, 2, 3, 4], dtype=torch.long)
        regions = torch.tensor([2, 1, 0, 2, 1, 2, 0], dtype=torch.long)
        events = (images == 4).long()
        bank = dict(original_rows=torch.arange(len(images), dtype=torch.long),
                    member_ids=torch.arange(len(images), dtype=torch.long), images=images,
                    events=events, regions=regions,
                    event_names=["bata_train", "alpha_train"],
                    t128=torch.full((len(images), 128), float("nan")),
                    f256=torch.full((len(images), 256), float("nan")),
                    distribution=torch.full((len(images), 4), -1.0))
        torch.save(bank, self.archive / "paired_bank.pt")
        journal = []
        for index, image_id in enumerate(images.unique(sorted=True).tolist()):
            rows = (images == image_id).nonzero().flatten()
            journal.append(dict(index=index, frame_id=index, image=self.names[image_id],
                                train_image_id=image_id, rows=len(rows), forward_count=index + 1,
                                extraction_forward_count=1, member_ids=bank["member_ids"][rows].tolist(),
                                original_rows=bank["original_rows"][rows].tolist(),
                                region_ids=regions[rows].tolist(),
                                native_a1={"passed": True}, q_check={"passed": True},
                                label_check={"passed": True}))
        (self.archive / "images.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in journal), encoding="utf-8")
        split = parent._split_identity(self.bundle)
        source_hashes = dict(front_sha256=self.bundle["original_front_sha256"],
                             a1_sha256=self.bundle["a1_sha256"])
        member_sha = parent.membership_sha(bank)
        bank_sha = digest(self.archive / "paired_bank.pt")
        protocol = dict(format="SRAC_MEMORY_REBUILD_PROTOCOL_V3", split_identity=split,
                        source_hashes=source_hashes, paired_bank_sha256=bank_sha,
                        membership_sha256=member_sha, support_count=len(images))
        write_json(self.archive / "protocol.json", protocol)
        shared = dict(format=parent.ARCHIVE_FORMAT, complete=True, smoke=False,
                      paired_bank_sha256=bank_sha, membership_sha256=member_sha,
                      support_count=len(images))
        manifest = dict(**shared, train_only=True, test_accessed=False, training=False,
                        split_identity=split, source_hashes=source_hashes,
                        rebuild_protocol_sha256=digest(self.archive / "protocol.json"),
                        images_journal_sha256=digest(self.archive / "images.jsonl"),
                        selected_train_image_ids=list(range(5)), selected_source_images=self.names,
                        source_image_fingerprints=self.image_hashes, image_count=5)
        write_json(self.archive / "manifest.json", manifest)
        write_json(self.archive / "summary.json", shared)

    def _run(self, name="fresh", smoke=False):
        audit, native, pool, region, transformer, paired = parent._repo_dependencies(self.repo)
        stub = SimpleNamespace(FrozenFrontCapture=audit.FrozenFrontCapture,
                               full_image_loader=lambda _dataset, names, augment, workers, seed:
                                   [self.batches[name] for name in names],
                               parse_batch=lambda batch, _device: batch)
        with patch.object(parent, "_repo_dependencies",
                          return_value=(stub, native, pool, region, transformer, paired)):
            return parent.build_bank(self.bundle, self.archive, self.root / name,
                                     self.repo, workers=0, smoke=smoke)

    def test_new_bank_uses_only_archive_identity_and_current_q_values(self):
        result = self._run()
        self.assertTrue(result["complete"])
        self.assertEqual(self.front.calls, 5)
        self.assertEqual(result["image_count"], 5)
        bank, manifest = parent.load_bank(self.root / "fresh", self.bundle)
        self.assertEqual(bank["t128"].dtype, torch.float32)
        self.assertEqual(bank["f256"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(bank["t128"]).all())
        self.assertTrue(torch.isfinite(bank["f256"]).all())
        self.assertTrue(torch.allclose(bank["distribution"].sum(-1), torch.ones(7)))
        self.assertEqual(bank["regions"].tolist(), [2, 1, 0, 2, 1, 2, 0])
        self.assertEqual(manifest["archived_membership_sha256"],
                         json.loads((self.archive / "manifest.json").read_text())["membership_sha256"])
        self.assertEqual(len(self.front._forward_hooks), 0)
        self.assertEqual(len(self.front.base.clf_2._forward_pre_hooks), 0)

    def test_smoke_uses_four_images_and_formal_reader_rejects_it(self):
        result = self._run("smoke", smoke=True)
        self.assertEqual(result["image_count"], 4)
        with self.assertRaisesRegex(ValueError, "smoke"):
            parent.load_bank(self.root / "smoke", self.bundle)
        bank, _ = parent.load_bank(self.root / "smoke", self.bundle, allow_smoke=True)
        self.assertEqual(len(bank["t128"]), 6)
        self.assertEqual(bank["member_ids"].tolist(), list(range(6)))
        manifest = json.loads((self.root / "smoke" / "manifest.json").read_text())
        self.assertEqual(manifest["attempted_train_image_ids"], [0, 4, 1, 2])
        self.assertEqual(manifest["support_coverage"]["eligible_event_count"], 2)

    def test_different_initial_front_or_split_is_rejected(self):
        self.bundle["original_front_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "initial front"):
            self._run("bad_front")
        self.bundle["original_front_sha256"] = "a" * 64
        self.bundle["train_names"] = list(reversed(self.names))
        with self.assertRaisesRegex(ValueError, "split"):
            self._run("bad_split")

    def test_archive_member_journal_mismatch_is_rejected(self):
        path = self.archive / "images.jsonl"
        frames = [json.loads(line) for line in path.read_text().splitlines()]
        frames[0]["region_ids"][0] = 99
        path.write_text("".join(json.dumps(row) + "\n" for row in frames))
        manifest = json.loads((self.archive / "manifest.json").read_text())
        manifest["images_journal_sha256"] = digest(path)
        write_json(self.archive / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "journal member"):
            self._run("bad_journal")

    def test_saved_bank_detects_changed_train_content(self):
        self._run()
        path = self.dataset / "target" / f"{self.names[0]}_building_damage.tif"
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Train image changed"):
            parent.load_bank(self.root / "fresh", self.bundle)

    def test_invalid_current_region_is_filtered_with_explicit_renumber_audit(self):
        self.front.invalid_for_offset = 0.0
        with patch.object(parent, "MAX_DROP_FRACTION", 0.2):
            self._run("filtered")
            bank, manifest = parent.load_bank(self.root / "filtered", self.bundle)
        self.assertEqual(len(bank["t128"]), 6)
        self.assertEqual(bank["member_ids"].tolist(), list(range(6)))
        self.assertEqual(manifest["original_selected_count"], 7)
        self.assertEqual(manifest["dropped_invalid_count"], 1)
        self.assertEqual(manifest["dropped_members"][0]["reason"], "current_region_invalid")
        self.assertEqual(manifest["dropped_members"][0]["archived_member_id"], 1)
        self.assertEqual(manifest["archived_member_ids_in_bank_order"], [0, 2, 3, 4, 5, 6])
        self.assertEqual(manifest["member_id_renumber_map"][1],
                         {"archived_member_id": 2, "new_member_id": 1})

    def test_material_invalid_current_region_loss_fails_with_audit(self):
        self.front.invalid_for_offset = 0.0
        with self.assertRaisesRegex(ValueError, "Material current-Q member loss"):
            self._run("material_loss")
        manifest = json.loads((self.root / "material_loss" / "manifest.json").read_text())
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["dropped_invalid_count"], 1)
        self.assertEqual(manifest["dropped_members"][0]["reason"], "current_region_invalid")

    def test_smoke_allows_absent_rare_labels_but_formal_requires_them(self):
        for name, batch in list(self.batches.items()):
            self.batches[name] = (*batch[:3], torch.zeros_like(batch[3]), batch[4])
        smoke = self._run("rare_smoke", smoke=True)
        self.assertTrue(smoke["complete"])
        with self.assertRaisesRegex(ValueError, "damaged/destroyed"):
            self._run("rare_formal")


if __name__ == "__main__":
    unittest.main()
