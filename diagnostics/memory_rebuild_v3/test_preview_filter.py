"""Synthetic CPU-only tests for read-only support-filter preview."""
import contextlib
import hashlib
import io
from pathlib import Path
import unittest
from unittest import mock
import torch

import PREVIEW_MEMORY_SUPPORT_FILTER as preview
from test_summary import SummaryTests


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.records = SummaryTests()
        self.records.setUp()
        self.addCleanup(self.records.doCleanups)
        self.root = self.records.root

    def bank(self):
        y = torch.tensor([[.5, .25, .25, 0], [.51, .49, 0, 0], [.9, 0, .1, 0],
                          [.4, .2, .2, .2], [.2, 0, .8, 0], [0, 0, 0, 1],
                          [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=torch.float32)
        n = len(y)
        bank = dict(t128=torch.ones(n, 128), f256=torch.ones(n, 256), distribution=y,
                    label_mass=torch.arange(1, n + 1, dtype=torch.float32),
                    events=torch.tensor([0, 0, 1, 1, 0, 1, 2, 1, 2]), event_names=["alpha", "beta", "gamma"],
                    images=torch.arange(n), regions=torch.arange(n),
                    original_rows=torch.arange(n) + 200, member_ids=torch.arange(n) + 100,
                    eligible=torch.tensor([True, True, True, True, True, True, False, True, True]))
        bank["f256"][5].zero_()
        return bank

    def fixture(self):
        _, summary, manifest, protocol, _, result = self.records.fixture()
        bank = self.bank()
        path = self.root / "keys/paired_bank.pt"
        torch.save(bank, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        for record in (summary, manifest, protocol):
            record["support_count"] = len(bank["distribution"])
            record["paired_bank_sha256"] = sha
        for record in (summary, manifest):
            record["image_count"] = record["extraction_forward_count"] = len(bank["distribution"])
        result["paired_bank_sha256"] = sha
        ph = self.records.write("keys/protocol.json", protocol)
        for record in (summary, manifest, result):
            record["rebuild_protocol_sha256"] = ph
        self.records.write("keys/summary.json", summary)
        result["paired_manifest_sha256"] = self.records.write("keys/manifest.json", manifest)
        self.records.write("diagnose/result.json", result)
        return bank, summary, manifest, protocol, result

    def test_half_foreground_boundary_passes_without_extra_purity_gate(self):
        base, selected = preview.select(self.bank(), .5, 0)
        self.assertEqual(base.nonzero().flatten().tolist(), [0, 1, 2, 3, 4, 7, 8])
        self.assertEqual(selected.nonzero().flatten().tolist(), [0, 3, 4, 7])
        self.assertTrue(bool(selected[0]))
        self.assertFalse(bool(selected[1]))

    def test_purity_is_conditioned_on_foreground_and_low_foreground_cannot_pass(self):
        _, selected = preview.select(self.bank(), .5, .8)
        self.assertEqual(selected.nonzero().flatten().tolist(), [4, 7])
        self.assertFalse(bool(selected[2]))  # 90% background, but foreground purity=1.
        self.assertFalse(bool(selected[3]))  # 60% foreground, foreground purity=1/3.

    def test_pure_background_has_zero_foreground_purity(self):
        _, selected = preview.select(self.bank(), 0, .5)
        self.assertFalse(bool(selected[8]))
        _, neutral = preview.select(self.bank(), 0, 0)
        self.assertTrue(bool(neutral[8]))

    def test_both_key_norms_and_original_eligibility_are_required(self):
        bank = self.bank()
        base, _ = preview.select(bank, 0, 0)
        self.assertFalse(bool(base[5]))
        self.assertFalse(bool(base[6]))
        bank["t128"][4].zero_()
        base, _ = preview.select(bank, 0, 0)
        self.assertFalse(bool(base[4]))

    def test_empty_class_and_empty_mask_stats_keep_correct_denominators(self):
        bank = self.bank()
        base, selected = preview.select(bank)
        row = preview.stats(bank, selected)
        self.assertEqual(row["regions"], 4)
        self.assertEqual(row["dominant_counts"], [2, 0, 1, 1])
        self.assertAlmostEqual(row["label_quality_share"][0], 3.1 / 18, places=6)
        self.assertAlmostEqual(row["label_quality_share"][2], 5.05 / 18, places=6)
        self.assertAlmostEqual(row["soft_share"][0], 1.1 / 4, places=6)
        empty = preview.stats(bank, torch.zeros_like(base))
        self.assertEqual(empty["dominant_counts"], [0, 0, 0, 0])
        self.assertEqual(empty["label_quality_share"], [None] * 4)

    def test_cross_event_counts_not_unfiltered_global_counts(self):
        bank = self.bank()
        _, selected = preview.select(bank)
        alpha_other = preview.stats(bank, selected & (bank["events"] != 0))
        self.assertEqual(alpha_other["regions"], 2)
        self.assertEqual(alpha_other["dominant_counts"], [1, 0, 0, 1])
        output = io.StringIO()
        base, selected = preview.select(bank)
        with contextlib.redirect_stdout(output):
            preview.report(bank, base, selected, .5, 0)
        self.assertIn("alpha |", output.getvalue())
        self.assertIn("不足64", output.getvalue())

    def test_original_row_and_member_ids_unchanged_and_order_retained(self):
        bank = self.bank()
        before = {key: value.clone() for key, value in bank.items() if torch.is_tensor(value)}
        base, selected = preview.select(bank)
        with contextlib.redirect_stdout(io.StringIO()):
            preview.report(bank, base, selected, .5, 0)
        self.assertEqual(bank["original_rows"][selected].tolist(), [200, 203, 204, 207])
        self.assertEqual(bank["member_ids"][selected].tolist(), [100, 103, 104, 107])
        for key, tensor in before.items():
            self.assertTrue(torch.equal(bank[key], tensor), key)

    def test_invalid_thresholds_rejected(self):
        for threshold in (True, -.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                preview.select(self.bank(), threshold, 0)

    def test_fp16_or_nonfinite_or_bad_labels_rejected(self):
        for field in ("fp16", "nonfinite", "negative_label"):
            bank = self.bank()
            if field == "fp16":
                bank["t128"] = bank["t128"].half()
            elif field == "nonfinite":
                bank["f256"][0, 0] = float("nan")
            else:
                bank["distribution"][0, 0] = -1
            with self.assertRaises(ValueError):
                preview.select(bank)

    def test_checksum_mismatch_refused_before_torch_load(self):
        self.fixture()
        with (self.root / "keys/paired_bank.pt").open("ab") as stream:
            stream.write(b"changed")
        with mock.patch.object(torch, "load") as load:
            with self.assertRaisesRegex(ValueError, "SHA256 错误"):
                preview.load_bank(self.root)
            load.assert_not_called()

    def test_protocol_or_manifest_json_binding_mismatch_refused(self):
        for name in ("protocol", "manifest"):
            self.fixture()
            obj = self.records.read("keys/" + name + ".json")
            obj["changed"] = 1
            self.records.write("keys/" + name + ".json", obj)
            with self.assertRaisesRegex(ValueError, "哈希绑定"):
                preview.load_bank(self.root)

    def test_missing_source_hash_not_accepted_by_none_equality(self):
        _, summary, manifest, protocol, result = self.fixture()
        for record in (manifest, protocol, result):
            record.pop("source_sha256", None)
        ph = self.records.write("keys/protocol.json", protocol)
        for record in (summary, manifest, result):
            record["rebuild_protocol_sha256"] = ph
        self.records.write("keys/summary.json", summary)
        result["paired_manifest_sha256"] = self.records.write("keys/manifest.json", manifest)
        self.records.write("diagnose/result.json", result)
        with self.assertRaisesRegex(ValueError, "格式无效"):
            preview.load_bank(self.root)

    def test_loaded_file_preview_cpu_only_read_only_and_no_external_data_reads(self):
        self.fixture()
        before = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        real_open, accessed = Path.open, []
        def read_only(path, mode="r", *args, **kwargs):
            self.assertEqual(mode, "rb")
            self.assertTrue(path.suffix == ".json" or path.name == "paired_bank.pt")
            accessed.append(path.name)
            return real_open(path, mode, *args, **kwargs)
        with mock.patch.object(Path, "open", read_only), mock.patch.object(torch.cuda, "is_available") as cuda:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(preview.main(["--root", str(self.root)]), 0)
            cuda.assert_not_called()
        self.assertEqual(set(accessed), {"summary.json", "manifest.json", "protocol.json", "result.json", "paired_bank.pt"})
        after = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_load_bank_keeps_cpu_tensor_schema(self):
        self.fixture()
        bank = preview.load_bank(self.root)
        self.assertTrue(all(value.device.type == "cpu" for value in bank.values() if torch.is_tensor(value)))
        self.assertEqual(len(bank["distribution"]), 9)

    def test_k64_capacity_boundary_is_descriptive_not_a_filter(self):
        n = 128
        bank = dict(t128=torch.ones(n, 128), f256=torch.ones(n, 256), distribution=torch.tensor([[0., 1., 0., 0.]] * n),
                    label_mass=torch.ones(n), eligible=torch.ones(n, dtype=torch.bool),
                    events=torch.tensor([0] * 64 + [1] * 64), event_names=["a", "b"],
                    **{field: torch.arange(n) for field in ("images", "regions", "original_rows", "member_ids")})
        base, selected = preview.select(bank)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            preview.report(bank, base, selected, .5, 0)
        self.assertEqual(output.getvalue().count("可用 K64"), 2)
        bank["distribution"][64] = torch.tensor([1., 0., 0., 0.])
        base, selected = preview.select(bank)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            preview.report(bank, base, selected, .5, 0)
        self.assertIn("不足64", output.getvalue())
        self.assertEqual(int(selected.sum()), 127)


if __name__ == "__main__":
    unittest.main()
