"""V2 correction fitting must consume matching, complete V2 diagnostics."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .diagnose import FORMAT as DIAGNOSTIC_FORMAT
from .train_correction import validate_diagnostics


class TrainingProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="srac_v2_training_provenance_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.diagnostics = self.root / "diagnostics"
        self.diagnostics.mkdir()
        key_root = self.root / "keys"
        key_root.mkdir()
        self.bank_path, self.manifest_path = key_root / "paired_bank.pt", key_root / "manifest.json"
        self.bank_path.write_bytes(b"synthetic Train-only paired key artifact")
        self.manifest_path.write_text('{"synthetic":true}\n', encoding="utf-8")
        self.ledger = {path: self.sha256(path) for path in (self.bank_path, self.manifest_path)}
        self.bundle = {"protocol_sha256": "a" * 64}
        self.own_source = "b" * 64
        self.base_report = dict(
            format=DIAGNOSTIC_FORMAT, complete=True, full_validation_evidence=True,
            image_count=450, smoke=False, test_accessed=False,
            protocol_sha256=self.bundle["protocol_sha256"], source_sha256=self.own_source,
            paired_bank_sha256=self.ledger[self.bank_path],
            paired_manifest_sha256=self.ledger[self.manifest_path], paired_bank_smoke=False,
            retrieval_policies=["global", "per_class"], topk=8, temperature=.1,
            reproduction={"passed": True}, e0_reproduction={"passed": True})
        self.base_status = {key: self.base_report[key] for key in (
            "format", "complete", "full_validation_evidence", "image_count", "smoke", "test_accessed")}

    @staticmethod
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_records(self, report=None, status=None):
        records = (("result.json", self.base_report if report is None else report),
                   ("summary.json", self.base_status if status is None else status))
        for name, record in records:
            (self.diagnostics / name).write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

    def validate(self):
        return validate_diagnostics(self.diagnostics, self.bundle, self.ledger,
                                    policy="global", topk=8, temperature=.1,
                                    own_source=self.own_source)

    def test_complete_matching_current_format_returns_both_file_hashes(self):
        self.write_records()
        actual = self.validate()
        expected = {path: self.sha256(path) for path in
                    (self.diagnostics / "result.json", self.diagnostics / "summary.json")}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 2)

    def test_v1_diagnostic_is_rejected_in_either_record(self):
        for record_name in ("report", "status"):
            with self.subTest(record=record_name):
                report, status = copy.deepcopy(self.base_report), copy.deepcopy(self.base_status)
                (report if record_name == "report" else status)["format"] = "SRAC_MEMORY_KEY_DIAGNOSE_V1"
                self.write_records(report, status)
                with self.assertRaisesRegex(ValueError, "completed non-smoke Val450"):
                    self.validate()

    def test_source_protocol_and_each_key_hash_must_match(self):
        for field in ("source_sha256", "protocol_sha256", "paired_bank_sha256", "paired_manifest_sha256"):
            with self.subTest(field=field):
                report = copy.deepcopy(self.base_report)
                report[field] = "c" * 64
                self.write_records(report)
                with self.assertRaisesRegex(ValueError, "source/protocol/keys/retrieval/gates differ"):
                    self.validate()

    def test_incomplete_nonformal_or_failed_reproduction_is_rejected(self):
        bad_fields = {"complete": False, "full_validation_evidence": False,
                      "image_count": 449, "smoke": True, "test_accessed": True}
        for record_name in ("report", "status"):
            for field, value in bad_fields.items():
                with self.subTest(record=record_name, field=field):
                    report, status = copy.deepcopy(self.base_report), copy.deepcopy(self.base_status)
                    (report if record_name == "report" else status)[field] = value
                    self.write_records(report, status)
                    with self.assertRaisesRegex(ValueError, "completed non-smoke Val450"):
                        self.validate()
        for field, value in (("reproduction", {"passed": False}),
                             ("e0_reproduction", {"passed": False}),
                             ("paired_bank_smoke", True)):
            with self.subTest(field=field):
                report = copy.deepcopy(self.base_report)
                report[field] = value
                self.write_records(report)
                with self.assertRaisesRegex(ValueError, "source/protocol/keys/retrieval/gates differ"):
                    self.validate()


if __name__ == "__main__":
    unittest.main()
