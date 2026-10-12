"""A failed, partial, or changed result must not become a formal report."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from .report import read_formal


class ReportContractTests(unittest.TestCase):
    def test_completed_payload_is_hash_bound_and_formal(self):
        result = dict(format="SRAC_REGION_HEAD_ADAPT_EVAL_V1", complete=True,
                      full_validation_evidence=True, image_count=450,
                      immutable={"models": True}, native_a1_replay_ok=True,
                      reproduction={"passed": True}, test_accessed=False)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = json.dumps(result).encode()
            (root / "result.json").write_bytes(payload)
            summary = dict(format=result["format"], complete=True,
                           full_validation_evidence=True, result_file="result.json",
                           result_sha256=hashlib.sha256(payload).hexdigest())
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(read_formal(root), result)
            (root / "result.json").write_bytes(payload + b"\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                read_formal(root)
            result["reproduction"]["passed"] = False
            payload = json.dumps(result).encode()
            (root / "result.json").write_bytes(payload)
            summary["result_sha256"] = hashlib.sha256(payload).hexdigest()
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "1e-5"):
                read_formal(root)


if __name__ == "__main__":
    unittest.main()
