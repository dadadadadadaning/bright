import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import recheck_replay


class RecheckEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.file = Path(self.temp.name) / "evidence.pt"
        self.record = dict(format="SRAC_MEMORY_KEY_REPLAY_EVIDENCE_V2", image="sample_0",
                           replayed_fp32=torch.tensor([[.0312654]], dtype=torch.float32),
                           archived_tokens=torch.tensor([[.03125]], dtype=torch.float16),
                           protocol_sha256="e0", original_bank_sha256="original",
                           report={"passed": True})

    def invoke(self, sha=None):
        argv = ["recheck_replay", "--input", str(self.file)]
        if sha is not None:
            argv += ["--sha256", sha]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            recheck_replay.main()
        return json.loads(output.getvalue())

    def test_cpu_evidence_load_checks_actual_quantization_report(self):
        torch.save(self.record, self.file)
        sha = hashlib.sha256(self.file.read_bytes()).hexdigest()
        result = self.invoke(sha)
        self.assertTrue(result["report"]["passed"])
        self.assertFalse(result["report"]["legacy_report"]["passed"])
        self.assertEqual(result["evidence_sha256"], sha)
        self.assertEqual(result["original_bank_sha256"], "original")

    def test_checksum_mismatch_blocks_read(self):
        torch.save(self.record, self.file)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.invoke("wrong")

    def test_fake_saved_pass_does_not_hide_real_drift(self):
        self.record["replayed_fp32"] += .1
        torch.save(self.record, self.file)
        with self.assertRaises(SystemExit) as caught:
            self.invoke()
        self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
