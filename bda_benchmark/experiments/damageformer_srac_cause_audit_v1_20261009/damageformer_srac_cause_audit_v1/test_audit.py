import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import SRACTransformerFront
from . import audit
from .support import digest, verify_splits


class AuditIntegration(unittest.TestCase):
    def test_split_content_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"train_set.txt"
            path.write_text("first\n")
            payload = dict(train_sha256=digest(path), val_sha256=digest(path))
            verify_splits(payload, path, path)
            path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "Split differs"):
                verify_splits(payload, path, path)

    def test_reproduction_has_predeclared_tolerance_and_requires_450(self):
        payload = dict(valid_miou=.61, load_report=dict(expected_valid_miou=.62))
        result = dict(image_count=450, metrics=dict(a1=dict(valid_miou=.62), region=dict(valid_miou=.61)))
        self.assertTrue(audit.check_reproduction(result, payload, False)["passed"])
        result["metrics"]["region"]["valid_miou"] = .61002
        self.assertFalse(audit.check_reproduction(result, payload, False)["passed"])
        self.assertIsNone(audit.check_reproduction(result, payload, True)["passed"])

    def test_two_image_end_to_end_saves_json_and_preserves_failed_evidence(self):
        torch.set_num_threads(2)
        torch.manual_seed(35)
        model = SRACTransformerFront(dim=16, decoder_width=16, stride=4, iterations=1).requires_grad_(False).eval()
        pre, post = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        gt = torch.randint(0, 4, (1, 64, 64))
        batch = (pre, post, None, gt, ["haiti-earthquake_0001"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshot = root/"snapshot"; snapshot.mkdir()
            checkpoint = snapshot/"front_snapshot.pth"; checkpoint.write_bytes(b"fixture")
            (snapshot/"snapshot.json").write_text("{}")
            train = root/"train_set.txt"; train.write_text("train\n")
            val = root/"val_set.txt"; val.write_text("val\n")
            payload = dict(step=1000, valid_miou=.123, load_report=dict(expected_valid_miou=.62),
                           smoke=False, a1_sha256="a1", source_hash="source", train_sha256=digest(train), val_sha256=digest(val))
            args = argparse.Namespace(checkpoint=str(checkpoint), a1_checkpoint="unused.pth", dataset="unused",
                                      train_list=str(train), val_list=str(val), output=str(root/"audit_smoke"),
                                      device="cpu", workers=0, seed=3407, power=2., threshold=.5, smoke=True)
            with mock.patch.object(audit, "lists", return_value=(["train"]*2918, ["val"]*450)), \
                    mock.patch.object(audit, "loader", return_value=[batch, batch]), \
                    mock.patch.object(audit, "parse_batch", side_effect=lambda b, d: b), \
                    mock.patch.object(audit, "load_snapshot", return_value=(model, payload)), \
                    contextlib.redirect_stdout(io.StringIO()):
                audit.run(args)
                result = json.loads((root/"audit_smoke"/"audit.json").read_text())
                self.assertTrue(result["complete"])
                self.assertTrue(result["weights_and_buffers_unchanged"])
                self.assertEqual(result["image_count"], 2)
                self.assertFalse(result["test_accessed"])
                self.assertEqual(len(result["metrics"]), 10)
                args.smoke = False; args.output = str(root/"failed_full")
                with self.assertRaisesRegex(RuntimeError, "Evidence saved"):
                    audit.run(args)
                failed = json.loads((root/"failed_full"/"audit.json").read_text())
                self.assertFalse(failed["complete"])
                self.assertIn("paired", failed)
                self.assertFalse(failed["reproduction"]["passed"])


if __name__ == "__main__":
    unittest.main()
