"""CPU contracts for the checkpoint-500 loss fork; no BRIGHT/CUDA evidence."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import continuation as subject
import evaluation
import study


class StaticBindingContracts(unittest.TestCase):
    """Exercise provenance checks using synthetic files, never model loading."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root, self.tool = self.base / "old_run", self.base / "original_tool"
        self.train = self.root / "train"
        (self.train / "checkpoints").mkdir(parents=True)
        (self.train / "val_step_000500").mkdir()
        self.tool.mkdir()
        (self.tool / "run.py").write_text("# Synthetic source binding fixture\n")
        checksum = hashlib.sha256()
        checksum.update(b"run.py")
        checksum.update((self.tool / "run.py").read_bytes())
        self.source = checksum.hexdigest()
        self.dump(self.tool / "SHA256SUMS.json", {"run.py": study.digest(self.tool / "run.py")})
        self.dump(self.train / "summary.json", {"complete": True, "full_validation_evidence": True, "smoke": False})
        self.dump(self.root / "workflow_state.json", {"complete": True, "full_validation_evidence": True})
        self.names = ["train_%04d" % i for i in range(2918)]
        self.dump(self.train / "manifest.json", {"source_hash": self.source, "train_names": self.names,
                  "val_names": ["val_%04d" % i for i in range(450)]})
        checkpoint = self.train / "checkpoints/checkpoint_000500.pt"
        checkpoint.write_bytes(b"synthetic artifact; never deserialized")
        self.dump(checkpoint.with_suffix(".pt.json"), {"step": 500, "mode": "train", "smoke": False,
                  "training_complete": False, "sha256": study.digest(checkpoint), "source_hash": self.source})
        self.dump(self.train / "val_step_000500/result.json", {"image_count": 450, "execution_complete": True,
                  "metrics": {"grid_probq": {}, "b": {}, "d": {}}})
        self.rows = [{"step": i + 1, "image": self.names[i % 2918], "epoch": i // 2918}
                     for i in range(5000)]
        self.write_journal()
        self.args = SimpleNamespace(train_root=str(self.root), tool_dir=str(self.tool))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def dump(path, data):
        path.write_text(json.dumps(data))

    def write_journal(self):
        (self.train / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def validate(self):
        with patch.object(study, "EXPECTED_SOURCE", self.source):
            return study.validate_static(self.args)

    def test_validates_binding_and_detects_original_artifact_changes(self):
        record = self.validate()
        self.assertEqual(len(record["rows"]), 5000)
        study.check_old_files(record)
        with (self.train / "summary.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            study.check_old_files(record)

    def test_incomplete_original_is_not_promoted_to_complete(self):
        self.dump(self.train / "summary.json", {"complete": False, "full_validation_evidence": True, "smoke": False})
        with self.assertRaisesRegex(ValueError, "must be complete"):
            self.validate()

    def test_checkpoint_file_must_match_original_manifest(self):
        (self.train / "checkpoints/checkpoint_000500.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "binding failed"):
            self.validate()

    def test_old_source_extra_python_is_not_silently_imported(self):
        (self.tool / "model.py").write_text("# unbound file\n")
        with self.assertRaisesRegex(ValueError, "Unexpected Python"):
            self.validate()

    def test_original_order_requires_contiguous_unique_first1500(self):
        self.rows[999]["image"] = self.rows[998]["image"]
        self.write_journal()
        with self.assertRaisesRegex(ValueError, "epoch-zero"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
