import json
import tempfile
import unittest
from pathlib import Path

from bda_benchmark.experiments.spi_rgid_semantic_v1.scripts.verify_contract import (
    verify_provenance,
)


class ProvenanceTests(unittest.TestCase):
    def _write(self, payload: dict) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        with handle:
            json.dump(payload, handle)
        return Path(handle.name)

    def test_train_only_pseudo_provenance_passes(self):
        path = self._write(
            {
                "official_instances_accessed": False,
                "standard_ml_val_accessed": False,
                "original_standard_test_accessed": False,
                "calibration_split": "train_only",
                "p3_sha256": "abc",
            }
        )
        verify_provenance(path)

    def test_validation_calibration_is_rejected(self):
        path = self._write(
            {
                "official_instances_accessed": False,
                "standard_ml_val_accessed": True,
                "original_standard_test_accessed": False,
                "calibration_split": "validation",
                "p3_sha256": "abc",
            }
        )
        with self.assertRaises(ValueError):
            verify_provenance(path)


if __name__ == "__main__":
    unittest.main()
