import json
import tempfile
import unittest
from pathlib import Path

from bda_benchmark.script.standard_ML.spi_stage0.annotations import (
    load_stage0_instance_records,
)


class Stage0AnnotationAdapterTests(unittest.TestCase):
    def test_single_image_file_uses_authoritative_split_sample_id(self):
        payload = {
            "images": [{"id": 7, "height": 4, "width": 5}],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "intact"},
                {"id": 2, "name": "damaged"},
                {"id": 3, "name": "destroyed"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "libya-flood_001_instance_damage.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            records, manifest = load_stage0_instance_records(
                ["libya-flood_001"], per_image_json_dir=str(directory)
            )
        self.assertEqual(list(records), ["libya-flood_001"])
        self.assertEqual(records["libya-flood_001"]["image"]["id"], 7)
        self.assertEqual(manifest["adapted_missing_file_name"], 1)

    def test_multiple_images_without_identity_remain_rejected(self):
        payload = {
            "images": [
                {"id": 7, "height": 4, "width": 5},
                {"id": 8, "height": 4, "width": 5},
            ],
            "annotations": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "libya-flood_001_instance_damage.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(KeyError):
                load_stage0_instance_records(
                    ["libya-flood_001"], per_image_json_dir=str(directory)
                )


if __name__ == "__main__":
    unittest.main()
