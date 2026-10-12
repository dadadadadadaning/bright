"""Stage-0-only adapter for official BRIGHT instance annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import file_sha256
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import (
    index_coco_annotations,
    load_official_instance_records,
)


def load_stage0_instance_records(
    sample_ids: Sequence[str],
    *,
    unified_coco_json: str = "",
    per_image_json_dir: str = "",
) -> tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    """Load Stage-0 annotations without weakening unified-COCO identity checks.

    Some official per-image exports omit ``images[0].file_name`` because the
    sample identity is already encoded in the JSON filename.  For exactly one
    image, Stage-0 injects that authoritative filename before using the shared
    strict COCO parser.  Unified files remain unchanged and strict.
    """

    if bool(unified_coco_json) == bool(per_image_json_dir):
        raise ValueError(
            "provide exactly one of unified_coco_json or per_image_json_dir"
        )
    if unified_coco_json:
        return load_official_instance_records(
            sample_ids, unified_coco_json=unified_coco_json
        )

    requested = list(dict.fromkeys(str(value) for value in sample_ids))
    directory = Path(per_image_json_dir)
    selected: Dict[str, Dict[str, object]] = {}
    digest_rows = []
    missing_paths = []
    adapted_without_file_name = 0
    for sample in requested:
        path = directory / f"{sample}_instance_damage.json"
        if not path.is_file():
            missing_paths.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        images = payload.get("images") if isinstance(payload, Mapping) else None
        if (
            isinstance(images, list)
            and len(images) == 1
            and isinstance(images[0], Mapping)
            and "file_name" not in images[0]
        ):
            payload = dict(payload)
            image = dict(images[0])
            image["file_name"] = f"{sample}_post_disaster.tif"
            payload["images"] = [image]
            adapted_without_file_name += 1
        local_index = index_coco_annotations(payload)
        if sample in local_index:
            record = local_index[sample]
        elif len(local_index) == 1:
            record = next(iter(local_index.values()))
        else:
            raise RuntimeError(
                f"cannot resolve sample {sample!r} inside official JSON {path}"
            )
        selected[sample] = record
        digest_rows.append(f"{sample}\t{file_sha256(path)}")
    if missing_paths:
        raise RuntimeError(
            f"missing {len(missing_paths)} official per-image instance JSONs; "
            f"examples={missing_paths[:5]}"
        )
    manifest_sha256 = hashlib.sha256(
        ("\n".join(digest_rows) + "\n").encode("utf-8")
    ).hexdigest()
    return selected, {
        "source": "official_per_image_coco_jsons",
        "path": str(directory.resolve()),
        "manifest_sha256": manifest_sha256,
        "requested_samples": len(requested),
        "adapted_missing_file_name": adapted_without_file_name,
    }
