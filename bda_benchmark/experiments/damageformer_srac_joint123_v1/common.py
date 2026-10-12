"""Reuse strict loaders unchanged; bind new joint heads to their own code."""
import hashlib
from pathlib import Path

from bda_benchmark.experiments.damageformer_srac_qmetric_v1 import common as inherited
from bda_benchmark.experiments.damageformer_srac_qmetric_v1.common import (
    device_bank, digest, event, lists, load_front, load_metric, read_cache,
    save_json, save_pt, seed_everything,
)


def source_hash():
    checksum = hashlib.sha256(inherited.source_hash().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        checksum.update(path.name.encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()
