"""Provenance helpers for the isolated retrieval audit."""
import hashlib
from pathlib import Path

from bda_benchmark.experiments.damageformer_roi_memory_selective_v1 import common as parent


add_arguments = parent.add_arguments
load_source = parent.load_source
source_identity = parent.source_identity
capture = parent.capture
routes = parent.routes
loader = parent.loader
parse_batch = parent.parse_batch
seed_everything = parent.seed_everything
new_output = parent.new_output
save_json = parent.save_json
digest = parent.digest
event = parent.event


def parent_hash():
    return parent.experiment_hash()


def implementation_hash():
    value = hashlib.sha256(parent_hash().encode())
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.sh'):
            value.update(path.name.encode())
            value.update(path.read_bytes())
    return value.hexdigest()


def assert_immutable(bundle, expected_hash, expected_parent_hash):
    parent.assert_selective_immutable(
        bundle, expected_parent_hash, parent.parent_experiment_hash())
    if implementation_hash() != expected_hash:
        raise RuntimeError('Retrieval-audit implementation changed during the run')

