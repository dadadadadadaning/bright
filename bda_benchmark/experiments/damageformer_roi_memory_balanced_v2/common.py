"""V2 provenance layered on the immutable ROI_FOLLOWUP_V1 token contract."""
import hashlib
from pathlib import Path

from bda_benchmark.experiments.damageformer_roi_followup_v1 import common as v1
from bda_benchmark.experiments.damageformer_region_memory_v2 import common as residual_v2


# These aliases are intentional.  A V1 bank must be validated against the
# exact V1 source identity that created it; V2 never weakens or rewrites it.
add_arguments = v1.add_arguments
validate_training_args = v1.validate_training_args
load_source = v1.load_source
source_identity = v1.source_identity
assert_source = v1.assert_source
capture = v1.capture
routes = v1.routes
loader = v1.loader
parse_batch = v1.parse_batch
metrics = v1.metrics
cm_update = v1.cm_update
seed_everything = v1.seed_everything
new_output = v1.new_output
save_json = v1.save_json
save_pt = v1.save_pt
digest = v1.digest
event = v1.event
objective = v1.objective
state_digest = v1.state_digest
append_json = v1.append_json


def experiment_hash():
    """Hash V2 plus the complete V1 implementation chain it consumes."""
    h = hashlib.sha256((v1.experiment_hash() + residual_v2.implementation_hash()).encode())
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.sh'):
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def assert_v2_immutable(bundle, expected_hash):
    assert_source(bundle)
    if experiment_hash() != expected_hash:
        raise RuntimeError('Balanced-Memory V2 implementation changed during the experiment')
