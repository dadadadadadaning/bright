"""Critical-instance Memory Rescue (CMR) research utilities.

The package is deliberately isolated from the accepted P3 model.  Importing it
does not alter P3 modules, checkpoints, logits, or inference behaviour.
"""

from .metric import (
    CMRResidualMetric,
    event_aware_supcon_loss,
    selective_damaged_positive_mask,
    selective_damaged_supcon_loss,
)
from .pooling import CMR_KEY_VARIANTS, extract_cmr_local_tokens, extract_cmr_pooling_keys
from .multiprototype import spherical_kmeans
from .dynamic_queue import CMREventClassQueue, ema_update
from .pairwise import (
    CMRBoundedPairwiseScorer,
    CMRCandidateConditionalPairwiseScorer,
    class_balanced_hard_top1_loss,
    multi_positive_candidate_loss,
)

__all__ = [
    "CMR_KEY_VARIANTS",
    "CMRResidualMetric",
    "event_aware_supcon_loss",
    "selective_damaged_positive_mask",
    "selective_damaged_supcon_loss",
    "extract_cmr_pooling_keys",
    "extract_cmr_local_tokens",
    "spherical_kmeans",
    "CMREventClassQueue",
    "ema_update",
    "CMRBoundedPairwiseScorer",
    "CMRCandidateConditionalPairwiseScorer",
    "class_balanced_hard_top1_loss",
    "multi_positive_candidate_loss",
]
