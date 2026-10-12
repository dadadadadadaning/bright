import torch

from bda_benchmark.model.DamageFormer_AffectedOccurrence_V1 import DamageFormerAffectedOccurrenceV1, Stage2OccurrenceEvidence
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.model.safe_affected_plugins import SafeAffectedResidual


def test_occurrence_residual_is_bitwise_parent_at_zero_gain():
    parent = torch.randn(2, 4, 8, 8)
    module = SafeAffectedResidual()
    output, delta = module(parent, torch.randn(2, 1, 8, 8), torch.ones(2, 1, 8, 8), torch.ones(2, 1, 8, 8))
    assert torch.equal(output, parent)
    assert torch.count_nonzero(delta) == 0


def test_occurrence_residual_preserves_severity_difference():
    parent = torch.randn(1, 4, 8, 8)
    module = SafeAffectedResidual(); module.gain.data.fill_(0.5)
    output, _ = module(parent, torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8))
    assert torch.allclose(output[:, 3] - output[:, 2], parent[:, 3] - parent[:, 2])


def test_stage2_head_uses_prediction_only_boundary_and_continuous_quality():
    head = Stage2OccurrenceEvidence(hidden=8, use_boundary=True, use_sar_quality=True)
    logits, diagnostics = head(
        torch.randn(1, 192, 8, 8), torch.randn(1, 192, 8, 8),
        torch.randn(1, 3, 32, 32), torch.rand(1, 1, 8, 8), (8, 8),
    )
    assert logits.shape == (1, 1, 8, 8)
    assert diagnostics["sar_quality"].min() >= 0
    assert diagnostics["sar_quality"].max() <= 1
    assert diagnostics["predicted_boundary"].shape == (1, 1, 8, 8)


def test_wrapper_starts_as_exact_p3_when_shared_state_is_loaded():
    torch.manual_seed(7)
    parent = DamageFormer_S8_MICB_DSMIL(mil_residual_scale=0.0).eval()
    wrapper = DamageFormerAffectedOccurrenceV1("m4_sar_quality").eval()
    wrapper.load_state_dict(parent.state_dict(), strict=False)
    pre, post = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        parent_loc, parent_damage = parent(pre, post)
        wrapper_loc, wrapper_damage = wrapper(pre, post)
    assert torch.equal(wrapper_loc, parent_loc)
    assert torch.equal(wrapper_damage, parent_damage)
