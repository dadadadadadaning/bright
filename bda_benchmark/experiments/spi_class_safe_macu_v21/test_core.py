import torch

from .core import ARMS, apply_arm_policy, arm_acceptance, threshold_pairs


def test_three_nontrivial_arms_are_distinct():
    metric = torch.tensor([0.8, 0.8, 0.2, 0.2])
    harm = torch.tensor([0.2, 0.8, 0.2, 0.8])
    assert arm_acceptance("b1_benefit_only", metric, harm, 0.5, 0.5).tolist() == [True, True, False, False]
    assert arm_acceptance("b2_harm_only", metric, harm, 0.5, 0.5).tolist() == [True, False, True, False]
    assert arm_acceptance("b3_dual", metric, harm, 0.5, 0.5).tolist() == [True, False, False, False]


def test_threshold_grid_is_preregistered_and_includes_single_head_controls():
    assert len(threshold_pairs("b1_benefit_only")) == 9
    assert len(threshold_pairs("b2_harm_only")) == 9
    assert len(threshold_pairs("b3_dual")) == 81
    assert all(harm == 1.0 for _, harm in threshold_pairs("b1_benefit_only"))
    assert all(metric == 0.0 for metric, _ in threshold_pairs("b2_harm_only"))


def test_fixed_s2_control_accepts_every_cluster():
    metric = torch.rand(10)
    harm = torch.rand(10)
    policy = {"metric_threshold": 0.0, "harm_threshold": 1.0}
    assert bool(apply_arm_policy("b0_fixed_s2", metric, harm, policy).all())
    assert ARMS[2] == "b2_harm_only"


def test_unknown_arm_is_rejected():
    try:
        arm_acceptance("unknown", torch.zeros(1), torch.zeros(1), 0.5, 0.5)
    except ValueError as error:
        assert "unknown V2.1 arm" in str(error)
    else:
        raise AssertionError("unknown arm was accepted")
