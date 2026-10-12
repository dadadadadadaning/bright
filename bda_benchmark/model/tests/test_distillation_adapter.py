import torch
from torch import nn

from bda_benchmark.model.distillation.adapter import (
    forward_for_distillation,
    normalize_model_output,
)
from bda_benchmark.model.distillation.pair import DistillationPair


class _PairedModel(nn.Module):
    def forward(self, pre, post, return_aux=False):
        loc = pre[:, :2]
        dam = torch.cat([pre, post[:, :1]], dim=1)
        if return_aux:
            return loc, dam, {"final_feat_damage": pre + post}
        return loc, dam


class _ConcatModel(nn.Module):
    def forward(self, value):
        return value[:, :4]


class _TrainableConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(6, 4, kernel_size=1)

    def forward(self, value):
        return self.projection(value)


def test_tuple_output_is_normalized_with_features():
    pre = torch.randn(2, 3, 8, 8)
    post = torch.randn(2, 3, 8, 8)
    result = forward_for_distillation(
        _PairedModel(), "damageformer_s1c", pre, post
    )
    assert result.localization_logits.shape[1] == 2
    assert result.damage_logits.shape[1] == 4
    assert "final_feat_damage" in result.feature_maps


def test_concatenated_input_model():
    pre = torch.randn(2, 3, 8, 8)
    post = torch.randn(2, 3, 8, 8)
    result = forward_for_distillation(_ConcatModel(), "unet", pre, post)
    assert result.damage_logits.shape == (2, 4, 8, 8)


def test_mapping_output_is_normalized():
    damage = torch.randn(1, 4, 4, 4)
    feature = torch.randn(1, 16, 4, 4)
    result = normalize_model_output(
        {"output_dam": damage, "final_feat_damage": feature}
    )
    assert result.damage_logits is damage
    assert result.feature_maps["final_feat_damage"] is feature


def test_pair_keeps_teacher_frozen_and_student_trainable():
    pair = DistillationPair(
        _TrainableConcatModel(),
        _TrainableConcatModel(),
        "unet",
        "unet",
    )
    pair.train()
    assert not pair.teacher.training
    assert pair.student.training
    assert not any(parameter.requires_grad for parameter in pair.teacher.parameters())
    assert all(parameter.requires_grad for parameter in pair.student.parameters())

    pre = torch.randn(2, 3, 8, 8)
    post = torch.randn(2, 3, 8, 8)
    outputs = pair(pre, post)
    assert not outputs["teacher"].damage_logits.requires_grad
    assert outputs["student"].damage_logits.requires_grad
