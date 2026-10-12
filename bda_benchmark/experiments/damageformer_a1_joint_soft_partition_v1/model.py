"""A1 + cross-modal damage-aware soft partition, trained jointly."""
from __future__ import annotations
from typing import Dict
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.model import DamageFormerA1CrossModalSuperpixel

FIXED_VARIANT = "c5_aligned_assoc"

class DamageFormerA1JointSoftPartition(DamageFormerA1CrossModalSuperpixel):
    def __init__(self, region_stride: int = 8) -> None:
        super().__init__(variant=FIXED_VARIANT, region_stride=region_stride)

    def enable_joint_training(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def set_joint_training(self) -> None:
        self.train()

    def checkpoint_contract(self) -> Dict[str, object]:
        contract = dict(super().checkpoint_contract())
        contract.update({
            "protocol": "DAMAGEFORMER_A1_JOINT_SOFT_PARTITION_V1",
            "training_policy": "joint_finetune_A1_and_cmsp",
            "initialization": "A1_best_checkpoint_plus_new_cmsp",
            "a1_frozen": False,
            "cmsp_trainable": True,
            "optimizer_steps": 20000,
            "building_reasoning": False,
            "graph": False,
            "mil": False,
            "memory": False,
            "test_accessed": False,
        })
        return contract

__all__ = ["DamageFormerA1JointSoftPartition", "FIXED_VARIANT"]
