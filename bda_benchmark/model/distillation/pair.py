"""Teacher/student wrapper with correct train/eval and gradient semantics."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn

from ..registry import ModelSpec, create_model, get_model_spec
from .adapter import DistillationOutput, forward_for_distillation


class DistillationPair(nn.Module):
    """Hold a frozen teacher and trainable student under one stable interface."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        teacher_spec: ModelSpec | str,
        student_spec: ModelSpec | str,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.teacher_spec = (
            get_model_spec(teacher_spec)
            if isinstance(teacher_spec, str)
            else teacher_spec
        )
        self.student_spec = (
            get_model_spec(student_spec)
            if isinstance(student_spec, str)
            else student_spec
        )
        self.freeze_teacher()

    @classmethod
    def from_registry(
        cls,
        teacher_name: str,
        student_name: str,
        *,
        teacher_kwargs: Optional[Mapping[str, Any]] = None,
        student_kwargs: Optional[Mapping[str, Any]] = None,
        offline: bool = False,
    ) -> "DistillationPair":
        teacher = create_model(
            teacher_name, offline=offline, **dict(teacher_kwargs or {})
        )
        student = create_model(
            student_name, offline=offline, **dict(student_kwargs or {})
        )
        return cls(teacher, student, teacher_name, student_name)

    def freeze_teacher(self) -> None:
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True) -> "DistillationPair":
        super().train(mode)
        self.teacher.eval()
        self.student.train(mode)
        return self

    def forward(
        self,
        pre_image: Tensor,
        post_image: Tensor,
        *,
        teacher_forward_kwargs: Optional[Mapping[str, Any]] = None,
        student_forward_kwargs: Optional[Mapping[str, Any]] = None,
        request_aux: bool = True,
    ) -> Mapping[str, DistillationOutput]:
        with torch.no_grad():
            teacher_output = forward_for_distillation(
                self.teacher,
                self.teacher_spec,
                pre_image,
                post_image,
                request_aux=request_aux,
                forward_kwargs=teacher_forward_kwargs,
            )
        student_output = forward_for_distillation(
            self.student,
            self.student_spec,
            pre_image,
            post_image,
            request_aux=request_aux,
            forward_kwargs=student_forward_kwargs,
        )
        return {"teacher": teacher_output, "student": student_output}

