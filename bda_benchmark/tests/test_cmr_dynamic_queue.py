# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from model.cmr.dynamic_queue import CMREventClassQueue, ema_update


class CMRDynamicQueueTests(unittest.TestCase):
    def test_fifo_capacity_and_quality_filter(self):
        queue = CMREventClassQueue(2, 2, capacity_per_cell=2)
        feature = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        written = queue.enqueue(
            feature, torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long),
            quality_mask=torch.tensor([True, False, True]),
        )
        self.assertEqual(written, 2)
        self.assertEqual(int(queue.valid.sum()), 2)
        self.assertEqual(int(queue.total_seen[0, 0]), 2)

    def test_retrieval_excludes_same_event_and_caps_event(self):
        queue = CMREventClassQueue(2, 3, capacity_per_cell=4)
        feature = torch.tensor([[1., 0.], [.9, .1], [.8, .2], [.7, .3], [.6, .4], [.5, .5]])
        queue.enqueue(
            feature,
            torch.ones(6, dtype=torch.long),
            torch.tensor([0, 0, 1, 1, 2, 2]),
        )
        result = queue.retrieve_cross_event(
            torch.tensor([[1., 0.]]), torch.tensor([0]),
            class_index=1, topk=4, maximum_per_event=2,
        )
        self.assertFalse(bool((result["events"] == 0).any()))
        self.assertEqual(result["events"].tolist()[0].count(1), 2)
        self.assertEqual(result["events"].tolist()[0].count(2), 2)

    def test_entries_are_detached_from_student_graph(self):
        queue = CMREventClassQueue(2, 2, capacity_per_cell=2)
        feature = torch.randn(2, 2, requires_grad=True)
        queue.enqueue(feature, torch.tensor([0, 1]), torch.tensor([0, 1]))
        self.assertFalse(queue.features.requires_grad)
        self.assertIsNone(feature.grad)

    def test_ema_updates_teacher_only(self):
        student = nn.Linear(2, 2, bias=False)
        teacher = copy.deepcopy(student)
        with torch.no_grad():
            student.weight.fill_(2.0)
            teacher.weight.zero_()
        ema_update(teacher, student, 0.5)
        torch.testing.assert_close(teacher.weight, torch.ones_like(teacher.weight))
        torch.testing.assert_close(student.weight, torch.full_like(student.weight, 2.0))


if __name__ == "__main__":
    unittest.main()

