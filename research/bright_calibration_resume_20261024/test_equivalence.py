"""CPU equivalence on the actual published Reader/routes and NLL + Lovasz.

This verifies finite-input semantics, not CUDA stability or fault resolution.
Set BRIGHT_SEVERITY_PACKAGE, BRIGHT_REFERENCE_MODEL and BRIGHT_LOVASZ_SOURCE
to their installed files when running outside the delivery workspace.
"""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from typing import List, Tuple

import torch
from torch.nn import functional as F

import lean_adapter

PACKAGE = Path(os.environ.get('BRIGHT_SEVERITY_PACKAGE', '/workspace/bright_severity_memory_20261023'))
REFERENCE = Path(os.environ.get('BRIGHT_REFERENCE_MODEL', '/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2/deps/original/model.py'))
LOVASZ = Path(os.environ.get('BRIGHT_LOVASZ_SOURCE', '/workspace/bright-design-delivery/bda_benchmark/script/standard_ML/train_S5_S8_GPU.py'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def fixture(slots, empty=False):
    torch.manual_seed(187)
    members = 0 if empty else 4
    pixel = torch.randn(1, 256, 2, 3)
    q = torch.rand(1, 9, 6)
    q = q / q.sum(1, keepdim=True)
    ids = torch.arange(54).reshape(9, -1) % 3
    built = torch.rand(1, 1, 4, 6)
    prepared = slots.prepare_slots(pixel, q, ids, 3, (2, 3), built, 'soft')
    labels = torch.rand(members, 2, 4)
    if members:
        labels /= labels.sum(-1, keepdim=True)
    return dict(pixel_features=pixel, region_f=torch.randn(1, 3, 256),
        region_t=torch.randn(1, 3, 128), q=q, ids=ids,
        base_probability_full=torch.rand(1, 4, 4, 6).softmax(1),
        **{key: prepared[key] for key in ('slot_f', 'slot_fraction', 'slot_valid', 'route_q', 'route_ids')},
        support_ids=(torch.full((3, 3), -1, dtype=torch.long) if empty else
                     torch.tensor([[0, 1, -1], [1, 2, 3], [-1, -1, -1]])),
        support_f=torch.randn(members, 2, 256), support_t=torch.randn(members, 128),
        support_labels=labels, support_fraction=torch.full((members, 2), .5),
        support_valid=torch.ones(members, 2, dtype=torch.bool))


def nodes(loss):
    seen, pending, result = set(), [loss.grad_fn], []
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        result.append(node)
        pending.extend(child for child, _ in node.next_functions)
    return result


class Equivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        package = types.ModuleType('_lean_reference')
        package.__path__ = [str(REFERENCE.parent)]
        sys.modules[package.__name__] = package
        cls.base = module('_lean_reference.model', REFERENCE).JointRegionMemoryClassifier
        cls.slots = sys.modules['_lean_reference.slots']
        cls.old = module('_lean_old_readout', PACKAGE / 'old_model_adapter.py')
        cls.published = module('_lean_published', PACKAGE / 'severity_adapter.py')
        # Execute only the unchanged loss functions, avoiding training imports.
        tree = ast.parse(LOVASZ.read_text())
        names = {'lovasz_grad', 'flatten_probas', 'lovasz_softmax_flat', 'lovasz_softmax'}
        body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if len(body) != len(names):
            raise ValueError('Actual Lovasz source functions missing')
        scope = dict(torch=torch, F=F, List=List, Tuple=Tuple)
        exec(compile(ast.Module(body=body, type_ignores=[]), str(LOVASZ), 'exec'), scope)
        cls.lovasz = staticmethod(scope['lovasz_softmax'])

    def model(self, adapter, mode='calibration'):
        torch.manual_seed(73)
        model = adapter.make_classifier(self.base, self.old, self.slots, mode)(
            hidden=8, heads=2, chunk_size=2, shuffle_seed=3407)
        with torch.no_grad():
            model.joint_head[-1].weight.normal_(0, .03)
            model.joint_head[-1].bias.normal_(0, .03)
            if mode != 'original':
                model.severity_bias_raw.copy_(torch.tensor([.3, -.2, .5]))
                # Nonzero inactive gain tests AdamW decay, not only zero init.
                model.memory_gain_raw.fill_(.4)
        return model

    def loss(self, output):
        p = output['probabilities']
        y = (torch.arange(p.shape[-2] * p.shape[-1]) % 4).reshape(1, *p.shape[-2:])
        table = p.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
        return F.nll_loss(table.clamp_min(1e-8).log(), y.flatten()) + self.lovasz(p, y)

    def same(self, before, after, label=''):
        if isinstance(before, torch.Tensor):
            torch.testing.assert_close(before, after, atol=1e-7, rtol=1e-6, msg=label)
        elif isinstance(before, dict):
            self.assertEqual(before.keys(), after.keys(), label)
            for key in before:
                self.same(before[key], after[key], label + '/' + str(key))
        elif isinstance(before, (tuple, list)):
            self.assertEqual(len(before), len(after), label)
            for i, (a, b) in enumerate(zip(before, after)):
                self.same(a, b, label + '/' + str(i))
        else:
            self.assertEqual(before, after, label)

    def test_forward_diagnostics_rng_and_input_storage_unchanged(self):
        for empty in (False, True):
            old, new = self.model(self.published), self.model(lean_adapter)
            inputs = fixture(self.slots, empty)
            snapshots = copy.deepcopy(inputs)
            rng = torch.get_rng_state().clone()
            before, after = old(**inputs), new(**inputs)
            self.same(before, after, 'all forward outputs')
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.same(inputs, snapshots, 'input mutation')
            self.same(old.state_dict(), new.state_dict(), 'state schema')

    def test_all_gradients_and_adamw_state_match_over_four_updates(self):
        old, new = self.model(self.published), self.model(lean_adapter)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=.005) for m in (old, new)]
        inputs = fixture(self.slots)
        for step in range(4):
            losses = []
            for model, optimizer in zip((old, new), optimizers):
                optimizer.zero_grad(set_to_none=True)
                loss = self.loss(model(**inputs))
                losses.append(loss.detach())
                loss.backward()
            self.same(losses[0], losses[1], 'loss')
            for (name, p), (name2, q) in zip(old.named_parameters(), new.named_parameters()):
                self.assertEqual(name, name2)
                self.same(p.grad, q.grad, name + ' gradient')
            self.assertIsNotNone(new.memory_gain_raw.grad)
            self.assertEqual(new.memory_gain_raw.grad.item(), 0)
            for model, optimizer in zip((old, new), optimizers):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            self.same(old.state_dict(), new.state_dict(), 'parameters')
            self.same(optimizers[0].state_dict(), optimizers[1].state_dict(), 'AdamW')
            self.assertEqual(int(optimizers[1].state[new.memory_gain_raw]['step']), step+1)

    def test_zero_gradient_evidence_gathers_removed_but_reader_stays_live(self):
        counts = []
        for adapter in (self.published, lean_adapter):
            model = self.model(adapter)
            loss = self.loss(model(**fixture(self.slots)))
            graph = nodes(loss)
            gathers = [n for n in graph if type(n).__name__.startswith('GatherBackward')]
            extra = [n for n in gathers if n._saved_self.shape[-1] == 4]
            incoming = []
            handles = [n.register_prehook(lambda grads: incoming.append(float(grads[0].abs().sum()))) for n in extra]
            loss.backward()
            for h in handles:
                h.remove()
            counts.append((len(gathers), len(extra), len(incoming)))
            self.assertTrue(all(value == 0 for value in incoming))
            self.assertGreater(float(model.query_projection.weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.value_projection.weight.grad.abs().sum()), 0)
        self.assertEqual(counts, [(63, 18, 18), (45, 0, 0)])

    def test_other_modes_keep_outputs_gradients_and_live_memory_gain(self):
        for mode in ('original', 'severity_memory', 'shuffled_memory'):
            old, new = self.model(self.published, mode), self.model(lean_adapter, mode)
            before, after = old(**fixture(self.slots)), new(**fixture(self.slots))
            self.same(before, after, mode)
            self.loss(before).backward()
            self.loss(after).backward()
            for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
                self.same(p.grad, q.grad, mode + name)
            if mode != 'original':
                self.assertGreater(float(new.memory_gain_raw.grad.abs()), 0)


if __name__ == '__main__':
    unittest.main()
