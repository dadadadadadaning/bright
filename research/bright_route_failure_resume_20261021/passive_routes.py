"""Retain the original predicate operands only until each validator returns.

Same validator code object, arguments, operators and short-circuit behavior.
No extra CUDA operations or tensor copies on accepted calls. Python wrappers and
briefly retained predicate tensors still alter timing/allocation lifetimes.
"""
from __future__ import annotations

import builtins
import contextvars
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
import types

from evaluate_checkpoint import write_json
from route_trace import RouteTrace


class TorchProxy:
    def __init__(self, original, current):
        self.original, self.current = original, current

    def __getattr__(self, name):
        return getattr(self.original, name)

    def allclose(self, *args, **kwargs):
        call = self.current.get()
        item = dict(operands=args, kwargs=kwargs, result=None)
        call["allclose"].append(item)
        result = self.original.allclose(*args, **kwargs)
        item["result"] = result
        return result


class PassiveRoutes:
    def __init__(self, torch, classifier, output, position):
        self.torch, self.classifier, self.output, self.position = torch, classifier, Path(output), position
        self.calls = 0
        self.failures = 0
        self.restores = []
        self.identities = []
        self.evidence_error = None

    def wrap(self, original, binding):
        if original.__globals__.get("bool", builtins.bool) is not builtins.bool:
            raise ValueError("Route validator has a nonstandard bool binding")
        if original.__globals__.get("torch") is not self.torch:
            raise ValueError("Route validator torch binding differs")
        source = inspect.getsource(original)
        self.identities.append(dict(binding=binding, source_file=inspect.getsourcefile(original),
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            code_sha256=hashlib.sha256(original.__code__.co_code).hexdigest()))
        current = contextvars.ContextVar("route_predicates_" + binding)

        def record_bool(value):
            item = dict(operand=value, result=None, line=sys._getframe(1).f_lineno)
            current.get()["bool"].append(item)
            result = builtins.bool(value)
            item["result"] = result
            return result

        scope = dict(original.__globals__, torch=TorchProxy(self.torch, current), bool=record_bool)
        instrumented = types.FunctionType(original.__code__, scope, original.__name__,
                                          original.__defaults__, original.__closure__)
        instrumented.__kwdefaults__ = original.__kwdefaults__
        if instrumented.__code__ is not original.__code__:
            raise AssertionError("Validator code object changed")

        @functools.wraps(original)
        def call(*args, **kwargs):
            self.calls += 1
            operands = dict(bool=[], allclose=[])
            token = current.set(operands)
            try:
                return instrumented(*args, **kwargs)
            except BaseException as exc:
                self.failures += 1
                frame = sys._getframe(1)
                sites = []
                for _ in range(2):
                    if frame is None:
                        break
                    sites.append(dict(file=frame.f_code.co_filename,line=frame.f_lineno,function=frame.f_code.co_name))
                    frame = frame.f_back
                del frame
                values = {key: args[i] if len(args) > i else kwargs.get(key)
                          for i,key in enumerate(("q","ids","count","batch","pixels"))}
                try:
                    self.capture(exc, binding, sites, values, operands)
                except BaseException as evidence_exc:
                    self.evidence_error = type(evidence_exc).__name__ + ": " + str(evidence_exc)
                    print("ROUTE EVIDENCE SAVE ERROR: " + self.evidence_error, flush=True)
                raise
            finally:
                current.reset(token)
        return call

    def capture(self, exc, binding, sites, values, operands):
        report = dict(format="BRIGHT_ORIGINAL_ROUTE_PREDICATES_V1", position=dict(self.position),
            sequence=self.calls, binding=binding, callsites=sites,
            error=type(exc).__name__ + ": " + str(exc),
            original_bool=[dict(line=v["line"],result=v["result"]) for v in operands["bool"]],
            original_allclose=[dict(kwargs=v["kwargs"],result=v["result"]) for v in operands["allclose"]],
            copied_only_after_failure=True, route_values_modified=False,
            tensor_reads_skipped=not isinstance(exc,ValueError), cpu_copy_errors=[])
        cpu = {}
        blocked = not isinstance(exc,ValueError)
        helper = RouteTrace(self.torch,self.classifier,lambda *a,**k:None)
        def copy_tensor(name,tensor):
            nonlocal blocked
            if blocked or not isinstance(tensor,self.torch.Tensor):
                return
            try:
                cpu[name] = tensor.detach().to(device="cpu",copy=True).contiguous()
            except BaseException as copy_exc:
                report["cpu_copy_errors"].append(dict(name=name,error=type(copy_exc).__name__+": "+str(copy_exc)))
                blocked = True
        # First preserve original reduction operands, not newly recomputed sums.
        for i,item in enumerate(operands["allclose"]):
            for j,tensor in enumerate(item["operands"][:2]):
                copy_tensor("allclose_%s_%s" % (i,j),tensor)
        for i,item in enumerate(operands["bool"]):
            copy_tensor("bool_%s" % i,item["operand"])
        for name in ("q","ids"):
            copy_tensor(name,values[name])
        if "q" in cpu:
            report["weights"] = helper._stats(cpu["q"])
            report["q_cpu_recomputed"] = helper._weight_details(cpu["q"],cpu.get("ids"))
        for i,item in enumerate(report["original_allclose"]):
            left,right = cpu.get("allclose_%s_0" % i),cpu.get("allclose_%s_1" % i)
            if left is not None and right is not None:
                item["cpu_check_of_original_operands"] = bool(self.torch.allclose(left,right,**item["kwargs"]))
                item["original_left_sums_copied_to_cpu"] = helper._sum_stats(left)[0]
        report["tensor_reads_stopped"] = blocked
        self.output.mkdir(parents=True,exist_ok=True)
        write_json(self.output/"route_failure.json",report)
        # Serialization uses only owned CPU snapshots, even after CUDA errors.
        self.torch.save(dict(report=report,tensors=cpu),self.output/"route_failure_cpu.pt")
        print("ORIGINAL ROUTE CHECK FAILED " + json.dumps(report,allow_nan=False),flush=True)

    def __enter__(self):
        namespace = self.classifier.forward.__globals__
        candidates = (("classifier",namespace),("slots",namespace["sparse_unpool"].__globals__))
        seen = set()
        try:
            for binding,scope in candidates:
                if id(scope) in seen:
                    continue
                seen.add(id(scope))
                original = scope["validate_routes"]
                wrapper = self.wrap(original,binding)
                self.restores.append((scope,original))
                scope["validate_routes"] = wrapper
            write_json(self.output/"route_observer_identity.json",dict(
                validators=self.identities, validator_code_object_unchanged=True,
                original_predicate_calls_once=True, original_short_circuit_retained=True,
                successful_call_extra_cuda_operations=0, successful_call_cpu_copies=0,
                exact_historical_timing=False, inputs_modified=False))
        except BaseException:
            self.__exit__(None,None,None)
            raise
        return self

    def __exit__(self,*exc):
        for scope,original in reversed(self.restores):
            scope["validate_routes"] = original
        try:
            write_json(self.output/"route_observer_summary.json",dict(
                calls=self.calls,failures=self.failures,evidence_error=self.evidence_error,
                bindings_restored=True, successful_call_extra_cuda_operations=0))
        except OSError:
            pass
        return False
