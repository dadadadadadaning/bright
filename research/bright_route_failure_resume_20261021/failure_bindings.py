"""Bind failure-only route evidence to the unchanged continuation workflow."""
from __future__ import annotations

import contextlib
import functools
from pathlib import Path

from passive_routes import PassiveRoutes
from training_common import observe_bindings as probability_observations


@contextlib.contextmanager
def observe_bindings(context,output):
    bridge = context["bridge"]
    position = dict(mode="loading",step=None,index=None,image=None,active_arm=None,stage="loading")
    evaluation = dict(names=None,index=-1)
    addon = context["addon"]
    with probability_observations(context,output), contextlib.ExitStack() as stack:
        load_runtime,named = bridge.load_runtime,bridge.named
        def restore_attr(obj,name,value):
            setattr(obj,name,value)

        def load(source):
            bindings = load_runtime(source)
            stack.enter_context(PassiveRoutes(bindings.torch,
                bindings.original_train.JointRegionMemoryClassifier,output,position))
            original_runtime = bindings.Runtime
            def construct(*args,**kwargs):
                runtime = original_runtime(*args,**kwargs)
                original_prepare = runtime.prepare
                had_prepare = "prepare" in vars(runtime)
                saved_prepare = vars(runtime).get("prepare")
                def prepare(batch):
                    if evaluation["names"] is not None:
                        evaluation["index"] += 1
                        index = evaluation["index"]
                        if index >= len(evaluation["names"]):
                            raise ValueError("Evaluation exceeded declared image count")
                        position.update(index=index,image=evaluation["names"][index],
                                        active_arm=None,stage="prepare_begin")
                    else:
                        position.update(mode="training_prepare",step=None,index=None,
                                        image=None,active_arm=None,stage="prepare_begin")
                    example = original_prepare(batch)
                    position.update(image=example.image,stage="prepare_end")
                    return example
                runtime.prepare = prepare
                def restore_prepare():
                    if had_prepare:
                        runtime.prepare = saved_prepare
                    else:
                        del runtime.prepare
                stack.callback(restore_prepare)
                return runtime
            bindings.Runtime = construct
            stack.callback(restore_attr,bindings,"Runtime",original_runtime)
            return bindings

        def module(path,label):
            result = named(path,label)
            if Path(path) == addon/"supports.py":
                cls = result.SupportFactory
                forward,prepare = cls.forward,cls.prepare
                @functools.wraps(forward)
                def call(self,head,example,arm,support):
                    position.update(image=example.image,active_arm=arm,stage="forward_begin")
                    value = forward(self,head,example,arm,support)
                    position.update(stage="after_forward",active_arm=None,last_forward_arm=arm)
                    return value
                @functools.wraps(prepare)
                def supports(self,example,arms):
                    position.update(image=example.image,active_arm=None,stage="support_prepare_begin")
                    value = prepare(self,example,arms)
                    position["stage"] = "support_prepare_end"
                    return value
                cls.forward,cls.prepare = call,supports
                stack.callback(restore_attr,cls,"forward",forward)
                stack.callback(restore_attr,cls,"prepare",prepare)
            elif Path(path) == addon/"evaluation.py":
                original = result.evaluate
                @functools.wraps(original)
                def assess(runtime,heads,names,folder,**kwargs):
                    position.update(mode="evaluation",step=kwargs["step"],split=kwargs["split"],
                                    index=None,image=None,active_arm=None,stage="evaluation_begin")
                    evaluation.update(names=list(names),index=-1)
                    try:
                        return original(runtime,heads,names,folder,**kwargs)
                    finally:
                        evaluation.update(names=None,index=-1)
                result.evaluate = assess
                stack.callback(restore_attr,result,"evaluate",original)
            elif Path(path) == addon/"ablation_runner.py":
                original = result.update_arms
                @functools.wraps(original)
                def update(heads,optimizers,example,runtime,factory,state,**kwargs):
                    position.update(mode="training",step=kwargs["step"],split="train",index=None,
                                    image=example.image,active_arm=None,stage="update_begin")
                    return original(heads,optimizers,example,runtime,factory,state,**kwargs)
                result.update_arms = update
                stack.callback(restore_attr,result,"update_arms",original)
            return result

        bridge.load_runtime,bridge.named = load,module
        try:
            yield
        finally:
            bridge.load_runtime,bridge.named = load_runtime,named
