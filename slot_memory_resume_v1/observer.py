"""Scoped phase observation around the unchanged original arm update function."""
from __future__ import annotations

from nll_trace import NLLTrace

_ABSENT = object()


class UpdateObserver:
    """Mark original updates; optional dense snapshots inspect saved NLL targets.

    ``completed_arms`` records optimizer post-hooks, not journal commits. Prefix
    markers use Python metadata only. Dense snapshots synchronize and change
    allocation timing, so passing a replay does not establish a CUDA fix.
    """

    def __init__(self, torch, runtime, factory, heads, optimizers, emit, dense=False):
        self.torch, self.runtime, self.factory = torch, runtime, factory
        self.heads, self.optimizers, self.emit, self.dense = heads, optimizers, emit, dense
        if set(heads) != set(optimizers) or not heads:
            raise ValueError("Observer requires the same nonempty head/optimizer arm set")
        if len({id(opt) for opt in optimizers.values()}) != len(optimizers):
            raise ValueError("Observer requires independent optimizer instances")
        self.step = None
        self.arm = None
        self.image = None
        self.last_marker = {}
        self.completed_arms = []
        self.failed_trace = None
        self._trace = None
        self._loss_hook = None
        self._optimizer_hooks = []
        self._saved_attributes = []
        self._entered = False

    def _mark(self, phase):
        self.last_marker = dict(
            step=self.step, arm=self.arm, image=self.image, phase=phase,
            completed_arms=list(self.completed_arms), dense=bool(self.dense),
        )
        self.emit("update_phase", **self.last_marker)

    def _trace_emit(self, event, **fields):
        self.emit(event, step=self.step, arm=self.arm, image=self.image, **fields)

    def _replace(self, instance, name, value):
        saved = instance.__dict__.get(name, _ABSENT)
        setattr(instance, name, value)
        self._saved_attributes.append((instance, name, saved))

    def _close_arm(self, exception=(None, None, None)):
        if self._loss_hook is not None:
            self._loss_hook.remove()
            self._loss_hook = None
        if self._trace is not None:
            trace, self._trace = self._trace, None
            if exception[0] is not None:
                self.failed_trace = trace
            trace.__exit__(*exception)

    def set_step(self, step):
        if type(step) is not int or step < 1:
            raise ValueError("Observer step must be a positive integer")
        self._close_arm()
        self.step, self.arm, self.image = step, None, None
        self.completed_arms = []
        self._mark("step_begin")

    def __enter__(self):
        if self._entered:
            raise RuntimeError("UpdateObserver cannot be entered twice")
        self._entered = True
        original_forward = self.factory.forward
        original_loss = self.runtime.loss

        def forward(head, example, arm, support):
            if self.step is None:
                raise RuntimeError("Call observer.set_step before update_arms")
            self._close_arm()
            self.arm, self.image = arm, str(example.image)
            self._mark("forward_begin")
            if arm not in self.heads or head is not self.heads[arm]:
                raise ValueError("Observed forward head/arm identity differs")
            if self.dense:
                self._trace = NLLTrace(self.torch, self.runtime.deps, self._trace_emit)
                self._trace.__enter__()
            result = original_forward(head, example, arm, support)
            self._mark("forward_end")
            return result

        def loss(*args, **kwargs):
            self._mark("loss_begin")
            result = original_loss(*args, **kwargs)
            if self._trace is not None:
                self._trace.snapshot("after_original_loss")
            self._mark("loss_end")

            def before_backward(gradient):
                self._mark("backward_begin")
                if self._trace is not None:
                    self._trace.snapshot("before_backward")
                return None

            if result.requires_grad:
                if self._loss_hook is not None:
                    self._loss_hook.remove()
                self._loss_hook = result.register_hook(before_backward)
            return result

        def pre_hook(arm):
            def hook(optimizer, args, kwargs):
                if self.arm != arm or optimizer is not self.optimizers[arm]:
                    raise ValueError("Observed optimizer/arm identity differs")
                self._mark("optimizer_step_begin")
                return None
            return hook

        def post_hook(arm):
            def hook(optimizer, args, kwargs):
                self.completed_arms.append(arm)
                self._mark("optimizer_step_end")
            return hook

        try:
            self._replace(self.factory, "forward", forward)
            self._replace(self.runtime, "loss", loss)
            for arm, optimizer in self.optimizers.items():
                self._optimizer_hooks.append(optimizer.register_step_pre_hook(pre_hook(arm)))
                self._optimizer_hooks.append(optimizer.register_step_post_hook(post_hook(arm)))
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        for hook in reversed(self._optimizer_hooks):
            hook.remove()
        self._optimizer_hooks.clear()
        for instance, name, saved in reversed(self._saved_attributes):
            if saved is _ABSENT:
                delattr(instance, name)
            else:
                setattr(instance, name, saved)
        self._saved_attributes.clear()

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._close_arm((exc_type, exc, traceback))
        finally:
            self._restore()
        return False
