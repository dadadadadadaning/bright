"""Scalar allocator observations only; never hold new model tensors or repair routes."""
import json
import time
from evaluate_checkpoint import write_json
from memcheck_routes import CorroboratedRoutes


class MemoryTrace:
    def __init__(self, torch, output):
        self.torch, self.output = torch, output
        self.rows = 0
        self.failed = False
        self.last = None
        self.started = time.monotonic()
        self.stream = (output / 'memory.jsonl').open('x')

    def write(self, row):
        row = dict(row, elapsed_seconds=round(time.monotonic() - self.started, 3))
        self.stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.stream.flush()
        self.rows += 1
        self.last = row
        write_json(self.output / 'memory_latest.json', row)

    def sample(self, label, position=None):
        if self.failed:
            return
        # No synchronize(), tensor allocation, empty_cache() or GC intervention.
        # mem_get_info is a CUDA API and can affect observation timing.
        cuda = self.torch.cuda
        stats = cuda.memory_stats()
        free, total = cuda.mem_get_info()
        self.write(dict(stage=label, position=dict(position or {}),
            allocated_bytes=cuda.memory_allocated(), reserved_bytes=cuda.memory_reserved(),
            peak_allocated_bytes=cuda.max_memory_allocated(),
            peak_reserved_bytes=cuda.max_memory_reserved(), device_free_bytes=free,
            device_total_bytes=total, grad_enabled=self.torch.is_grad_enabled(),
            allocator={k: stats.get(k) for k in (
                'active_bytes.all.current', 'inactive_split_bytes.all.current',
                'allocation.all.current', 'num_alloc_retries', 'num_ooms')}))

    def failure(self, label, exc, position=None):
        if self.failed:
            return
        self.failed = True
        # Preserve the original exception; no allocator/CUDA query after failure.
        self.write(dict(stage=label, position=dict(position or {}),
                        error=type(exc).__name__ + ': ' + str(exc),
                        cuda_queried_after_failure=False))

    def close(self):
        if not self.stream.closed:
            self.stream.close()


class MemoryRoutes(CorroboratedRoutes):
    def __init__(self, *args, trace, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = trace
        self.handles = []

    def measured(self, label, original):
        def invoke(*args, **kwargs):
            try:
                position = dict(self.position)
                if label == 'arm_forward':
                    position['active_arm'] = args[2] if len(args) > 2 else kwargs['arm']
                self.trace.sample(label + '_before', position)
                result = original(*args, **kwargs)
                self.trace.sample(label + '_after', self.position)
                return result
            except BaseException as exc:
                self.trace.failure(label + '_failed', exc, self.position)
                raise
        return invoke

    def __enter__(self):
        super().__enter__()
        try:
            for obj, name, label in ((self.runtime, 'prepare', 'prepare'),
                                     (self.factory, 'prepare', 'supports'),
                                     (self.factory, 'forward', 'arm_forward')):
                self.bind(obj, name, self.measured(label, getattr(obj, name)))
            front = self.runtime.bundle['front']
            modules = [('front', front)] + [('front.' + k, m) for k, m in front.named_children()]
            if hasattr(front, 'embedding'):
                modules += [('front.embedding.' + k, m) for k, m in front.embedding.named_children()]
            seen = set()
            for label, module in modules:
                if id(module) in seen:
                    continue
                seen.add(id(module))
                def before(_module, _inputs, label=label):
                    self.trace.sample(label + '_before', self.position)
                def after(_module, _inputs, _output, label=label):
                    self.trace.sample(label + '_after', self.position)
                self.handles.extend((module.register_forward_pre_hook(before),
                                     module.register_forward_hook(after)))
            return self
        except BaseException:
            for handle in self.handles:
                handle.remove()
            self.restore()
            self.journal.close()
            raise

    def __exit__(self, kind, error, tb):
        if error is not None:
            self.trace.failure('evaluation_failed', error, self.position)
        try:
            return super().__exit__(kind, error, tb)
        finally:
            for handle in self.handles:
                handle.remove()
