"""Identity check using the classifier output from the original A1 forward."""
from __future__ import annotations


class NativeA1LogitsCapture:
    """Capture the single native A1 classifier call from SRAC front().

    This SRAC front wraps DamageFormer_S8_MICB_DSMIL directly: clf_2 runs
    once. Probe calls occur only after take(), and are cleared before the
    next image. Neither zero nor multiple native calls are accepted.
    """

    def __init__(self, front):
        self._calls = 0
        self._last = None
        self._handle = front.base.clf_2.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self._calls += 1
        self._last = output.detach()

    def clear(self):
        self._calls, self._last = 0, None

    def take(self):
        if self._calls != 1 or self._last is None:
            raise RuntimeError(f"Expected exactly one native A1 classifier call, got {self._calls}")
        result = self._last
        self.clear()
        return result

    def close(self):
        self._handle.remove()


def native_replay(low_logits, original_logits, output_size):
    """The extra frozen probes are deliberately excluded from this identity gate."""
    from torch.nn import functional as F
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.shared import close_report

    if (low_logits.ndim != 4 or low_logits.shape[1] != 4
            or original_logits.shape[:2] != low_logits.shape[:2]
            or tuple(original_logits.shape[-2:]) != tuple(output_size)):
        raise ValueError("Native A1 grid and original image logits disagree")
    replay = F.interpolate(low_logits.float(), size=output_size,
                           mode="bilinear", align_corners=False).softmax(1)
    return close_report(replay, original_logits.float().softmax(1))
