import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def capture(model, pre, post):
    """Single forward: same Q, original and modulated content, exact pixel input.

    No inference_mode: captured tensors may subsequently feed trainable heads.
    No old files/buffers/weights are edited. The caller freezes the front end.
    """
    model.eval()
    fused, inputs = [], []
    handles = [
        model.base.fusion_layer_3.register_forward_hook(lambda m, i, o: fused.append(o)),
        model.base.clf_2.register_forward_pre_hook(lambda m, i: inputs.append(i[0])),
    ]
    try:
        r = model(pre, post)
    finally:
        for h in handles:
            h.remove()
    if len(fused) != 2 or len(inputs) != 1:
        raise RuntimeError('Expected V2: two fusion calls and one pixel head call')
    replay = F.interpolate(model.base.clf_2(inputs[0]), pre.shape[-2:],
                           mode='bilinear', align_corners=False)
    if not torch.allclose(replay, r['pixel'], atol=1e-5, rtol=1e-5):
        raise RuntimeError('Exact original pixel-input/head replay failed')
    r['features'] = dict(raw=fused[0].detach(), normal=fused[1].detach(),
                         roi=inputs[0].detach())
    r['identity_max_abs'] = float((replay-r['pixel']).abs().max())
    return r


class BoundedFeatureModulation(nn.Module):
    """Optional future FiLM-inspired spatial adaptation, NOT enabled by train.py.

    Leaves RGB/SAR unchanged. Identity initialization is not a safety guarantee
    after training. `condition` must be predicted features, never GT masks.
    """
    def __init__(self, channels=256, condition_channels=1, hidden=32, bound=0.1):
        super().__init__()
        if not 0 < bound <= 1:
            raise ValueError('bound must lie in (0,1]')
        self.bound = bound
        self.net = nn.Sequential(nn.Conv2d(condition_channels, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, 2*channels, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature, condition):
        condition = F.interpolate(condition.detach().float(), feature.shape[-2:],
                                  mode='bilinear', align_corners=False)
        gamma, beta = self.net(condition).chunk(2, 1)
        # Spatially constant per-image/channel scale; detached, robust to zeros.
        scale = feature.detach().float().square().mean((2, 3), keepdim=True).add(1e-6).sqrt()
        return feature * (1+self.bound*gamma.tanh()) + self.bound*scale*beta.tanh()
