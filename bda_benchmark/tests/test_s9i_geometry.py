import sys
import types
from dataclasses import dataclass

import numpy as np
import torch

model_stub = types.ModuleType("model.DamageFormer_S8_MICB_DSMIL")

@dataclass
class BuildingBagSpec:
    batch_index: int
    mask: torch.Tensor
    label: int = -1
    area: int = 0

model_stub.BuildingBagSpec = BuildingBagSpec
sys.modules["model.DamageFormer_S8_MICB_DSMIL"] = model_stub

common_stub = types.ModuleType("script.standard_ML.s9_instance_common")

@dataclass(frozen=True)
class MaskInstance:
    instance_id: int
    batch_index: int
    mask: np.ndarray
    area: int
    bbox_yxyx: tuple
    label: int | None = None
    purity: float = 0.0

common_stub.MaskInstance = MaskInstance
common_stub.area_bin = lambda area: "large_ge256"
common_stub.classification_report = lambda x: {}
common_stub.confusion_update = lambda *args: None
common_stub.iou_matrix = lambda *args: np.zeros((0, 0))
common_stub.match_instances = lambda *args, **kwargs: []
common_stub.merge_split_diagnostics = lambda *args, **kwargs: {
    "merge_pred_instances": 0,
    "split_gt_instances": 0,
}
common_stub.normalize_label_tensor = lambda x: x
common_stub.make_predicted_specs = lambda loc, builder, min_area=32: builder(
    loc, torch.where(loc == 1, 1, 0), min_area=min_area
)
common_stub.instances_from_bag_specs = lambda specs, keep_labels=False: [
    MaskInstance(
        i,
        spec.batch_index,
        spec.mask.numpy().astype(bool),
        spec.area,
        (
            int(np.nonzero(spec.mask.numpy())[0].min()),
            int(np.nonzero(spec.mask.numpy())[1].min()),
            int(np.nonzero(spec.mask.numpy())[0].max()) + 1,
            int(np.nonzero(spec.mask.numpy())[1].max()) + 1,
        ),
    )
    for i, spec in enumerate(specs)
]
sys.modules["script.standard_ML.s9_instance_common"] = common_stub

from s9_geometry_common import LearnedConfig, learned_geometry_instances


def fake_builder(loc, damage, min_area=32):
    mask = loc[0].numpy() == 1
    return [
        BuildingBagSpec(
            0,
            torch.from_numpy(mask),
            1,
            int(mask.sum()),
        )
    ]


yy, xx = np.mgrid[:96, :96]
left = (yy - 48) ** 2 + (xx - 34) ** 2 <= 18 ** 2
right = (yy - 48) ** 2 + (xx - 62) ** 2 <= 18 ** 2
building = np.zeros((96, 96), np.float32)
building[left | right] = 0.95
center = np.zeros_like(building)
center[(yy - 48) ** 2 + (xx - 34) ** 2 <= 5 ** 2] = 0.95
center[(yy - 48) ** 2 + (xx - 62) ** 2 <= 5 ** 2] = 0.95
boundary = np.zeros_like(building)
boundary[:, 48:49] = 1.0

instances, specs, diagnostics = learned_geometry_instances(
    building,
    center,
    boundary,
    fake_builder,
    LearnedConfig(min_region_area_to_split=128),
)
assert len(instances) == len(specs) == 2, diagnostics
print("Synthetic S9-I center-boundary test: PASS", diagnostics)
