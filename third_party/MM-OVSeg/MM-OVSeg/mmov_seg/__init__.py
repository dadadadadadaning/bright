# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_cat_seg_config

# dataset loading
from .data.dataset_mappers.detr_panoptic_dataset_mapper import DETRPanopticDatasetMapper
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
    MaskFormerSemanticDatasetMapper_test,
)

# models
from .test_time_augmentation import SemanticSegmentorWithTTA

#from .MMOV import MMOV   ### If you want to use DINO v1 (ViT-B/16)
from .MMOVL14 import MMOV   ### If you want to use DINO v3 (ViT-L/14)
from .vision_transformer import vit_base


# from .vision_transformer import * 
# from .vision_transformer import vits
