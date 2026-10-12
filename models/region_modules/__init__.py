"""Region-level auxiliary losses for BRIGHT damage assessment."""

from .region_pooling_aux import RegionPoolingAuxLoss
from .region_graph_aux import RegionGraphAuxLoss, SimpleGCNLayer

__all__ = ["RegionPoolingAuxLoss", "RegionGraphAuxLoss", "SimpleGCNLayer"]

