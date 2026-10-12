import torch

from .core import affected_component_area, alignment_displacement_proxy, core_and_ring, modification_cluster_area, robust_sar_noise


def test_core_ring_partitions_gt_building():
    building = torch.zeros(7, 7, dtype=torch.bool); building[1:6, 1:6] = True
    core, ring = core_and_ring(building, 1)
    assert int(core.sum()) == 9
    assert torch.equal(core | ring, building)
    assert not bool((core & ring).any())


def test_affected_component_areas_are_constant_per_component():
    target = torch.zeros(5, 5, dtype=torch.long); target[1:3, 1:3] = 2; target[4, 4] = 3
    area = affected_component_area(target)
    assert torch.equal(area[1:3, 1:3], torch.full((2, 2), 4))
    assert int(area[4, 4]) == 1


def test_modification_clusters_are_prediction_only_connected_components():
    changed = torch.zeros(4, 5, dtype=torch.bool); changed[0:2, 0:2] = True; changed[3, 4] = True
    area, count = modification_cluster_area(changed)
    assert count == 2 and int(area[0, 0]) == 4 and int(area[3, 4]) == 1


def test_noise_and_alignment_maps_are_finite_and_shaped_like_input():
    rgb = torch.rand(3, 9, 11); sar = torch.rand(9, 11)
    assert robust_sar_noise(sar).shape == sar.shape
    proxy = alignment_displacement_proxy(rgb, sar)
    assert proxy.shape == sar.shape and torch.isfinite(proxy).all() and bool((proxy >= 0).all())
