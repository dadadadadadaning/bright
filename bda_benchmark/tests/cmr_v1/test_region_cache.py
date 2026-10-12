import unittest

import numpy as np

from model.cmr_v1.region_cache import pack_component, unpack_component


class RegionCacheTests(unittest.TestCase):
    def test_sparse_component_round_trip(self):
        component = np.zeros((17, 23), dtype=bool)
        component[3:9, 5:14] = True
        component[5:7, 8:11] = False
        packed, bbox, image_shape = pack_component(component)
        restored = unpack_component(packed, bbox, image_shape)
        self.assertTrue(np.array_equal(restored, component))
        self.assertEqual(bbox, (5, 3, 14, 9))

    def test_empty_component_is_rejected(self):
        with self.assertRaises(ValueError):
            pack_component(np.zeros((4, 4), dtype=bool))


if __name__ == "__main__":
    unittest.main()
