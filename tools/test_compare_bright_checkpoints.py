"""Small synthetic correctness checks; fixtures are NOT research results."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
import compare_bright_checkpoints as c


class ComparisonTests(unittest.TestCase):
    def test_state(self):
        v=torch.zeros(2)
        self.assertEqual(list(c.state_of({'model':{'module.a':v}})),['a'])
        with self.assertRaises(ValueError):c.state_of({'plugin_model':{'a':v}})

    def test_normalization(self):
        x=np.zeros((2,3,3),dtype=np.uint8)
        t=c.normalize(x)
        self.assertEqual(tuple(t.shape),(1,3,2,3))
        np.testing.assert_allclose(t[0,:,0,0].numpy(),-np.array([123.675,116.28,103.53])/np.array([58.395,57.12,57.375]),rtol=1e-6)

    def test_no_shadow_sample_is_forced(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            split=root/'bda_benchmark/dataset/splitname/standard_ML'
            split.mkdir(parents=True)
            (split/'val_set.txt').write_text('bright_0001\ndark_0001\n')
            for folder in ['pre-event','post-event','target']:(root/folder).mkdir()
            for sid,value in [('bright_0001',150),('dark_0001',10)]:
                Image.fromarray(np.full((32,32,3),value,np.uint8)).save(root/'pre-event'/f'{sid}_pre_disaster.tif')
                Image.fromarray(np.zeros((32,32),np.uint8)).save(root/'post-event'/f'{sid}_post_disaster.tif')
                Image.fromarray(np.ones((32,32),np.uint8)).save(root/'target'/f'{sid}_building_damage.tif')
            args=SimpleNamespace(data=str(root),output=str(root/'out'),crop=32,per_event=1,
                dark_threshold=40,max_dark=.15,max_invalid=.01,min_building_pixels=1)
            with patch.object(c,'ROOT',root):c.select(args)
            result=c.load_json(root/'out/selection.json')
            self.assertEqual([s['sample_id'] for s in result['samples']],['bright_0001'])
            self.assertFalse(result['samples'][0]['approved'])
            self.assertEqual(result['rejected_images'],['dark_0001'])
            # Completely dark SAR did NOT cause rejection of the bright RGB pair.


if __name__=='__main__':unittest.main()
