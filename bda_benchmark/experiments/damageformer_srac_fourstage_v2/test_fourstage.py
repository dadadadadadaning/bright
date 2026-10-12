import unittest
import torch
from .model import modulate_optical,candidate_map
from .metadata import geometry,disaster_id
from .retrieval import retrieve,SupportClassifier
from .build_neighbors import select_ids,gather_support

class FourStage(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(2)

    def test_modulation_is_in_raw_image_space(self):
        x=torch.randn(1,3,8,8)
        b=torch.ones(1,1,2,2); e=torch.zeros_like(b)
        torch.testing.assert_close(modulate_optical(x,b,e),x)
        mean=torch.tensor([123.675,116.28,103.53])[None,:,None,None]
        std=torch.tensor([58.395,57.12,57.375])[None,:,None,None]
        black=-mean/std
        torch.testing.assert_close(modulate_optical(black.expand_as(x),b,e),black.expand_as(x))
        dark=modulate_optical(x,torch.zeros_like(b),e)
        torch.testing.assert_close(dark*std+mean,(x*std+mean)*.1,atol=1e-5,rtol=1e-5)

    def test_geometry_no_gt(self):
        ids,valid,gh,gw=candidate_map(8,8,4,'cpu')
        q=torch.zeros(1,9,64); q[:,4]=1
        r=dict(shape=(8,8),q=q,ids=ids,tokens=torch.zeros(1,4,16))
        g=geometry(r)
        self.assertEqual(g.shape,(1,4,5))
        torch.testing.assert_close(g[0,:,0],torch.full((4,),.25))
        self.assertTrue(torch.isfinite(g).all())
        self.assertEqual(disaster_id('mexico-hurricane'),6)
        self.assertEqual(disaster_id('not_in_schema'),0)

    def test_neighbor_table_matches_online(self):
        bank=dict(tokens=torch.randn(7,16),distribution=torch.rand(7,4).softmax(-1),
                  events=torch.tensor([0,0,1,1,2,2,2]),images=torch.arange(7),
                  disasters=torch.ones(7,dtype=torch.long),geometry=torch.rand(7,5))
        q=torch.randn(3,16)
        ids=select_ids(q,bank,0,2,5)
        s,d,v,m=gather_support(ids,bank)
        online=retrieve(q,bank,0,2,5,return_metadata=True)
        self.assertTrue(torch.equal(v,online[2]))
        torch.testing.assert_close(s[v],online[0][v])
        self.assertTrue((bank['events'][ids[v]]!=0).all())
        self.assertTrue((bank['images'][ids[v]]!=2).all())
        head=SupportClassifier(16,num_events=3).eval()
        a=head(q,s,d,v,metadata=m)
        b=head(q,*online[:3],metadata=online[3])
        torch.testing.assert_close(a,b)

    def test_mmap_table_roundtrip_and_corruption_guard(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import json
        from .common import digest
        from .build_neighbors import main,open_table
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=root/'cache'; cache.mkdir()
            bank=dict(tokens=torch.randn(5,16),events=torch.tensor([0,1,1,2,2]),images=torch.arange(5))
            shard=dict(tokens=torch.randn(3,16),event=0,image=0)
            torch.save(shard,cache/'shard.pt')
            meta=dict(shards=[dict(path='shard.pt',sha256=digest(cache/'shard.pt'))],bank_sha256='unit-bank')
            (cache/'manifest.json').write_text(json.dumps(meta))
            out=root/'table'
            original_load=torch.load
            mmap_paths=[]
            def compatible_load(path,*args,**kwargs):
                if kwargs.get('mmap'):
                    self.assertIsInstance(path,str,'PyTorch 2.2 mmap requires a string filename')
                    mmap_paths.append(path)
                return original_load(path,*args,**kwargs)
            with patch('sys.argv',['neighbors','--checkpoint','unit','--cache',str(cache),'--output',str(out)]), \
                 patch('bda_benchmark.experiments.damageformer_srac_fourstage_v2.build_neighbors.read_cache',return_value=(cache,meta,bank)), \
                 patch('bda_benchmark.experiments.damageformer_srac_fourstage_v2.build_neighbors.torch.load',side_effect=compatible_load):
                main()
            self.assertEqual(len(mmap_paths),2)
            table,info=open_table(out,cache,'unit-bank',8)
            self.assertEqual(table.shape,(3,8))
            self.assertEqual(info['offsets'],[dict(start=0,count=3)])
            self.assertTrue((table[:,4:]==-1).all())
            del table
            with self.assertRaises(ValueError): open_table(out,cache,'different-bank',8)

    def test_full_image_support_evaluation(self):
        from .model import SRACSwin
        from .common import evaluate_front
        model=SRACSwin(dim=16,stride=4).eval()
        pre=torch.randn(1,3,64,64); post=torch.randn_like(pre)
        gt=torch.randint(0,4,(1,64,64)); loc=(gt>0).long()
        bank=dict(tokens=torch.randn(8,16),distribution=torch.rand(8,4).softmax(-1),
                  events=torch.ones(8,dtype=torch.long),images=torch.arange(8),
                  disasters=torch.ones(8,dtype=torch.long),geometry=torch.rand(8,5),
                  event_names=['query-earthquake','support-earthquake'])
        head=SupportClassifier(16,2).eval()
        result=evaluate_front(model,[(pre,post,loc,gt,['query-earthquake_0001'])],
                              torch.device('cpu'),head,bank)
        self.assertEqual(result['image_count'],1)
        self.assertEqual(result['metrics']['region']['count'],4096)
        self.assertEqual(result['support_coverage'],1)
        self.assertFalse(result['test_accessed'])

if __name__=='__main__': unittest.main()
