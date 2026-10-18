"""Offline contracts; full BRIGHT/CUDA is gated by the remote smoke stage."""
import unittest
from collections import defaultdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import torch
from io_utils import HERE,verify_release,old_modules
from new_members import priority,select,selected_rows,_parent_from_rows
from runtime_adapter import NewBaseRuntime,NewRuntime,NewFrozenSlotRuntime

class FreshBackendContracts(unittest.TestCase):
    def test_study_inputs_match_smoke_and_formal_artifacts(self):
        import study_backend
        with TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'repo'/'bda_benchmark').mkdir(parents=True)
            bank=root/'bank'; values=root/'values'
            bank.mkdir(); values.mkdir()
            protocol=root/'source.json'
            protocol.write_text(json.dumps({'format':study_backend.SOURCE_FORMAT,
                'release_sha256':'tested-release','input_paths':{},'input_fingerprints':{}}))
            args=SimpleNamespace(mode='smoke',topk=64,key_name='t128',chunk_size=32,
                lr=1e-4,weight_decay=.005,protocol=str(protocol),repo=str(root/'repo'),
                bank_root=str(bank),values=str(values))
            with patch.object(study_backend,'EXPECTED',{}), patch.object(study_backend,'verify_release',return_value='tested-release'):
                for smoke,mode in ((True,'smoke'),(False,'train')):
                    for folder,format_name in ((bank,study_backend.BANK_FORMAT),
                                               (values,'SRAC_SOFT_MEMORY_VALUES_V1')):
                        (folder/'summary.json').write_text(json.dumps({'complete':True,
                            'format':format_name,'smoke':smoke}))
                    args.mode=mode
                    self.assertTrue(study_backend.check_inputs(args))
                    args.mode='train' if smoke else 'smoke'
                    with self.assertRaisesRegex(ValueError,'smoke status differ'):
                        study_backend.check_inputs(args)

    def test_study_dependency_preflight_uses_requested_mode(self):
        import io_utils,study_backend
        config=dict(repo='/placeholder/repo',protocol='/placeholder/source.json',
            membership_root='/placeholder/members',bank_root='/placeholder/members',
            values='/placeholder/values',device='cuda:0',workers=4,seed=3407,
            key_name='t128',topk=64,chunk_size=32,lr=1e-4,weight_decay=.005,
            steps=5000,eval_every=500)
        bridge=SimpleNamespace(load_legacy=lambda _:object())
        common=SimpleNamespace(source_hash=lambda:study_backend.ORIGINAL_SOURCE_SHA)
        checked=[]
        for mode in ('smoke','train'):
            args=SimpleNamespace(mode=mode,config='/placeholder/config.json',helpers='/placeholder/helpers')
            with (patch.object(io_utils,'named',return_value=bridge),
                  patch.object(study_backend,'read',return_value=config),
                  patch.object(study_backend,'check_inputs',side_effect=checked.append),
                  patch.object(study_backend,'old_modules',return_value=(None,None,None,None,None,common)),
                  patch.object(study_backend,'digest',return_value='config-hash')):
                _,static=study_backend.dependencies(args)
            self.assertEqual(static['config']['mode'],mode)
            self.assertIs(static['config']['allow_smoke_values'],mode=='smoke')
        self.assertEqual([item.mode for item in checked],['smoke','train'])

    def test_values_smoke_constructs_namespace(self):
        import fresh_chain
        values=ModuleType('build_values')
        seen={}
        def build(args,runtime_factory):
            seen['args']=args
            seen['runtime_factory']=runtime_factory
            return 'called'
        values.run_build=build
        with TemporaryDirectory() as directory:
            with patch.object(fresh_chain,'old_modules',return_value=(None,)*6), patch.dict(sys.modules,{'build_values':values}):
                result=fresh_chain._run_stage(Path(directory),'values_smoke')
        self.assertEqual(result,'called')
        self.assertIsInstance(seen['args'],SimpleNamespace)
        self.assertEqual(seen['args'].command,'build')
        self.assertEqual(seen['args'].limit_images,4)
        self.assertIs(seen['runtime_factory'],NewFrozenSlotRuntime)

    def row(self,image,event,region,cls):
        return dict(images=image,events=event,regions=region,original_rows=image*1000000+region,
            t128=torch.ones(128),f256=torch.ones(256),
            distribution=torch.nn.functional.one_hot(torch.tensor(cls),4).float(),
            mass=1.,label_mass=1.,event=str(event),**{'class':cls},
            image_id=image,region_id=region,priority=priority(3407,image,region))

    def test_separate_events_and_classes_with_bounded_cells(self):
        heaps=defaultdict(list)
        candidates=[self.row(0,0,r,1) for r in range(10)] + [self.row(1,1,2,2),self.row(2,2,3,3)]
        for candidate in reversed(candidates): select(heaps,candidate,1)
        rows=selected_rows(heaps,3411)
        self.assertEqual(len(rows),3)
        self.assertEqual({(r['event'],r['class']) for r in rows},{('0',1),('1',2),('2',3)})
        self.assertEqual([r['region_id'] for r in rows if r['event']=='0'],
                         [min(candidates[:10],key=lambda r:r['priority'])['region_id']])
        bank=_parent_from_rows(rows,['0','1','2'])
        self.assertEqual(len(bank['member_ids']),3)
        self.assertTrue(bool(bank['eligible'].all()))

    def test_runtime_uses_new_base_beneath_original_slot_reader(self):
        self.assertIn(NewBaseRuntime,NewRuntime.__mro__)
        self.assertIn(NewBaseRuntime,NewFrozenSlotRuntime.__mro__)
        self.assertLess(NewRuntime.__mro__.index(NewBaseRuntime),NewRuntime.__mro__.index(old_modules()[2].Runtime))

    def test_release_and_original_code_bound(self):
        self.assertEqual(len(verify_release()),64)
        _,_,_,_,_,common=old_modules()
        self.assertEqual(common.source_hash(),'b84e2867a4117911b8279e2f97ccf9c5fbd021a7dc539f59e6bbb06396546142')

if __name__=='__main__':unittest.main()
