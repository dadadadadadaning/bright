"""Pipeline failure propagation with subprocesses replaced by controlled children."""
import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run


class PipelineContracts(unittest.TestCase):
    def args(self, folder):
        return argparse.Namespace(output=str(Path(folder)/'new_run'), repo=folder, v3_root=folder,
                                  device='cuda:0', workers=0, seed=3407, topk=64, chunk_size=64,
                                  lr=1e-4, weight_decay=5e-3, protocol=None, command='full')

    def test_failed_smoke_never_launches_formal_training(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(folder)
            commands = []
            class FailedChild:
                pid = 123
                def __init__(self, command, **kwargs):
                    commands.append(command)
                    self.stdout = io.StringIO('controlled smoke failure\n')
                def wait(self):
                    return 7
            with patch.object(run, 'preflight', return_value={'complete': True}), \
                 patch.object(run.subprocess, 'Popen', FailedChild), \
                 patch('sys.stdout', new_callable=io.StringIO):
                with self.assertRaisesRegex(RuntimeError, 'smoke failed'):
                    run.full(args)
            self.assertEqual(len(commands), 1)
            state=json.loads((Path(args.output)/'workflow_state.json').read_text())
            self.assertFalse(state['complete'])
            self.assertFalse(state['training_complete'])
            self.assertEqual(state['stages']['smoke']['returncode'], 7)
            self.assertNotIn('train', state['stages'])
            self.assertTrue((Path(args.output)/'traceback.txt').is_file())

    def test_zero_exit_without_complete_smoke_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.args(folder)
            class SilentChild:
                pid=124
                def __init__(self, command, **kwargs):
                    self.stdout=io.StringIO('only progress, no final gate\n')
                def wait(self): return 0
            with patch.object(run,'preflight',return_value={'complete':True}), \
                 patch.object(run.subprocess,'Popen',SilentChild), \
                 patch('sys.stdout',new_callable=io.StringIO):
                with self.assertRaisesRegex(RuntimeError, 'complete summary'):
                    run.full(args)
            state=json.loads((Path(args.output)/'workflow_state.json').read_text())
            self.assertFalse(state['complete'])
            self.assertNotIn('train',state['stages'])

    def test_read_report_handles_failed_stage_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            p=root/'summary.json'
            text=json.dumps({'complete':False,'error':'controlled failure','smoke':True})
            p.write_text(text)
            with patch('sys.stdout',new_callable=io.StringIO) as capture:
                run.report(root)
            self.assertIn('controlled failure',capture.getvalue())
            self.assertEqual(p.read_text(),text)
            self.assertEqual([x.name for x in root.iterdir()],['summary.json'])


if __name__=='__main__':
    unittest.main()
