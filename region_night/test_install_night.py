import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from install_night import FILES, install


class DownloadContracts(unittest.TestCase):
    def fixture(self, corrupt=False, extra=False):
        files = {name: ('# verified file ' + name).encode() for name in FILES}
        ledger = {name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
        if extra:
            ledger['../escape.py'] = 'a'*64
        def fetch(url):
            name = url.rsplit('/',1)[1]
            if name == 'SHA256SUMS.json':
                return json.dumps(ledger).encode()
            return b'corrupt' if corrupt else files[name]
        return fetch

    def test_atomic_verified_install_and_repeat_does_not_modify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'tools'
            install('a'*40,root,self.fixture())
            before = (root/'night_run.py').stat().st_mtime_ns
            install('a'*40,root,self.fixture())
            self.assertEqual(before,(root/'night_run.py').stat().st_mtime_ns)
            self.assertEqual(json.loads((root/'release.json').read_text())['commit'],'a'*40)

    def test_checksum_failure_never_leaves_runnable_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'tools'
            with self.assertRaises(ValueError):
                install('a'*40,root,self.fixture(corrupt=True))
            self.assertFalse(root.exists())

    def test_refuses_path_injection_and_noncommit_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'tools'
            with self.assertRaises(ValueError):
                install('main',root,self.fixture())
            with self.assertRaises(ValueError):
                install('a'*40,root,self.fixture(extra=True))
            self.assertFalse(root.exists())

    def test_existing_modified_source_or_other_commit_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'tools'
            install('a'*40,root,self.fixture())
            with self.assertRaises(FileExistsError):
                install('b'*40,root,self.fixture())
            (root/'night_run.py').write_text('running modified source')
            with self.assertRaises(FileExistsError):
                install('a'*40,root,self.fixture())
            self.assertEqual((root/'night_run.py').read_text(),'running modified source')


if __name__ == '__main__':
    unittest.main()
