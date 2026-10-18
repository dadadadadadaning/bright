import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import chain_io as io


class SharedFileContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'code.py').write_text('original\n')
        io.write(self.root/'SHA256SUMS.json',{'code.py':io.digest(self.root/'code.py')})
        self.pin=io.digest(self.root/'SHA256SUMS.json')

    def test_pinned_package_returns_full_immutable_ledger(self):
        files=io.package_files(self.root,self.pin)
        self.assertEqual(set(files),{str(self.root/'code.py'),str(self.root/'SHA256SUMS.json')})

    def test_changed_code_rejected_without_overwrite(self):
        (self.root/'code.py').write_text('user changes\n')
        with self.assertRaises(ValueError):io.package_files(self.root,self.pin)
        self.assertEqual((self.root/'code.py').read_text(),'user changes\n')

    def test_unbound_python_rejected(self):
        (self.root/'extra.py').write_text('extra\n')
        with self.assertRaisesRegex(ValueError,'Unexpected Python'):io.package_files(self.root,self.pin)

    def test_manifest_path_escape_rejected(self):
        io.write(self.root/'SHA256SUMS.json',{'../escaped.py':'0'*64})
        with self.assertRaises(ValueError):io.package_files(self.root)

    def test_nonfinite_report_not_accepted_as_evidence(self):
        (self.root/'bad.json').write_text('{"value":NaN}')
        with self.assertRaises(ValueError):io.read(self.root/'bad.json')
        with self.assertRaises(ValueError):io.write(self.root/'new.json',{'value':float('nan')})
        self.assertFalse((self.root/'new.json').exists())


if __name__=='__main__':unittest.main()
