"""The launcher must inspect imported code before starting model/data work."""
import re
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from . import reproduction, train


PACKAGE_DIR = Path(__file__).resolve().parent


class LauncherVersionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        launcher = (PACKAGE_DIR / 'run.sh').read_text(encoding='utf-8')
        snippets = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", launcher, re.DOTALL)
        if len(snippets) != 1:
            raise AssertionError('run.sh must contain exactly one Python version check')
        cls.guard = compile(snippets[0], str(PACKAGE_DIR / 'run.sh'), 'exec')

    def run_guard(self):
        output = StringIO()
        with mock.patch.object(sys, 'argv', ['-', str(PACKAGE_DIR)]), redirect_stdout(output):
            exec(self.guard, {'__name__': '__main__'})
        return output.getvalue()

    def test_current_package_passes_and_prints_imported_paths(self):
        output = self.run_guard()
        self.assertIn('VERSION CHECK PASSED: history_bounded_v1', output)
        self.assertIn('PYTHON: ' + sys.executable, output)
        for filename in ('train.py', 'reproduction.py'):
            self.assertIn('IMPORTED SOURCE: ' + str(PACKAGE_DIR / filename), output)
        self.assertEqual(output.count('SOURCE SHA256:'), 2)

    def test_missing_initial_gate_is_rejected(self):
        with mock.patch.object(train, 'initial_gate', None):
            with self.assertRaisesRegex(SystemExit, 'old train.py; missing initial_gate'):
                self.run_guard()

    def test_loaded_old_main_is_rejected_with_current_source_on_disk(self):
        def old_main(argv=None):
            return train.verify_baselines(None, None, argv)

        # Only the loaded function changes; the current source file stays intact.
        with mock.patch.object(train, 'main', old_main):
            with self.assertRaisesRegex(SystemExit, 'main does not use history_bounded_v1'):
                self.run_guard()

    def test_import_from_another_package_directory_is_rejected(self):
        stale_path = PACKAGE_DIR.parent / 'stale_package' / 'train.py'
        with mock.patch.object(train, '__file__', str(stale_path)):
            with self.assertRaisesRegex(SystemExit, 'imported source is outside this package directory'):
                self.run_guard()

    def test_incomplete_reproduction_module_is_rejected(self):
        with mock.patch.object(reproduction, 'json_safe_evidence', None):
            with self.assertRaisesRegex(SystemExit, 'incomplete reproduction.py; missing json_safe_evidence'):
                self.run_guard()


if __name__ == '__main__':
    unittest.main()
