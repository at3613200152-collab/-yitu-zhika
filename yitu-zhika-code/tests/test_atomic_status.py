import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.capsicum_job import atomic_json


class AtomicStatusTests(unittest.TestCase):
    def test_transient_windows_reader_lock_recovers(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'status.json'
            atomic_json(target,{'epoch':1})
            replace = Path.replace
            calls = []
            def temporary_lock(source,destination):
                calls.append(1)
                if len(calls)<3:
                    self.assertEqual(json.loads(target.read_text()),{'epoch':1})
                    raise PermissionError('reader has no delete sharing')
                return replace(source,destination)
            with patch.object(Path,'replace',temporary_lock),patch('scripts.capsicum_job.time.sleep'):
                atomic_json(target,{'epoch':2})
            self.assertEqual(len(calls),3)
            self.assertEqual(json.loads(target.read_text()),{'epoch':2})
    def test_permanent_permission_failure_is_not_suppressed(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'status.json'
            atomic_json(target,{'epoch':1})
            with patch.object(Path,'replace',side_effect=PermissionError('denied')) as replace,patch('scripts.capsicum_job.time.sleep'):
                with self.assertRaises(PermissionError):
                    atomic_json(target,{'epoch':2})
                self.assertEqual(replace.call_count,7)
            self.assertEqual(json.loads(target.read_text()),{'epoch':1})


if __name__=='__main__':
    unittest.main()
