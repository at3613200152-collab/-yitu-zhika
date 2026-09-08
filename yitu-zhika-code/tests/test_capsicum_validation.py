import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.capsicum_job import job_lock, validate_dataset


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='capsicum-pair-test-')
        self.root = Path(self.temp.name)
        for s, split in enumerate(('train', 'val', 'test')):
            for side in ('A', 'B'):
                (self.root / f'{split}_{side}').mkdir()
            for i in range(2):
                gray = np.full((16,16,3), 20+s*10+i, dtype=np.uint8)
                color = gray.copy()
                color[:,:,1] += 50
                name = f'frame_{split}_{i}-0000.png'
                Image.fromarray(color).save(self.root/f'{split}_A'/name)
                Image.fromarray(gray).save(self.root/f'{split}_B'/name)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        return validate_dataset(self.root, lambda **kw: None, expected_pairs=6)

    def test_exact_pairs_and_orientation(self):
        m = self.validate()
        self.assertEqual(m['counts'], {'train':2, 'val':2, 'test':2})
        self.assertTrue(all('_A' in p['rgb'] and '_B' in p['nir'] for p in m['pairs']))

    def test_missing_pair_cannot_fall_back_to_index(self):
        p = next((self.root/'train_B').glob('*.png'))
        p.rename(p.with_name('different.png'))
        with self.assertRaisesRegex(ValueError, 'pair exactly'):
            self.validate()

    def test_cross_split_content_leakage_rejected(self):
        a = next((self.root/'train_A').glob('*.png'))
        b = next((self.root/'val_A').glob('*.png'))
        shutil.copyfile(a, b)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            self.validate()

    def test_test_set_cannot_substitute_for_validation(self):
        (self.root/'val_A').rename(self.root/'missing_val_A')
        with self.assertRaisesRegex(ValueError, 'Missing explicit'):
            self.validate()

    @unittest.skipUnless(sys.platform == 'win32', 'Windows job lock')
    def test_singleton_prevents_duplicate_jobs(self):
        p = self.root/'job.lock'
        with job_lock(p):
            with self.assertRaisesRegex(RuntimeError, 'already holds'):
                with job_lock(p):
                    pass
        with job_lock(p):
            pass


if __name__ == '__main__':
    unittest.main()
