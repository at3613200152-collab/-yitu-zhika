import tempfile
from pathlib import Path
import unittest
import numpy as np
from src.data.hsi_manifest import parse_header,read_nir,split_days,training_scale


class HsiManifestTests(unittest.TestCase):
    def test_interleaves_and_byte_order(self):
        cube = np.arange(4*3*2,dtype=np.float32).reshape(4,3,2)
        with tempfile.TemporaryDirectory() as folder:
            for layout in ('bil','bip','bsq'):
                for order in (0,1):
                    hdr = Path(folder)/'x.hdr'
                    hdr.write_text('ENVI\nsamples = 3\nlines = 4\nbands = 2\ndata type = 4\nheader offset = 8\nbyte order = '+str(order)+'\ninterleave = '+layout+'\nacquisition date = 03-08-2021\nwavelength = {550,\n859.42}\n')
                    data = cube if layout=='bip' else cube.transpose(0,2,1) if layout=='bil' else cube.transpose(2,0,1)
                    raw = hdr.with_suffix('.dat')
                    raw.write_bytes(b'12345678'+data.astype('<f4' if order==0 else '>f4').tobytes())
                    h = parse_header(hdr)
                    self.assertEqual(h['capture_day'],'2021-08-03')
                    band,index,nm = read_nir(raw,h)
                    np.testing.assert_array_equal(band,cube[:,:,1])
                    self.assertEqual(index,1)
                    self.assertEqual(nm,859.42)
                    raw.write_bytes(raw.read_bytes()+b'extra')
                    with self.assertRaises(ValueError):
                        read_nir(raw,h)

    def test_training_scaling_excludes_holdouts(self):
        rows = [{'split':'train','nir':np.arange(64,dtype=np.float32).reshape(8,8)},
                {'split':'test','nir':np.full((8,8),1e9)}]
        first = training_scale(rows)
        rows[1]['nir'][:] = -1e9
        self.assertEqual(first,training_scale(rows))
        self.assertLess(first['high'],64)

    def test_dates_never_cross_splits(self):
        days = ['2021-08-03','2021-08-06','2021-08-07','2021-08-08','2021-08-10','2021-08-14','2021-10-09']
        mapping = split_days(days+days)
        self.assertEqual(mapping,split_days(list(reversed(days))))
        self.assertEqual(mapping['2021-08-08'],'test')
        self.assertEqual(mapping['2021-08-07'],'val')


if __name__=='__main__':
    unittest.main()
