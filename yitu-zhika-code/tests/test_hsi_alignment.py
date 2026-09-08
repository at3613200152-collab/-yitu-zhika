import unittest
import numpy as np
from scripts.prepare_hsi_aligned import align_nir


class AlignmentTests(unittest.TestCase):
    def test_corrects_native_rotation_and_preserves_rgb(self):
        image=np.zeros((256,256),dtype=np.float32)
        image[20:70,90:125]=1
        image[170:190,45:120]=.5
        array=np.stack([image]*3+[np.rot90(image,1)])
        corrected=align_nir(array)
        np.testing.assert_array_equal(corrected[3],image)
        np.testing.assert_array_equal(corrected[:3],array[:3])
        self.assertFalse(np.array_equal(array[3],image))
        np.testing.assert_array_equal(array[3],np.rot90(image,1))


if __name__=='__main__':
    unittest.main()
