import os
import unittest
from glob import glob

import numpy as np

try:
    import nibabel
except ImportError:
    nibabel = None


@unittest.skipIf(nibabel is None, "Needs nibabel")
class TestNiftiWrapper(unittest.TestCase):

    def _check_data(self, expected_data, f):
        dset = f["data"]

        self.assertEqual(expected_data.shape, dset.shape)
        shape = dset.shape

        #  bounding boxes for testing sub-sampling
        bbs = [0, np.s_[:]]
        for i in range(dset.ndim):
            bbs.extend([
                tuple(slice(0, shape[i] // 2) if d == i else slice(None) for d in range(dset.ndim)),
                tuple(slice(shape[i] // 2, None) if d == i else slice(None) for d in range(dset.ndim))
            ])
        bbs.append(
            tuple(slice(shape[i] // 4, 3 * shape[i] // 4) for i in range(dset.ndim))
        )

        for bb in bbs:
            self.assertTrue(np.allclose(dset[bb], expected_data[bb]))

    def test_read_nifti(self):
        from elf.io import open_file
        from nibabel.testing import data_path

        paths = glob(os.path.join(data_path, "*.nii"))
        for path in paths:
            expected_data = np.asarray(nibabel.load(path).dataobj).T
            # the resampled image causes errors
            if os.path.basename(path).startswith("resampled"):
                continue
            with open_file(path, "r") as f:
                self._check_data(expected_data, f)

    def test_read_nifti_compressed(self):
        from elf.io import open_file
        from nibabel.testing import data_path

        paths = glob(os.path.join(data_path, "*.nii.gz"))
        for path in paths:
            expected_data = np.asarray(nibabel.load(path).dataobj).T
            with open_file(path, "r") as f:
                self._check_data(expected_data, f)


if __name__ == "__main__":
    unittest.main()
