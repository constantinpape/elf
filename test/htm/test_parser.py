import os
import tempfile
import unittest

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


@unittest.skipUnless(h5py is not None, "Need h5py")
class TestParseSimpleHtm(unittest.TestCase):

    def _write_plate(self, folder, wells, sites, channels, labels=None, shape=(8, 8)):
        """Write a minimal plate of HDF5 files into `folder`.

        Each well/site combination becomes `<well>_p<site>.h5` with one dataset per
        channel and (optionally) a `segmentation/<label>` group.
        """
        for well in wells:
            for site in range(sites):
                path = os.path.join(folder, f"{well}_p{site}.h5")
                with h5py.File(path, "w") as f:
                    for ch in channels:
                        f.create_dataset(ch, data=np.full(shape, fill_value=site, dtype="uint8"))
                    if labels:
                        seg = f.create_group("segmentation")
                        for lab in labels:
                            seg.create_dataset(lab, data=np.zeros(shape, dtype="uint16"))

    def test_images_only(self):
        from elf.htm import parse_simple_htm
        with tempfile.TemporaryDirectory() as tmp:
            wells = ["A1", "B2"]
            self._write_plate(tmp, wells=wells, sites=2, channels=["DAPI"])
            image_data, label_data = parse_simple_htm(tmp)
            self.assertIsNone(label_data)
            self.assertEqual(set(image_data.keys()), {"DAPI"})
            self.assertEqual(set(image_data["DAPI"].keys()), set(wells))
            for well in wells:
                self.assertEqual(len(image_data["DAPI"][well]), 2)
                for im in image_data["DAPI"][well]:
                    self.assertEqual(im.shape, (8, 8))

    def test_with_segmentation(self):
        from elf.htm import parse_simple_htm
        with tempfile.TemporaryDirectory() as tmp:
            self._write_plate(
                tmp, wells=["A1"], sites=1, channels=["DAPI"], labels=["cells"]
            )
            image_data, label_data = parse_simple_htm(tmp)
            self.assertEqual(set(image_data.keys()), {"DAPI"})
            self.assertIsNotNone(label_data)
            self.assertIn("segmentation/cells", label_data)
            self.assertEqual(len(label_data["segmentation/cells"]["A1"]), 1)
            self.assertEqual(label_data["segmentation/cells"]["A1"][0].shape, (8, 8))

    def test_exclude_names(self):
        from elf.htm import parse_simple_htm
        with tempfile.TemporaryDirectory() as tmp:
            self._write_plate(
                tmp,
                wells=["A1"],
                sites=1,
                channels=["DAPI", "GFP"],
                labels=["cells", "nuclei"],
            )
            image_data, label_data = parse_simple_htm(
                tmp, exclude_names=["GFP", "nuclei"]
            )
            self.assertEqual(set(image_data.keys()), {"DAPI"})
            self.assertEqual(set(label_data.keys()), {"segmentation/cells"})

    def test_pattern_filter(self):
        from elf.htm import parse_simple_htm
        with tempfile.TemporaryDirectory() as tmp:
            self._write_plate(tmp, wells=["A1"], sites=1, channels=["DAPI"])
            # Drop an unrelated file that must be ignored by the glob.
            with open(os.path.join(tmp, "notes.txt"), "w") as f:
                f.write("ignore me")
            image_data, _ = parse_simple_htm(tmp, pattern="*.h5")
            self.assertEqual(set(image_data.keys()), {"DAPI"})
            self.assertEqual(set(image_data["DAPI"].keys()), {"A1"})


if __name__ == "__main__":
    unittest.main()
