import os
import json
import unittest
from shutil import rmtree
from sys import platform

import numpy as np
import requests
from elf.transformation import affine as affine_utils


NGFF_EXAMPLES = {
    "0.4": {
        "yx": "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/yx.ome.zarr",
        "zyx": "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/zyx.ome.zarr",
        "tczyx": "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/tczyx.ome.zarr",
    }
}


@unittest.skipIf(platform == "win32", "Download fails on windows")
class TestNgff(unittest.TestCase):
    versions = list(NGFF_EXAMPLES.keys())
    tmp_folder = "./tmp"

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.tmp_folder, exist_ok=True)
        for version in cls.versions:
            version_folder = os.path.join(cls.tmp_folder, version)
            os.makedirs(version_folder, exist_ok=True)
            examples = NGFF_EXAMPLES[version]
            for name, example_url in examples.items():
                url = os.path.join(example_url, ".zattrs")
                out_path = os.path.join(version_folder, f"{name}.json")
                with requests.get(url) as r, open(out_path, "w") as f:
                    f.write(r.content.decode("utf8"))

    @classmethod
    def tearDownClass(cls):
        try:
            rmtree(cls.tmp_folder)
        except OSError:
            pass

    def test_ngff_to_native_simple(self):
        from elf.transformation import ngff_to_native
        for version in self.versions:
            for name in ("yx", "zyx"):
                for scale_level in (0, 2):
                    example = os.path.join(self.tmp_folder, version, f"{name}.json")
                with open(example) as f:
                    multiscales = json.load(f)
                trafo = ngff_to_native(multiscales, scale_level=scale_level)
                self.assertIsInstance(trafo, np.ndarray)
                exp_shape = (3, 3) if name == "yx" else (4, 4)
                self.assertEqual(trafo.shape, exp_shape)
                scale = affine_utils.scale_from_matrix(trafo)
                ds_trafos = multiscales["multiscales"][0]["datasets"][scale_level]["coordinateTransformations"]
                exp_scale = ds_trafos[0]["scale"]
                self.assertTrue(np.allclose(scale, exp_scale))

    def test_ngff_to_native_axes(self):
        from elf.transformation import ngff_to_native
        axes = "zyx"
        for version in self.versions:
            name = "tczyx"
            example = os.path.join(self.tmp_folder, version, f"{name}.json")
            with open(example) as f:
                multiscales = json.load(f)
            trafo = ngff_to_native(multiscales, axes=axes)
            self.assertIsInstance(trafo, np.ndarray)
            exp_shape = (4, 4)
            self.assertEqual(trafo.shape, exp_shape)
            scale = affine_utils.scale_from_matrix(trafo)
            exp_scale = multiscales["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][2:]
            self.assertTrue(np.allclose(scale, exp_scale))

    def test_native_to_ngff_2d(self):
        from elf.transformation import native_to_ngff
        scale, translation = np.random.rand(2), np.random.rand(2)
        matrix = affine_utils.affine_matrix_2d(scale=scale, translation=translation)
        trafo = native_to_ngff(matrix)

        trafo_parts = trafo["coordinateTransformations"]
        self.assertEqual(trafo_parts[0]["type"], "scale")
        self.assertTrue(np.allclose(scale, trafo_parts[0]["scale"]))
        self.assertEqual(trafo_parts[1]["type"], "translation")
        self.assertTrue(np.allclose(translation, trafo_parts[1]["translation"]))


if __name__ == "__main__":
    unittest.main()
