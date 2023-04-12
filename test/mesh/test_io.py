import os
import unittest

import numpy as np
from elf.mesh import marching_cubes
try:
    import pywavefront
except ImportError:
    pywavefront = None


class TestMeshIo(unittest.TestCase):
    tmp_path = "./tmp.mesh"

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_obj(self):
        from elf.mesh.io import read_obj, write_obj

        shape = (64,) * 3
        seg = (np.random.rand(*shape) > 0.8).astype("uint32")
        verts, faces, normals = marching_cubes(seg)

        write_obj(self.tmp_path, verts, faces, normals)
        deverts, defaces, denormals, _ = read_obj(self.tmp_path)

        self.assertTrue(np.allclose(verts, deverts))
        self.assertTrue(np.allclose(faces, defaces))
        self.assertTrue(np.allclose(normals, denormals))

    def test_ply(self):
        from elf.mesh.io import read_ply, write_ply

        shape = (64,) * 3
        seg = (np.random.rand(*shape) > 0.8).astype("uint32")
        verts, faces, _ = marching_cubes(seg)

        write_ply(self.tmp_path, verts, faces)
        deverts, defaces = read_ply(self.tmp_path)

        self.assertTrue(np.allclose(verts, deverts))
        self.assertTrue(np.allclose(faces, defaces))

    @unittest.skipIf(pywavefront is None, "Needs pywavefront")
    def test_pywavefront(self):
        from elf.mesh.io import write_obj

        shape = (64,) * 3
        seg = (np.random.rand(*shape) > 0.8).astype("uint32")
        verts, faces, normals = marching_cubes(seg)
        write_obj(self.tmp_path, verts, faces, normals)

        scene = pywavefront.Wavefront(self.tmp_path)
        deverts = np.array(scene.vertices)
        self.assertTrue(np.allclose(verts, deverts))


if __name__ == '__main__':
    unittest.main()
