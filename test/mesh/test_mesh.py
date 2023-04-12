import unittest
import numpy as np


class TestMesh(unittest.TestCase):
    def test_marching_cubes(self):
        from elf.mesh import marching_cubes
        shape = (64,) * 3
        seg = (np.random.rand(*shape) > 0.8).astype("uint32")
        verts, faces, normals = marching_cubes(seg)
        self.assertGreater(len(verts), 0)
        self.assertGreater(len(faces), 0)
        self.assertGreater(len(normals), 0)

    def test_marching_cubes_smoothing(self):
        from elf.mesh import marching_cubes
        shape = (32,) * 3
        seg = (np.random.rand(*shape) > 0.8).astype("uint32")
        for smoothing_iterations in (1, 2, 3):
            verts, faces, normals = marching_cubes(seg, smoothing_iterations=smoothing_iterations)
            self.assertGreater(len(verts), 0)
            self.assertGreater(len(faces), 0)
            self.assertGreater(len(normals), 0)


if __name__ == '__main__':
    unittest.main()
