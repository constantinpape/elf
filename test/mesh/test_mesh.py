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

    def test_marching_cubes_resolution(self):
        from elf.mesh import marching_cubes
        shape = (32,) * 3
        rng = np.random.default_rng(0)
        seg = (rng.random(shape) > 0.8).astype("uint32")

        verts_unit, _, _ = marching_cubes(seg, resolution=(1.0, 1.0, 1.0))
        resolution = (2.0, 1.0, 0.5)
        verts_scaled, _, _ = marching_cubes(seg, resolution=resolution)

        self.assertEqual(verts_unit.shape, verts_scaled.shape)
        expected = verts_unit * np.array(resolution)
        self.assertTrue(np.allclose(verts_scaled, expected))

    def test_smooth_mesh_tetrahedron(self):
        from elf.mesh.mesh import smooth_mesh

        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        normals = verts + 0.5
        faces = np.array(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
        )

        out_verts, out_normals = smooth_mesh(verts, normals, faces, iterations=1)

        self.assertEqual(out_verts.shape, verts.shape)
        self.assertEqual(out_normals.shape, normals.shape)
        self.assertEqual(out_verts.dtype, verts.dtype)
        self.assertEqual(out_normals.dtype, normals.dtype)

        # Every vertex of a tetrahedron is connected to every other vertex,
        # so one Jacobi-Laplacian iteration collapses each vertex onto the
        # mean of all four vertices (the centroid).
        centroid_verts = verts.mean(axis=0)
        centroid_normals = normals.mean(axis=0)
        for i in range(len(verts)):
            self.assertTrue(np.allclose(out_verts[i], centroid_verts))
            self.assertTrue(np.allclose(out_normals[i], centroid_normals))

    def test_smooth_mesh_no_op(self):
        from elf.mesh.mesh import smooth_mesh

        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        normals = verts + 0.5
        faces = np.array(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
        )

        out_verts, out_normals = smooth_mesh(verts, normals, faces, iterations=0)

        self.assertEqual(out_verts.shape, verts.shape)
        self.assertEqual(out_normals.shape, normals.shape)
        self.assertEqual(out_verts.dtype, verts.dtype)
        self.assertEqual(out_normals.dtype, normals.dtype)
        self.assertTrue(np.array_equal(out_verts, verts))
        self.assertTrue(np.array_equal(out_normals, normals))


if __name__ == '__main__':
    unittest.main()
