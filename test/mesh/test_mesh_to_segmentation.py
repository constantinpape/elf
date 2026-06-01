import os
import tempfile
import unittest

import numpy as np

try:
    import madcad  # noqa: F401
except ImportError:
    madcad = None


def _make_sphere(shape, radius):
    z, y, x = np.indices(shape)
    center = np.array(shape) // 2
    dist = np.sqrt(
        (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    )
    return (dist <= radius).astype("uint8")


@unittest.skipIf(madcad is None, "Needs madcad")
class TestMeshToSegmentation(unittest.TestCase):
    def _roundtrip(self, block_shape):
        from elf.mesh import marching_cubes
        from elf.mesh.io import write_obj
        from elf.mesh.mesh_to_segmentation import mesh_to_segmentation

        shape = (32, 32, 32)
        seg = _make_sphere(shape, radius=10)
        verts, faces, _ = marching_cubes(seg)

        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            obj_path = f.name
        try:
            write_obj(obj_path, verts, faces)
            recovered = mesh_to_segmentation(
                obj_path, shape=shape, block_shape=block_shape
            )
        finally:
            if os.path.exists(obj_path):
                os.remove(obj_path)

        # The recovered mask should agree with the original sphere on the
        # interior up to a thin surface tolerance.
        diff = recovered.astype(bool) ^ seg.astype(bool)
        # Number of disagreements should be small relative to the sphere volume.
        self.assertLess(diff.sum(), seg.sum() * 0.5)

    def test_mesh_to_segmentation_serial(self):
        self._roundtrip(block_shape=None)

    def test_mesh_to_segmentation_blockwise(self):
        self._roundtrip(block_shape=(16, 16, 16))


if __name__ == "__main__":
    unittest.main()
