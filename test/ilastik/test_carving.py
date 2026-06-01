import os
import tempfile
import unittest

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


@unittest.skipUnless(h5py is not None, "Need h5py")
class TestCarving(unittest.TestCase):

    def _make_watershed(self):
        """Build a 4x4x4 uint32 watershed where each 2x2x2 octant is one label.

        Returns a (4, 4, 4) array with eight unique ids in [0, 7].
        """
        ws = np.zeros((4, 4, 4), dtype="uint32")
        label = 0
        for z in (0, 2):
            for y in (0, 2):
                for x in (0, 2):
                    ws[z:z + 2, y:y + 2, x:x + 2] = label
                    label += 1
        return ws

    def _write_project(self, path, ws, objects):
        """Write a minimal ilastik carving HDF5 project file.

        Args:
            path: Destination path.
            ws: 3D uint32 watershed (will be transposed before write to match
                the on-disk layout that ``load_watershed_and_rag`` expects).
            objects: Mapping of object name -> 1D array of supervoxel ids.
        """
        with h5py.File(path, "w") as f:
            f.create_dataset("preprocessing/graph/labels", data=ws.T)
            obj_grp = f.create_group("carving/objects")
            for name, sv in objects.items():
                obj_grp.create_dataset(f"{name}/sv", data=np.asarray(sv, dtype="uint32"))

    def test_get_object_names(self):
        from elf.ilastik.carving import get_object_names
        ws = self._make_watershed()
        objects = {"obj_a": [0, 1], "obj_b": [3, 7]}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.ilp")
            self._write_project(path, ws, objects)
            names = get_object_names(path)
            self.assertEqual(set(names), {"obj_a", "obj_b"})

    def test_load_watershed_and_rag(self):
        from elf.ilastik.carving import load_watershed_and_rag
        ws = self._make_watershed()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.ilp")
            self._write_project(path, ws, {"obj_a": [0]})
            ws_loaded, rag = load_watershed_and_rag(path)
            np.testing.assert_array_equal(ws_loaded, ws)
            self.assertEqual(rag.number_of_nodes, int(ws.max() + 1))

    def test_export_object_without_precomputed_rag(self):
        from elf.ilastik.carving import export_object
        ws = self._make_watershed()
        sv = np.array([0, 3], dtype="uint32")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.ilp")
            self._write_project(path, ws, {"obj_a": sv})
            seg = export_object(path, "obj_a")
            self.assertEqual(seg.shape, ws.shape)
            expected = np.isin(ws, sv).astype(seg.dtype)
            np.testing.assert_array_equal(seg, expected)

    def test_export_object_with_precomputed_ws_rag(self):
        from elf.ilastik.carving import export_object, load_watershed_and_rag
        ws = self._make_watershed()
        sv = np.array([1, 6], dtype="uint32")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.ilp")
            self._write_project(path, ws, {"obj_a": sv})
            ws_loaded, rag = load_watershed_and_rag(path)
            seg_precomp = export_object(path, "obj_a", ws=ws_loaded, rag=rag)
            seg_no_precomp = export_object(path, "obj_a")
            np.testing.assert_array_equal(seg_precomp, seg_no_precomp)

    def test_export_multiple_objects(self):
        from elf.ilastik.carving import export_object, load_watershed_and_rag
        ws = self._make_watershed()
        sv_a = np.array([0, 1], dtype="uint32")
        sv_b = np.array([5, 7], dtype="uint32")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.ilp")
            self._write_project(path, ws, {"obj_a": sv_a, "obj_b": sv_b})
            ws_loaded, rag = load_watershed_and_rag(path)
            seg_a = export_object(path, "obj_a", ws=ws_loaded, rag=rag)
            seg_b = export_object(path, "obj_b", ws=ws_loaded, rag=rag)
            np.testing.assert_array_equal(seg_a, np.isin(ws, sv_a).astype(seg_a.dtype))
            np.testing.assert_array_equal(seg_b, np.isin(ws, sv_b).astype(seg_b.dtype))
            self.assertFalse(np.array_equal(seg_a, seg_b))

    def test_export_object_partial_args_raises(self):
        from elf.ilastik.carving import export_object, load_watershed_and_rag
        ws = self._make_watershed()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.ilp")
            self._write_project(path, ws, {"obj_a": [0]})
            ws_loaded, rag = load_watershed_and_rag(path)
            with self.assertRaises(ValueError):
                export_object(path, "obj_a", ws=ws_loaded, rag=None)
            with self.assertRaises(ValueError):
                export_object(path, "obj_a", ws=None, rag=rag)

    def test_export_all_objects(self):
        from elf.ilastik.carving import export_all_objects, export_object
        ws = self._make_watershed()
        objects = {"obj_a": [0, 1], "obj_b": [5, 7]}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.ilp")
            self._write_project(path, ws, objects)
            result = export_all_objects(path)
            self.assertEqual(set(result.keys()), set(objects.keys()))
            for name in objects:
                np.testing.assert_array_equal(result[name], export_object(path, name))

    def test_export_all_objects_postprocess(self):
        from elf.ilastik.carving import export_all_objects
        ws = self._make_watershed()
        objects = {"obj_a": [0], "obj_b": [7]}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.ilp")
            self._write_project(path, ws, objects)
            result = export_all_objects(path, postprocess=lambda seg: seg.astype("uint8"))
            for name in objects:
                self.assertEqual(result[name].dtype, np.dtype("uint8"))


if __name__ == "__main__":
    unittest.main()
