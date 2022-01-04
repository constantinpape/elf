import unittest
import numpy as np


class TestMutexWatershed(unittest.TestCase):

    def test_mutex_watershed(self):
        from elf.segmentation.mutex_watershed import mutex_watershed
        shape = (10, 256, 256)
        aff_shape = (9,) + shape
        affs = np.random.rand(*aff_shape).astype("float32")

        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-3, 0, 0], [0, -3, 0], [0, 0, -3],
                   [-9, 0, 0], [0, -9, 0], [0, 0, -9]]
        strides = [4, 4, 4]
        seg = mutex_watershed(affs, offsets, strides, True)
        self.assertEqual(seg.shape, shape)
        # make sure the segmentation is not trivial
        self.assertGreater(len(np.unique(seg)), 10)

    def test_mutex_watershed_clustering(self):
        from elf.segmentation.mutex_watershed import mutex_watershed_clustering

        n_nodes = 1000

        n_edges = 100 * n_nodes
        uvs = np.random.randint(0, n_nodes, size=(n_edges, 2))
        uvs = uvs[uvs[:, 0] != uvs[:, 1]]
        uvs = np.unique(uvs, axis=0)
        weights = np.random.rand(len(uvs)).astype("float32")

        n_mutex_edges = 200 * n_nodes
        mutex_uvs = np.random.randint(0, n_nodes, size=(n_mutex_edges, 2))
        mutex_uvs = mutex_uvs[mutex_uvs[:, 0] != mutex_uvs[:, 1]]
        mutex_uvs = np.unique(mutex_uvs, axis=0)
        mutex_weights = np.random.rand(len(mutex_uvs)).astype("float32")

        # NOTE we may have duplicate edges in the weights and mutex weights, but that should be
        # fine, the one with the higher weight will just win

        node_labels = mutex_watershed_clustering(uvs, mutex_uvs, weights, mutex_weights)
        self.assertEqual(len(node_labels), n_nodes)
        self.assertFalse(np.allclose(node_labels, 0))

    def test_mutex_watershed_with_seeds(self):
        from elf.segmentation.mutex_watershed import mutex_watershed_with_seeds
        shape = (10, 256, 256)
        aff_shape = (9,) + shape
        affs = np.random.rand(*aff_shape).astype("float32")
        # TODO non-trivial seeds
        seeds = np.zeros(shape, dtype="uint64")

        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-3, 0, 0], [0, -3, 0], [0, 0, -3],
                   [-9, 0, 0], [0, -9, 0], [0, 0, -9]]
        strides = [4, 4, 4]
        seg = mutex_watershed_with_seeds(affs, offsets, seeds, strides, True)
        self.assertEqual(seg.shape, shape)
        # make sure the segmentation is not trivial
        self.assertGreater(len(np.unique(seg)), 10)
        # TODO check that seeds were actually taken into account

    def test_semantic_mutex_watershed(self):
        from elf.segmentation.mutex_watershed import semantic_mutex_watershed
        shape = (512, 512)
        aff_shape = (6,) + shape
        affs = np.random.rand(*aff_shape).astype("float32")
        sem_shape = (4,) + shape
        sem = np.random.rand(*sem_shape).astype("float32")
        offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3], [-9, 0], [0, -9]]
        strides = [4, 4]

        seg, sem = semantic_mutex_watershed(affs, sem, offsets, strides)
        self.assertEqual(seg.shape, shape)
        self.assertEqual(sem.shape, shape)
        self.assertGreater(len(np.unique(seg)), 10)
        self.assertTrue(np.allclose(np.unique(sem), np.array([0, 1, 2, 3])))

    def test_semantic_mutex_watershed_clustering(self):
        from elf.segmentation.mutex_watershed import semantic_mutex_watershed_clustering
        n_nodes = 789
        uvs = np.concatenate([
            np.arange(0, n_nodes - 1)[:, None], np.arange(1, n_nodes)[:, None]
        ], axis=1).astype("uint64")

        mutex_uvs = np.random.randint(0, n_nodes, size=(500, 2)).astype("uint64")
        keep_mutex = np.abs(mutex_uvs[:, 0] - mutex_uvs[:, 1]) > 1
        mutex_uvs = mutex_uvs[keep_mutex]

        weights = np.random.rand(len(uvs)).astype("float32")
        mutex_weights = np.random.rand(len(mutex_uvs)).astype("float32")

        n_classes = 4
        semantic_uts = np.concatenate([
            np.arange(n_nodes)[:, None], np.random.randint(0, n_classes, size=n_nodes)[:, None]
        ], axis=1)
        semantic_weights = np.random.rand(n_nodes).astype("float32")

        instances, semantic = semantic_mutex_watershed_clustering(
            uvs, mutex_uvs, weights, mutex_weights, semantic_uts, semantic_weights,
            n_nodes=n_nodes
        )
        self.assertEqual(len(instances), n_nodes)
        self.assertEqual(len(semantic), n_nodes)
        self.assertGreater(len(np.unique(instances)), 10)
        self.assertTrue(np.allclose(np.unique(semantic), np.arange(n_classes)))

    def test_blockwise_mutex_watershed(self):
        # from elf.segmentation.mutex_watershed import blockwise_mutex_watershed
        pass


if __name__ == "__main__":
    unittest.main()
