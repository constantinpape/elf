import unittest
import numpy as np
try:
    import affogato
except ImportError:
    affogato = None


@unittest.skipUnless(affogato, "Need affogato for mutex watershed functionality")
class TestMutexWatershed(unittest.TestCase):

    def test_mutex_watershed(self):
        from elf.segmentation.mutex_watershed import mutex_watershed
        shape = (10, 256, 256)
        aff_shape = (9,) + shape
        affs = np.random.rand(*aff_shape).astype('float32')

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
        weights = np.random.rand(len(uvs)).astype('float32')

        n_mutex_edges = 200 * n_nodes
        mutex_uvs = np.random.randint(0, n_nodes, size=(n_mutex_edges, 2))
        mutex_uvs = mutex_uvs[mutex_uvs[:, 0] != mutex_uvs[:, 1]]
        mutex_uvs = np.unique(mutex_uvs, axis=0)
        mutex_weights = np.random.rand(len(mutex_uvs)).astype('float32')

        # NOTE we may have duplicate edges in the weights and mutex weights, but that should be
        # fine, the one with the higher weight will just win

        node_labels = mutex_watershed_clustering(uvs, mutex_uvs, weights, mutex_weights)
        self.assertEqual(len(node_labels), n_nodes)
        self.assertFalse(np.allclose(node_labels, 0))

    # TODO remove expected failure once affogato is up-to-date
    @unittest.expectedFailure  # the affogato version on conda is not up-to date
    def test_mutex_watershed_with_seeds(self):
        from elf.segmentation.mutex_watershed import mutex_watershed_with_seeds
        shape = (10, 256, 256)
        aff_shape = (9,) + shape
        affs = np.random.rand(*aff_shape).astype('float32')
        # TODO non-trivial seeds
        seeds = np.zeros(shape, dtype='uint64')

        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-3, 0, 0], [0, -3, 0], [0, 0, -3],
                   [-9, 0, 0], [0, -9, 0], [0, 0, -9]]
        strides = [4, 4, 4]
        seg = mutex_watershed_with_seeds(affs, offsets, seeds, strides, True)
        self.assertEqual(seg.shape, shape)
        # make sure the segmentation is not trivial
        self.assertGreater(len(np.unique(seg)), 10)
        # TODO check that seeds were actually taken into account

    def test_blockwise_mutex_watershed(self):
        from elf.segmentation.mutex_watershed import blockwise_mutex_watershed


if __name__ == '__main__':
    unittest.main()
