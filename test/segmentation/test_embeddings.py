import unittest
import numpy as np


class TestEmbeddings(unittest.TestCase):
    def test_embeddings_to_affinities(self):
        from elf.segmentation.embeddings import embeddings_to_affinities

        shape = (8, 128, 128)
        x = np.random.rand(*shape)
        offsets = [[-1, 0], [0, -1],
                   [-3, 0], [0, -3],
                   [-9, 0], [0, -9]]
        delta = .5
        affs = embeddings_to_affinities(x, offsets, delta)
        self.assertTrue((affs >= 0).all())
        self.assertTrue((affs <= 1).all())
        self.assertGreater(affs.sum(), 0)


if __name__ == '__main__':
    unittest.main()
