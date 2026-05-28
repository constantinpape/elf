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

    def test_embedding_pca(self):
        from elf.segmentation.embeddings import embedding_pca

        embed = np.random.rand(8, 32, 32).astype("float32")
        out = embedding_pca(embed, n_components=3, as_rgb=True)
        self.assertEqual(out.shape, (3, 32, 32))
        self.assertEqual(out.dtype, np.uint8)

        out2 = embedding_pca(embed, n_components=2, as_rgb=False)
        self.assertEqual(out2.shape, (2, 32, 32))

    def test_edge_probabilities_from_embeddings(self):
        from elf.segmentation.embeddings import edge_probabilities_from_embeddings
        from elf.segmentation.features import compute_rag

        shape = (16, 16)
        rng = np.random.default_rng(0)
        seg = (rng.integers(0, 12, size=shape).cumsum(axis=0).cumsum(axis=1) % 13).astype("uint32")
        rag = compute_rag(seg)
        embed_dim = 4
        embeddings = rng.random((embed_dim,) + shape).astype("float32")
        probs = edge_probabilities_from_embeddings(embeddings, seg, rag, delta=0.5)
        self.assertEqual(len(probs), rag.numberOfEdges)
        self.assertTrue((probs >= 0).all())
        self.assertTrue((probs <= 1).all())

    def test_segment_mean_shift(self):
        from elf.segmentation.embeddings import segment_mean_shift
        rng = np.random.default_rng(1)
        embed = rng.standard_normal((3, 16, 16)).astype("float32")
        seg = segment_mean_shift(embed, bandwidth=1.5)
        self.assertEqual(seg.shape, (16, 16))
        self.assertEqual(seg.dtype, np.uint64)

    def test_segment_embeddings_mws(self):
        from elf.segmentation.embeddings import segment_embeddings_mws
        rng = np.random.default_rng(2)
        # 2D, 3 channel embedding
        embed = rng.standard_normal((3, 24, 24)).astype("float32")
        offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3]]
        seg = segment_embeddings_mws(embed, distance_type="l2", offsets=offsets)
        self.assertEqual(seg.shape, (24, 24))

    def test_segment_embeddings_mws_with_weight_function(self):
        from functools import partial
        from elf.segmentation.embeddings import segment_embeddings_mws, discriminative_loss_weight
        rng = np.random.default_rng(3)
        embed = rng.standard_normal((3, 24, 24)).astype("float32")
        offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3]]
        weight_function = partial(discriminative_loss_weight, delta=0.5)
        seg = segment_embeddings_mws(
            embed, distance_type="l2", offsets=offsets, weight_function=weight_function,
        )
        self.assertEqual(seg.shape, (24, 24))


if __name__ == '__main__':
    unittest.main()
