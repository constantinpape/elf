import unittest
import numpy as np

try:
    import z5py
except ImportError:
    z5py = None

try:
    import nifty
except ImportError:
    nifty = None

try:
    from cremi_tools.metrics import adapted_rand, voi
except ImportError:
    adapted_rand, voi = None, None


class TestMetrics(unittest.TestCase):
    # TODO implement download of example data in setUpClass
    url = ''
    seg_key = 'volumes/segmentation/multicut'
    gt_key = 'volumes/segmentation/groundtruth'

    # this takes quite a while, so we only do it on a sub-cutout
    # bb = np.s_[:30, :256, :256]
    # bb = np.s_[:]
    bb = np.s_[:100, :1024, :1024]

    # need to implement download first
    @unittest.skip
    # @unittest.skipUnless(voi and z5py and nifty, "Need cremi tools and z5py and nifty")
    def test_vi(self):
        from elf.evaluation import variation_of_information
        f = z5py.File(self.path)

        ds_gt = f[self.gt_key]
        ds_gt.n_threads = 4
        gt = ds_gt[self.bb]

        ds_seg = f[self.seg_key]
        ds_seg.n_threads = 4
        seg = ds_seg[self.bb]

        vi_s, vi_m = variation_of_information(seg, gt, ignore_gt=[0])
        vi_s_exp, vi_m_exp = voi(seg, gt)

        self.assertAlmostEqual(vi_s, vi_s_exp)
        self.assertAlmostEqual(vi_m, vi_m_exp)

    # need to implement download first
    @unittest.skip
    # @unittest.skipUnless(adapted_rand and z5py and nifty, "Need cremi tools and z5py and nifty")
    def test_ri(self):
        from elf.evaluation import rand_index
        f = z5py.File(self.path)

        ds_gt = f[self.gt_key]
        ds_gt.n_threads = 4
        gt = ds_gt[self.bb]

        ds_seg = f[self.seg_key]
        ds_seg.n_threads = 4
        seg = ds_seg[self.bb]

        ari, ri = rand_index(seg, gt, ignore_gt=[0])
        ari_exp = adapted_rand(seg, gt)

        self.assertAlmostEqual(ari, ari_exp)

    # need to implement download first
    @unittest.skip
    # @unittest.skipUnless(adapted_rand and z5py and nifty, "Need cremi tools and z5py and nifty")
    def test_cremi_score(self):
        from elf.evaluation import cremi_score
        f = z5py.File(self.path)

        ds_gt = f[self.gt_key]
        ds_gt.n_threads = 4
        gt = ds_gt[self.bb]

        ds_seg = f[self.seg_key]
        ds_seg.n_threads = 4
        seg = ds_seg[self.bb]

        vis, vim, ari, cs = cremi_score(seg, gt, ignore_gt=[0])

        ari_exp = adapted_rand(seg, gt)
        vis_exp, vim_exp = voi(seg, gt)

        cs_exp = np.sqrt(ari_exp * (vis_exp + vim_exp))

        self.assertAlmostEqual(ari, ari_exp)
        self.assertAlmostEqual(vis, vis_exp)
        self.assertAlmostEqual(vim, vim_exp)
        self.assertAlmostEqual(cs, cs_exp)

    # need to implement download first
    @unittest.skip
    # @unittest.skipUnless(adapted_rand and z5py and nifty, "Need cremi tools and z5py and nifty")
    def test_object_vi(self):
        from elf.evaluation import object_vi
        f = z5py.File(self.path)

        ds_gt = f[self.gt_key]
        ds_gt.n_threads = 4
        gt = ds_gt[self.bb]

        ds_seg = f[self.seg_key]
        ds_seg.n_threads = 4
        seg = ds_seg[self.bb]

        object_vis = object_vi(seg, gt, ignore_gt=[0])

        ids_exp = np.unique(gt)
        if 0 in gt:
            ids_exp = ids_exp[1:]
        ids = np.array(list(object_vis.keys()))
        ids = np.sort(ids)
        self.assertTrue(np.allclose(ids, ids_exp))

        for score in object_vis.values():
            vis, vim = score
            self.assertGreaterEqual(vis, 0)
            self.assertGreaterEqual(vim, 0)

    @unittest.skipUnless(adapted_rand and nifty, "Need cremi_tools and nifty")
    def test_ri_random_data(self):
        print("Blob")
        from elf.evaluation import rand_index
        shape = (256, 256)
        x = np.random.randint(0, 100, size=shape)
        y = np.random.randint(0, 100, size=shape)
        ari, ri = rand_index(x, y, ignore_gt=[0])
        ari_exp = adapted_rand(x, gy)
        self.assertAlmostEqual(ari, ari_exp)

    @unittest.skipUnless(voi and nifty, "Need cremi_tools and nifty")
    def test_vi_random_data(self):
        print("Blub")
        from elf.evaluation import rand_index
        shape = (256, 256)
        x = np.random.randint(0, 100, size=shape)
        y = np.random.randint(0, 100, size=shape)
        vi_s, vi_m = variation_of_information(x, y, ignore_gt=[0])
        vi_s_exp, vi_m_exp = voi(x, y)
        self.assertAlmostEqual(vi_s, vi_s_exp)
        self.assertAlmostEqual(vi_m, vi_m_exp)


if __name__ == '__main__':
    unittest.main()
