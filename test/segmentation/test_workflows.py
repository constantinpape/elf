import unittest
import numpy as np


# TODO test more workflows, test more options
class TestWorkflow(unittest.TestCase):

    def test_simple_multicut_workflow(self):
        from elf.segmentation.workflows import simple_multicut_workflow
        shape = (64, 64, 64)
        input_ = np.random.rand(*shape).astype('float32')

        # test with 2d ws computation
        seg = simple_multicut_workflow(input_, use_2dws=True)
        self.assertEqual(seg.shape, input_.shape)

        # test with 3d ws computation
        seg = simple_multicut_workflow(input_, use_2dws=False)
        self.assertEqual(seg.shape, input_.shape)

    def test_lifted_multicut_from_segmentation_workflow(self):
        from elf.segmentation.workflows import lifted_multicut_from_segmentation_workflow
        shape = (64, 64, 64)
        input_ = np.random.rand(*shape).astype('float32')
        input_seg = np.random.randint(0, 10, size=shape).astype('uint32')

        seg = lifted_multicut_from_segmentation_workflow(input_, input_seg,
                                                         use_2dws=True, overlap_threshold=0.1,
                                                         same_segment_cost=4., different_segment_cost=-4.,
                                                         graph_depth=4)
        self.assertEqual(seg.shape, input_.shape)


if __name__ == '__main__':
    unittest.main()
