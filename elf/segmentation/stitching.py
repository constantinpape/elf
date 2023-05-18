import nifty.tools as nt
import numpy as np
from tqdm import trange

from .features import compute_rag, get_stitch_edges, project_node_labels_to_pixels
from .multicut import compute_edge_costs, multicut_decomposition


def stitch_segmentation(
    input_, segmentation_function,
    tile_shape, tile_overlap, beta=0.6,
    shape=None, with_background=True, n_threads=None
):
    """
    """

    shape = input_.shape if shape is None else shape
    blocking = nt.blocking([0] * len(shape), shape, tile_shape)

    id_offset = 0
    block_segs = []
    seg = np.zeros(shape, dtype="uint64")

    # TODO enable parallelisation
    # run tiled segmentation
    for block_id in trange(blocking.numberOfBlocks, desc="Run tiled segmentation"):
        block = blocking.getBlockWithHalo(block_id, tile_overlap)
        outer_bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))

        block_input = input_[outer_bb]
        block_seg = segmentation_function(block_input, block_id)
        if with_background:
            block_seg[block_seg != 0] += id_offset
        else:
            block_seg += id_offset
        id_offset = block_seg.max()

        inner_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
        local_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))

        seg[inner_bb] = block_seg[local_bb]
        block_segs.append(block_seg)

    # compute the region adjacency graph for the tiled segmentation
    # and the edges between block boundaries (stitch edges)
    rag = compute_rag(seg, n_threads=n_threads)
    stitch_edge_mask = get_stitch_edges(rag, seg, tile_shape, n_threads=n_threads)

    # TODO
    # compute the IOUs for all overlapping objects and derive merge costs
    costs = compute_edge_costs(edge_overlaps, beta=beta)

    # run multicut to get the segmentation result
    node_labels = multicut_decomposition(rag, costs)
    seg = project_node_labels_to_pixels(rag, node_labels, n_threads=n_threads)
    return seg
