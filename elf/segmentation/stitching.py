import multiprocessing
from concurrent import futures

import nifty.tools as nt
import numpy as np
import vigra
from nifty.ground_truth import overlap
from tqdm import trange, tqdm

from .features import compute_rag, project_node_labels_to_pixels
from .multicut import compute_edge_costs, multicut_decomposition


def stitch_segmentation(
    input_, segmentation_function,
    tile_shape, tile_overlap, beta=0.5,
    shape=None, with_background=True, n_threads=None,
    return_before_stitching=False, verbose=True,
):
    """
    """

    shape = input_.shape if shape is None else shape
    ndim = len(shape)
    blocking = nt.blocking([0] * ndim, shape, tile_shape)

    id_offset = 0
    block_segs = []
    seg = np.zeros(shape, dtype="uint64")

    n_blocks = blocking.numberOfBlocks
    # TODO enable parallelisation
    # run tiled segmentation
    for block_id in trange(n_blocks, desc="Run tiled segmentation", disable=not verbose):
        block = blocking.getBlockWithHalo(block_id, list(tile_overlap))
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
    seg_ids = np.unique(seg)
    rag = compute_rag(seg, n_threads=n_threads)

    # we initialize the edge disaffinities with a high value (corresponding to a low overlap)
    # so that merging things that are not on the edge is very unlikely
    # but not completely impossible in case it is needed for a consistent solution
    edge_disaffinties = np.full(rag.numberOfEdges, 0.9, dtype="float32")

    def _compute_overlaps(block_id):

        # for each axis, load the face with the lower block neighbor and compute the object overlaps
        for axis in range(ndim):
            ngb_id = blocking.getNeighborId(block_id, axis, lower=True)
            if ngb_id == -1:
                continue

            this_block = blocking.getBlockWithHalo(block_id, list(tile_overlap))
            ngb_block = blocking.getBlockWithHalo(ngb_id, list(tile_overlap))

            # load the full block segmentations
            this_seg, ngb_seg = block_segs[block_id], block_segs[ngb_id]

            # get the global coordinates of the block face
            face = tuple(slice(beg_out, end_out) if d != axis else slice(beg_out, beg_in + tile_overlap[d])
                         for d, (beg_out, end_out, beg_in) in enumerate(zip(this_block.outerBlock.begin,
                                                                            this_block.outerBlock.end,
                                                                            this_block.innerBlock.begin)))
            # map to the two local face coordinates
            this_face_bb = tuple(
                slice(fa.start - offset, fa.stop - offset) for fa, offset in zip(face, this_block.outerBlock.begin)
            )
            ngb_face_bb = tuple(
                slice(fa.start - offset, fa.stop - offset) for fa, offset in zip(face, ngb_block.outerBlock.begin)
            )

            # load the two segmentations for the face
            this_face = this_seg[this_face_bb]
            ngb_face = ngb_seg[ngb_face_bb]
            assert this_face.shape == ngb_face.shape

            # compute the object overlaps
            overlap_comp = overlap(this_face, ngb_face)
            this_ids = np.unique(this_face)
            overlaps = {this_id: overlap_comp.overlapArraysNormalized(this_id, sorted=False) for this_id in this_ids}
            overlap_ids = {this_id: ovlps[0] for this_id, ovlps in overlaps.items()}
            overlap_values = {this_id: ovlps[1] for this_id, ovlps in overlaps.items()}
            overlap_uv_ids = np.array([
                [this_id, ovlp_id] for this_id, ovlp_ids in overlap_ids.items() for ovlp_id in ovlp_ids
            ])
            overlap_values = np.array([ovlp for ovlps in overlap_values.values() for ovlp in ovlps], dtype="float32")
            assert len(overlap_uv_ids) == len(overlap_values)

            # - get the edge ids
            # - exclude invalid edge
            # - set the global edge disaffinities to 1 - overlap

            # we might have ids in the overlaps that are not in the final seg, these need to be filtered
            valid_uv_ids = np.isin(overlap_uv_ids, seg_ids).all(axis=1)
            if valid_uv_ids.sum() == 0:
                continue
            overlap_uv_ids, overlap_values = overlap_uv_ids[valid_uv_ids], overlap_values[valid_uv_ids]
            assert len(overlap_uv_ids) == len(overlap_values)

            edge_ids = rag.findEdges(overlap_uv_ids)
            valid_edges = edge_ids != -1
            if valid_edges.sum() == 0:
                continue
            edge_ids, overlap_values = edge_ids[valid_edges], overlap_values[valid_edges]
            assert len(edge_ids) == len(overlap_values)

            edge_disaffinties[edge_ids] = (1.0 - overlap_values)

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(
            _compute_overlaps, range(n_blocks)), total=n_blocks, desc="Compute object overlaps", disable=not verbose
        ))

    # if we have background set all the edges that are connecting 0 to another element
    # to be very unlikely
    if with_background:
        uv_ids = rag.uvIds()
        bg_edges = rag.findEdges(uv_ids[(uv_ids == 0).any(axis=1)])
        edge_disaffinties[bg_edges] = 0.99
    costs = compute_edge_costs(edge_disaffinties, beta=beta)

    # run multicut to get the segmentation result
    node_labels = multicut_decomposition(rag, costs)
    seg_stitched = project_node_labels_to_pixels(rag, node_labels, n_threads=n_threads)

    if with_background:
        vigra.analysis.relabelConsecutive(seg_stitched, out=seg_stitched, start_label=1, keep_zeros=True)
    else:
        vigra.analysis.relabelConsecutive(seg_stitched, out=seg_stitched, start_label=1, keep_zeros=False)

    if return_before_stitching:
        return seg_stitched, seg
    return seg_stitched
