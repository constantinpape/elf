import multiprocessing
from concurrent import futures

import numpy as np
import nifty.tools as nt

from affogato.segmentation import compute_mws_segmentation
from vigra.analysis import relabelConsecutive

from . import features
from . import multicut


def compute_stitch_edges(rag, segmentation, blocking, with_mask=False):
    """@private
    """

    def edges_from_face(lower_block_id, upper_block_id, axis):
        lower_block = blocking.getBlock(lower_block_id)
        upper_block = blocking.getBlock(upper_block_id)

        lower_face = tuple(slice(beg, end) if d != axis else slice(end - 1, end)
                           for d, (beg, end) in enumerate(zip(lower_block.begin,
                                                              lower_block.end)))
        upper_face = tuple(slice(beg, end) if d != axis else slice(beg, beg + 1)
                           for d, (beg, end) in enumerate(zip(upper_block.begin,
                                                              upper_block.end)))
        lower_seg = segmentation[lower_face].flatten()
        upper_seg = segmentation[upper_face].flatten()

        edges = np.concatenate([lower_seg[:, None], upper_seg[:, None]], axis=1)
        edges = np.unique(edges, axis=0)

        # NOTE if we have a mask, we might get the id pair [0, 0], which
        # is not a valid edge in the rag
        if with_mask:
            edge_mask = (edges == 0).all(axis=1)
            edges = edges[~edge_mask]
            if edges.size == 0:
                return None

        edge_ids = rag.findEdges(edges)
        assert (edge_ids != -1).all()
        return edge_ids

    stitch_edges = []
    n_blocks = blocking.numberOfBlocks
    # parallelize ?
    for block_id in range(n_blocks):
        for axis in range(3):
            ngb_id = blocking.getNeighborId(block_id, axis, lower=True)
            if ngb_id == -1:
                continue

            this_stitch_edges = edges_from_face(ngb_id, block_id, axis)
            if this_stitch_edges is not None:
                stitch_edges.append(this_stitch_edges)

    stitch_edge_mask = np.zeros(rag.numberOfEdges, dtype="bool")

    if len(stitch_edges) > 0:
        stitch_edges = np.concatenate(stitch_edges, axis=0)
        stitch_edge_mask[stitch_edges] = 1

    return stitch_edge_mask


# TODO different mutex merging schemes?
# TODO enable verbosity via tqdm?
# TODO more blockwise mws options
def blockwise_mws_impl(affs, offsets, strides, block_shape,
                       randomize_strides=False, mask=None, noise_level=0,
                       solver_name="kernighan-lin", beta0=.75, beta1=.5,
                       n_threads=None):
    """@private
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads

    # allocate the segmentation
    ndim = affs.ndim - 1
    shape = affs.shape[1:]
    segmentation = np.zeros(shape, dtype="uint64")

    # TODO with halo ?
    # 1.) run mutex watersheds on the blocks in parallel
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    n_blocks = blocking.numberOfBlocks

    def mws_block(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        bb_affs = (slice(None),) + bb
        affs_ = affs[bb_affs].copy()  # we need to copy here to leave the original affs unchanged
        mask_ = None if mask is None else mask[bb]

        if noise_level > 0:
            affs_ += noise_level * np.random.rand(*affs_.shape)
        affs_[:ndim] *= -1
        affs_[:ndim] += 1
        seg = compute_mws_segmentation(affs_, offsets,
                                       number_of_attractive_channels=ndim,
                                       strides=strides, mask=mask_,
                                       randomize_strides=randomize_strides)
        max_id = relabelConsecutive(seg, out=seg, start_label=1, keep_zeros=mask is not None)[1]
        segmentation[bb] = seg
        return max_id

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(mws_block, block_id) for block_id in range(n_blocks)]
        id_offsets = [t.result() for t in tasks]

    # 2.) apply id_offsets to the blocks

    # compute the block offsets
    id_offsets = np.roll(id_offsets, 1)
    id_offsets[0] = 0
    id_offsets = np.cumsum(id_offsets)

    def apply_offset(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        if mask is None:
            segmentation[bb] += id_offsets[block_id]
        else:
            mask_ = mask[bb]
            segmentation[bb][mask_] += id_offsets[block_id]

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(apply_offset, block_id)
                 for block_id in range(n_blocks)]
        [t.result() for t in tasks]

    # 3.) compute rag, features and block edges (if specified)
    rag = features.compute_rag(segmentation, n_threads=n_threads)
    edge_feats = features.compute_affinity_features(rag, affs, offsets,
                                                    n_threads=n_threads)
    costs, sizes = edge_feats[:, 0], edge_feats[:, -1]
    # normalize the sizes by the number of affinities
    sizes /= len(offsets)

    if np.isclose(beta0, beta1):
        costs = multicut.transform_probabilities_to_costs(costs, edge_sizes=sizes)
    else:
        stitch_edges = compute_stitch_edges(rag, segmentation, blocking, with_mask=mask is not None)
        costs[stitch_edges] = multicut.transform_probabilities_to_costs(costs[stitch_edges],
                                                                        beta=beta1,
                                                                        edge_sizes=sizes[stitch_edges])
        costs[~stitch_edges] = multicut.transform_probabilities_to_costs(costs[~stitch_edges],
                                                                         beta=beta0,
                                                                         edge_sizes=sizes[~stitch_edges])
        # if we have a mask, set all edges with 0 to be maximally repulsive
        if mask is not None:
            max_repulsive = 5 * costs.min()
            uv_ids = rag.uvIds()
            costs[(uv_ids == 0).any(axis=1)] = max_repulsive

    # 4.) compute multicut
    solver = multicut.get_multicut_solver(solver_name)
    node_labels = solver(rag, costs)

    # 5.) project multicut results back to segmentation
    segmentation = features.project_node_labels_to_pixels(rag, node_labels, n_threads)
    return segmentation
