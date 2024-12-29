import multiprocessing
from concurrent import futures
from typing import Callable, Tuple, Optional, Union

import vigra
import numpy as np

import nifty.tools as nt
from nifty.ground_truth import overlap as compute_overlap

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from .features import compute_rag, project_node_labels_to_pixels
from .multicut import compute_edge_costs, multicut_decomposition


def stitch_segmentation(
    input_: np.ndarray,
    segmentation_function: Callable,
    tile_shape: Tuple[int, int],
    tile_overlap: Tuple[int, int],
    beta: float = 0.5,
    shape: Optional[Tuple[int, int]] = None,
    with_background: bool = True,
    n_threads: Optional[int] = None,
    return_before_stitching: bool = False,
    verbose: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Run segmentation function tilewise and stitch the results based on overlap.

    Args:
        input_: The input data. If the data has channels they need to be passed as last dimension,
            e.g. XYC for a 2D image with channels.
        segmentation_function: the function to perform segmentation for each tile.
            It must take the input (for the tile) as well as the id of the tile as input;
            i.e. the function needs to have a signature like this: 'def my_seg_func(tile_input_, tile_id)'.
            The tile_id is passed in case the segmentation differs based on the tile and can be ignored otherwise.
        tile_shape: Shape of the individual tiles.
        tile_overlap: Overlap of the tiles.
            The input to the segmentation function will have the size tile_shape + 2 * tile_overlap.
            The tile overlap will be used to compute the overlap between objects, which will be used for stitching.
        beta: Parameter to bias the stitching results towards more over-segmentation (beta > 0.5)
            or more under-segmentation (beta < 0.5). Has to be in the exclusive range 0 to 1.
        shape: Shape of the segmentation. By default this will use the shape of the input, but if the
            input has channels it needs to be passed.
        with_background: Whether this is a segmentation problem with background. In this case the
            background id (which is hard-coded to 0), will not be stitched.
        n_threads: Number of threads that will be used for parallelized operations.
            Set to the number of cores by default.
        return_before_stitching: Return the result before stitching for debugging.
        verbose: Whether to print progress bars.

    Returns:
        The stitched segmentation.
        The segmentation before stitching, if return_before_stitching is set to True.
    """

    shape = input_.shape if shape is None else shape
    ndim = len(shape)
    blocking = nt.blocking([0] * ndim, shape, tile_shape)

    id_offset = 0
    block_segs = []
    seg = np.zeros(shape, dtype="uint64")

    n_blocks = blocking.numberOfBlocks

    # Run tiled segmentation.
    for block_id in tqdm(range(n_blocks), total=n_blocks, desc="Run tiled segmentation", disable=not verbose):
        block = blocking.getBlockWithHalo(block_id, list(tile_overlap))
        outer_bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))

        block_input = input_[outer_bb]
        block_seg = segmentation_function(block_input, block_id)

        if with_background:
            block_mask = block_seg != 0
            # We need to make sure that empty blocks do not reset the offset.
            if block_mask.sum() > 0:
                block_seg[block_mask] += id_offset
                id_offset = block_seg.max()
        else:
            block_seg += id_offset
            id_offset = block_seg.max()

        inner_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
        local_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))

        seg[inner_bb] = block_seg[local_bb]
        block_segs.append(block_seg)

    # Compute the region adjacency graph for the tiled segmentation.
    # In order to computhe the the edges between block boundaries (stitch edges).
    seg_ids = np.unique(seg)
    rag = compute_rag(seg, n_threads=n_threads)

    # We initialize the edge disaffinities with a high value (corresponding to a low overlap),
    # so that merging pairs that are not on the edge is very unlikely
    # but not completely impossible in case it is needed for a consistent solution.
    edge_disaffinities = np.full(rag.numberOfEdges, 0.9, dtype="float32")

    def _compute_overlaps(block_id):

        # For each axis, load the face with the lower block neighbor and compute the object overlaps.
        for axis in range(ndim):
            ngb_id = blocking.getNeighborId(block_id, axis, lower=True)
            if ngb_id == -1:
                continue

            this_block = blocking.getBlockWithHalo(block_id, list(tile_overlap))
            ngb_block = blocking.getBlockWithHalo(ngb_id, list(tile_overlap))

            # Load the full block segmentations.
            this_seg, ngb_seg = block_segs[block_id], block_segs[ngb_id]

            # Get the global coordinates of the block face.
            face = tuple(
                slice(beg_out, end_out) if d != axis else slice(beg_out, beg_in + tile_overlap[d])
                for d, (beg_out, end_out, beg_in) in enumerate(
                    zip(this_block.outerBlock.begin, this_block.outerBlock.end, this_block.innerBlock.begin)
                )
            )

            # Map to the two local face coordinates.
            this_face_bb = tuple(
                slice(fa.start - offset, fa.stop - offset) for fa, offset in zip(face, this_block.outerBlock.begin)
            )
            ngb_face_bb = tuple(
                slice(fa.start - offset, fa.stop - offset) for fa, offset in zip(face, ngb_block.outerBlock.begin)
            )

            # Load the two segmentations for the face.
            this_face = this_seg[this_face_bb]
            ngb_face = ngb_seg[ngb_face_bb]
            assert this_face.shape == ngb_face.shape, (this_face.shape, ngb_face.shape)

            # Compute the object overlaps.
            overlap_comp = compute_overlap(this_face, ngb_face)
            this_ids = np.unique(this_face)
            overlaps = {this_id: overlap_comp.overlapArraysNormalized(this_id, sorted=False) for this_id in this_ids}
            overlap_ids = {this_id: ovlps[0] for this_id, ovlps in overlaps.items()}
            overlap_values = {this_id: ovlps[1] for this_id, ovlps in overlaps.items()}
            overlap_uv_ids = np.array([
                [this_id, ovlp_id] for this_id, ovlp_ids in overlap_ids.items() for ovlp_id in ovlp_ids
            ])
            overlap_values = np.array([ovlp for ovlps in overlap_values.values() for ovlp in ovlps], dtype="float32")
            assert len(overlap_uv_ids) == len(overlap_values)

            # Get the edge ids, then exclude invalid edges and set the edge disaffinities to 1 - overlap.

            # We might have ids in the overlaps that are not in the final segmentation, these need to be filtered.
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

            edge_disaffinities[edge_ids] = (1.0 - overlap_values)

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(
            _compute_overlaps, range(n_blocks)), total=n_blocks, desc="Compute object overlaps", disable=not verbose
        ))

    # If we have background, then set all the edges that are connecting 0 to another element to be very unlikely.
    if with_background:
        uv_ids = rag.uvIds()
        bg_edges = rag.findEdges(uv_ids[(uv_ids == 0).any(axis=1)])
        edge_disaffinities[bg_edges] = 0.99
    costs = compute_edge_costs(edge_disaffinities, beta=beta)

    # Run multicut to get the segmentation result.
    node_labels = multicut_decomposition(rag, costs)
    seg_stitched = project_node_labels_to_pixels(rag, node_labels, n_threads=n_threads)

    if with_background:
        vigra.analysis.relabelConsecutive(seg_stitched, out=seg_stitched, start_label=1, keep_zeros=True)
    else:
        vigra.analysis.relabelConsecutive(seg_stitched, out=seg_stitched, start_label=1, keep_zeros=False)

    if return_before_stitching:
        return seg_stitched, seg

    return seg_stitched


def stitch_tiled_segmentation(
    segmentation: np.ndarray,
    tile_shape: Tuple[int, int],
    overlap: int = 1,
    with_background: bool = True,
    n_threads: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Stitch a segmentation that is split into tiles.

    The ids in the tiles of the input segmentation have to be unique,
    i.e. the segmentations have to be separate across tiles.

    Args:
        segmentation: The input segmentation.
        tile_shape: The shape of tiles.
        overlap: The overlap between adjacent tiles that is used to compute overlap for stitching objects.
        with_background: Whether this is a segmentation problem with background. In this case the
            background id (which is hard-coded to 0), will not be stitched.
        n_threads: The number of threads used for parallelized operations. Set to the number of cores by default.
        verbose: Whether to print the progress bars.

    Returns:
        The stitched segmentation with merged labels.
    """
    shape = segmentation.shape
    ndim = len(shape)
    blocking = nt.blocking([0] * ndim, shape, tile_shape)
    n_blocks = blocking.numberOfBlocks

    block_segs = []

    # Get the tiles from the segmentation of shape: 'tile_shape'.
    for block_id in tqdm(range(n_blocks), desc="Get tiles from the segmentation", disable=not verbose):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        block_seg = segmentation[bb]
        block_segs.append(block_seg)

    # Conpute the Region Adjacency Graph (RAG) for the tiled segmentation.
    # and the edges between block boundaries (stitch edges).
    seg_ids = np.unique(segmentation)
    rag = compute_rag(segmentation)

    # We initialize the edge disaffinities with a high value (corresponding to a low overlap)
    # so that merging things that are not on the edge is very unlikely
    # but not completely impossible in case it is needed for a consistent solution.
    edge_disaffinities = np.full(rag.numberOfEdges, 0.9, dtype="float32")

    def _compute_overlaps(block_id):
        # For each axis, load the face with the lower block neighbor and compute the object overlaps
        for axis in range(ndim):
            ngb_id = blocking.getNeighborId(block_id, axis, lower=True)
            if ngb_id == -1:
                continue

            # Load the respective tiles.
            this_seg, ngb_seg = block_segs[block_id], block_segs[ngb_id]

            # Get the local face coordinates of the respective tiles.
            # We get the face region of the shape defined by 'overlap'
            # eg. The default '1' returns a 1d cross-section of the tile interfaces.
            face_bb = tuple(slice(None) if d != axis else slice(0, overlap) for d in range(ndim))
            ngb_face_bb = tuple(
                slice(None) if d != axis else slice(ngb_seg.shape[d] - overlap, ngb_seg.shape[d]) for d in range(ndim)
            )

            # Load the two segmentations for the face.
            this_face = this_seg[face_bb]
            ngb_face = ngb_seg[ngb_face_bb]

            # Both the faces from each tile are expected to be of the same shape
            assert this_face.shape == ngb_face.shape, (this_face.shape, ngb_face.shape)

            # Compute the object overlaps.
            # In this step, we compute the per-instance overlap over both faces
            overlap_comp = compute_overlap(this_face, ngb_face)
            this_ids = np.unique(this_face).astype("uint32")
            overlaps = {this_id: overlap_comp.overlapArraysNormalized(this_id, sorted=False) for this_id in this_ids}
            overlap_ids = {this_id: ovlps[0] for this_id, ovlps in overlaps.items()}
            overlap_values = {this_id: ovlps[1] for this_id, ovlps in overlaps.items()}
            overlap_uv_ids = np.array([
                [this_id, ovlp_id] for this_id, ovlp_ids in overlap_ids.items() for ovlp_id in ovlp_ids
            ])
            overlap_values = np.array([ovlp for ovlps in overlap_values.values() for ovlp in ovlps], dtype="float32")
            assert len(overlap_uv_ids) == len(overlap_values)

            # Next, we remove the invalid edges.
            # We might have ids in the overlaps that are not in the segmentation. We filter them out.
            valid_uv_ids = np.isin(overlap_uv_ids, seg_ids).all(axis=1)
            if valid_uv_ids.sum() == 0:
                continue
            overlap_uv_ids, overlap_values = overlap_uv_ids[valid_uv_ids], overlap_values[valid_uv_ids]
            assert len(overlap_uv_ids) == len(overlap_values)

            # Get the edge ids.
            edge_ids = rag.findEdges(overlap_uv_ids)
            valid_edges = edge_ids != -1
            if valid_edges.sum() == 0:
                continue
            edge_ids, overlap_values = edge_ids[valid_edges], overlap_values[valid_edges]
            assert len(edge_ids) == len(overlap_values)

            # And set the global edge disaffinities to (1 - overlap).
            edge_disaffinities[edge_ids] = (1.0 - overlap_values)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(
            _compute_overlaps, range(n_blocks)), total=n_blocks, desc="Compute object overlaps", disable=not verbose,
        ))

    uv_ids = rag.uvIds()
    if with_background:
        bg_edges = rag.findEdges(uv_ids[(uv_ids == 0).any(axis=1)])
        edge_disaffinities[bg_edges] = 0.99
    costs = compute_edge_costs(edge_disaffinities, beta=0.5)

    # Run multicut to get the segmentation result.
    node_labels = multicut_decomposition(rag, costs)
    seg_stitched = project_node_labels_to_pixels(rag, node_labels)

    return seg_stitched
