# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits
from typing import Optional, Tuple, Union

import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from .common import get_blocking


def map_points_to_objects(
    segmentation: ArrayLike,
    points: np.ndarray,
    block_shape: Tuple[int, ...],
    halo: Optional[Tuple[int, ...]] = None,
    sampling: Optional[Union[float, Tuple[float, ...]]] = None,
    n_threads: Optional[int] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
) -> Tuple[np.array, np.array]:
    """Map point coordinates to objects in a segmentation and measure the corresponding distance.

    Args:
        segmentation: The segmentation.
        points: The point coordinates.
        block_shape: The block shape for parallelization.
        halo: The halo for extending the blocks in parallelization.
            It should be chosen such that the maximum distance of interest can be computed
            within the halo.
        sampling: The sampling for computing the distances.
        n_threads: The number of threads for parallelization.
        verbose: Whether to print the progress of the computation.
        roi: The region of interest.

    Returns:
        object_ids:
        objetc_distances:
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(segmentation, block_shape, roi, n_threads)

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def map_block(block_id):
        if halo is None:
            block = blocking.getBlockWithHalo(block_id, list(halo))
        else:
            block = blocking.getBlockWithHalo(block_id, list(halo)).outerBlock

        # Get the bounding box and map the points to it.
        bb_min, bb_max = block.begin, block.end
        point_mask = np.logical_and.reduce(
            [points[:, i] >= bb_min[i] for i in range(points.shape[1])] +
            [points[:, i] < bb_max[i] for i in range(points.shape[1])]
        )
        # We don't need to do anything if there are no points.
        if point_mask.sum() == 0:
            return None

        # Get the ids of the points and convert their coordinates to local coordinates
        point_ids = np.where(point_mask)[0]
        block_points = points[point_mask]
        block_points -= np.array(bb_min)[None, :]

        # Get the segmentation for the block.
        bb = tuple(slice(beg, end) for beg, end in zip(bb_min, bb_max))
        block_seg = segmentation[bb]

        # Return empty assignments if the segmentation is empty.
        if block_seg.sum() == 0:
            object_ids = np.full(len(point_ids), 0, segmentation.dtype)
            object_distances = np.full(len(point_ids), np.inf, "float32")
            return point_ids, object_ids, object_distances

        distances, indices = distance_transform_edt(block_seg == 0, sampling=sampling, return_indices=True)

        # Index with the points and return their mapped objects and distances.
        point_coords = tuple(block_points[:, i] for i in range(block_points.shape[1]))
        object_distances = distances[point_coords]
        assert block_points.shape[1] == indices.shape[0]
        nearest_object_coord = tuple(indices[i][point_coords] for i in range(block_points.shape[1]))
        object_ids = block_seg[nearest_object_coord]

        assert len(point_ids) == len(object_distances)
        assert len(point_ids) == len(object_ids)
        return point_ids, object_ids, object_distances

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        results = list(tqdm(
            tp.map(map_block, range(n_blocks)), total=n_blocks,
            desc="Map points to objects", disable=not verbose
        ))

    n_points = len(points)
    object_ids = np.zeros(n_points, dtype=segmentation.dtype)
    object_distances = np.full(n_points, np.inf, dtype="float32")

    # Merge the results to the output arrays.
    for res in results:
        if res is None:
            continue
        this_point_ids, this_object_ids, this_distances = res

        # We may have previous assignments, due to overlap introduced by the halo
        previous_assignments = (object_ids[this_point_ids] != 0)
        if halo is None and previous_assignments.sum() > 0:
            raise RuntimeError("No previous assignments expected")
        elif previous_assignments.sum() > 0:  # We do have previous assignments.
            previous_ids, previous_distances = object_ids[this_point_ids], object_distances[this_point_ids]
            assert len(previous_ids) == len(this_object_ids)
            assert len(previous_distances) == len(this_distances)

            # Choose the values with the minimal distance.
            take_previous_assignment = this_distances > previous_distances
            this_object_ids[take_previous_assignment] = previous_ids[take_previous_assignment]
            this_distances[take_previous_assignment] = previous_distances[take_previous_assignment]

        object_ids[this_point_ids] = this_object_ids
        object_distances[this_point_ids] = this_distances

    return object_ids, object_distances


def distance_transform(
    data: ArrayLike,
    halo: Tuple[int, ...],
    sampling: Optional[Union[float, Tuple[float, ...]]] = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Optional[ArrayLike] = None,
    indices: Optional[ArrayLike] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
    """Compute distance transform in parallel over blocks.

    The results are only correct up to the distance to the block boundary plus halo.
    The function `scipy.ndimage.distance_transform_edt` is used to compute the distances.

    Args:
        data: The input data.
        halo: The halo, which is the padding added at each side of the block.
        sampling: The sampling value passed to distance_transfor_edt.
        return_distances: Whether to return the computed distances.
        return_indices: Whether to return the computed indices.
        distances: Pre-allocated array-like object for the distances.
        indices: Pre-allocated array-like object for the indices.
        block_shape: Shape of the blocks to use for parallelisation,
            by default chunks of the input will be used, if available.
        n_threads: Number of threads, by default all available threads are used.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.

    Returns:
        The distances, if return_distances is set to True.
        The indices, if return_distances is set ot True.
    """
    if data.ndim not in (2, 3):
        raise ValueError(
            f"Invalid input dimensionality. Expected input to have 2 or 3 dimensions, got {data.ndim}."
        )
    if (not return_distances) and (not return_indices):
        raise ValueError("Either return_distances or return_indices must be True.")

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi, n_threads)

    if return_distances:
        if distances is None:
            distances = np.zeros(data.shape, dtype="float32")
        else:
            assert distances.shape == data.shape

    if return_indices:
        if indices is None:
            indices = np.zeros((data.ndim,) + data.shape, dtype="int64")
        else:
            assert indices.shape == (data.ndim,) + data.shape

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def dist_block(block_id):
        block = blocking.getBlockWithHalo(block_id, list(halo))
        outer_bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
        block_data = data[outer_bb]

        ret_edt = distance_transform_edt(
            block_data, sampling=sampling, return_distances=return_distances, return_indices=return_indices
        )
        if return_distances and return_indices:
            dist, ind = ret_edt
        elif return_distances:
            dist = ret_edt
        else:
            ind = ret_edt

        inner_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
        local_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))

        if return_distances:
            distances[inner_bb] = dist[local_bb]

        if return_indices:
            offset = np.array([begin for begin in block.outerBlock.begin])
            slice_nd = np.s_[:, None, None] if data.ndim == 2 else np.s_[:, None, None, None]
            offset = offset[slice_nd]

            inner_bb, local_bb = (slice(None),) + inner_bb, (slice(None),) + local_bb
            ind = ind[local_bb] + offset
            indices[inner_bb] = ind

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(dist_block, range(n_blocks)), total=n_blocks,
            desc="Compute distance transform", disable=not verbose
        ))

    if return_indices and return_distances:
        return distances, indices
    elif return_distances:
        return distances
    else:
        return indices
