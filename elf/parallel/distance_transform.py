# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits

import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures

import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from .common import get_blocking


def distance_transform(
    data,
    halo,
    sampling=None,
    return_distances=True,
    return_indices=False,
    distances=None,
    indices=None,
    block_shape=None,
    n_threads=None,
    verbose=False,
    roi=None,
):
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
