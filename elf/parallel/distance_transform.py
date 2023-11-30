# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits

import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures

import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from .common import get_blocking


# TODO other distance transform arguments
def distance_transform(
    data,
    halo,
    out=None,
    block_shape=None,
    n_threads=None,
    verbose=False,
    roi=None,
):
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi, n_threads)

    if out is None:
        out = np.zeros(data.shape, dtype="float32")

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def dist_block(block_id):
        block = blocking.getBlockWithHalo(block_id, list(halo))
        outer_bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
        block_data = data[outer_bb]
        dist = distance_transform_edt(block_data)
        inner_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
        local_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))
        out[inner_bb] = dist[local_bb]

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(dist_block, range(n_blocks)), total=n_blocks,
            desc="Compute distance transform", disable=not verbose
        ))

    return out
