# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits
import multiprocessing
from typing import Optional, Tuple
from concurrent import futures

import numpy as np
from numpy.typing import ArrayLike
from skimage.feature import peak_local_max
from tqdm import tqdm

from .common import get_blocking


# TODO halo as optional argument?
def find_local_maxima(
    data: ArrayLike,
    min_distance: int = 1,
    threshold_abs: Optional[float] = None,
    threshold_rel: Optional[float] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
) -> np.ndarray:
    """

    Args:
        data:
        min_distance:
        threshold_abs:
        threshold_rel:
        block_shape:
        n_threads:
        verbose:
        roi:

    Returns:
        a
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi, n_threads)

    # We derive the halo from the min distance.
    ndim = data.ndim
    halo = [min_distance + 8] * ndim

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def local_maxima_block(block_id):
        block = blocking.getBlockWithHalo(block_id, list(halo))
        outer_bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
        block_data = data[outer_bb]
        local_maxima = peak_local_max(
            block_data, min_distance=min_distance, threshold_abs=threshold_abs, threshold_rel=threshold_rel
        )

        # Filter out maxima in halo.
        inner_block_start = np.array(block.innerBlockLocal.begin)[None, :]
        inner_block_stop = np.array(block.innerBlockLocal.end)[None, :]
        filter_mask = np.logical_and(
            np.greater_equal(local_maxima, inner_block_start),
            np.less(local_maxima, inner_block_stop),
        ).all(axis=1)
        local_maxima = local_maxima[filter_mask]

        # Apply offest coordinates of the inner block.
        offset = np.array(block.innerBlock.begin)[None, :]
        local_maxima += offset
        return local_maxima

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        results = list(tqdm(
            tp.map(local_maxima_block, range(n_blocks)), total=n_blocks,
            desc="Compute local maxima", disable=not verbose
        ))

    return np.concatenate(results)
