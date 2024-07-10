# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits

import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures
from tqdm import tqdm

from .common import get_blocking

import numpy as np


def unique(data, return_counts=False, block_shape=None, n_threads=None,
           mask=None, verbose=False, roi=None):
    """ Compute the unique values of the data.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        return_counts [bool] - whether to return the counts (default: False)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        np.ndarray - unique values
        np.ndarray - count values (only if return_counts is True)
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi, n_threads)
    n_blocks = blocking.numberOfBlocks

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def _unique(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype("bool")
            if m.sum() == 0:
                return None

        # load the data and apply the mask if given
        d = data[bb]
        if mask is not None:
            d = d[m]

        return np.unique(d, return_counts=return_counts)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        results = list(tqdm(tp.map(_unique, range(n_blocks)), total=n_blocks, disable=not verbose))
    results = [res for res in results if res is not None]

    if return_counts:

        unique_values = [res[0] for res in results]
        count_values = [res[1] for res in results]

        # We may have no values at all if everything was masked.
        # In that case return zero as only value and full count.
        try:
            uniques = np.unique(np.concatenate(unique_values))
        except ValueError:
            return np.array([0], dtype=data.dtype), np.array([data.size], dtype="uint64")

        counts = np.zeros(int(uniques[-1]) + 1, dtype="uint64")

        for uniques_v, counts_v in zip(unique_values, count_values):
            counts[uniques_v] += counts_v.astype("uint64")
        counts = counts[counts != 0]
        assert len(counts) == len(uniques)
        return uniques, counts

    else:

        try:
            uniques = np.unique(np.concatenate(results))
        except ValueError:
            return np.array([0], dtype=data.dtype)

        return uniques
