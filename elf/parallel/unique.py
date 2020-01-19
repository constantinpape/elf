import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures
from tqdm import tqdm

import nifty.tools as nt
from .common import get_block_shape
from ..util import set_numpy_threads
set_numpy_threads(1)
import numpy as np


def unique(data, return_counts=False, block_shape=None, n_threads=None,
           mask=None, verbose=False):
    """ Compute the unique values of the data.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        return_counts [bool] - whether to return the counts (default: False)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
    Returns:
        np.ndarray - unique values
        np.ndarray - count values (only if return_counts is True)
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    block_shape = get_block_shape(data, block_shape)

    # TODO support roi and use python blocking implementation
    shape = data.shape
    blocking = nt.blocking(data.ndim * [0], shape, block_shape)
    n_blocks = blocking.numberOfBlocks

    def _unique(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype('bool')
            if m.sum() == 0:
                return None

        # load the data and apply the mask if given
        d = data[bb]
        if mask is not None:
            d = d[m]

        return np.unique(d, return_counts=return_counts)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            results = list(tqdm(tp.map(_unique, range(n_blocks)), total=n_blocks))
        else:
            results = tp.map(_unique, range(n_blocks))
    results = [res for res in results if res is not None]

    if return_counts:

        unique_values = [res[0] for res in results]
        count_values = [res[1] for res in results]
        uniques = np.unique(np.concatenate(unique_values))
        counts = np.zeros(int(uniques[-1]) + 1, dtype='uint64')

        for uniques_v, counts_v in zip(unique_values, count_values):
            counts[uniques_v] += counts_v.astype('uint64')
        counts = counts[counts != 0]
        assert len(counts) == len(uniques)
        return uniques, counts

    else:
        return np.unique(np.concatenate(results))
