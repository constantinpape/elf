import multiprocessing
# would be nice to use dask for all of this instead of concurrent.futures
# so that this could be used on a cluster as well
from concurrent import futures

# TODO use python blocking implementation
import nifty.tools as nt

from .common import get_block_shape
from ..util import set_numpy_threads
set_numpy_threads(1)
import numpy as np


def mean(data, block_shape=None, n_threads=None, mask=None):
    """
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    block_shape = get_block_shape(data, block_shape)

    # TODO support roi and use python blocking implementation
    shape = data.shape
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    n_blocks = blocking.numberOfBlocks

    def _mean(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        d = data[bb]
        if mask is not None:
            d = d[mask[bb].astype('bool')]
        return np.mean(d)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_mean, block_id) for block_id in range(n_blocks)]
        means = [t.result() for t in tasks]

    return np.mean(means)


def mean_and_std(data, block_shape=None, n_threads=None, mask=None):
    """
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    block_shape = get_block_shape(data, block_shape)

    # TODO support roi and use python blocking implementation
    shape = data.shape
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    n_blocks = blocking.numberOfBlocks

    def _mean_and_std(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        d = data[bb]
        if mask is not None:
            d = d[mask[bb].astype('bool')]
        return np.mean(d), np.var(d), d.size

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_mean_and_std, block_id) for block_id in range(n_blocks)]
        results = [t.result() for t in tasks]

    means = np.array([res[0] for res in results])
    variances = np.array([res[1] for res in results])
    sizes = np.array([res[2] for res in results])

    mean_val = np.mean(means)
    # compute the new variance value and the new standard deviation
    var_val = (sizes * (variances + (means - mean_val) ** 2)).sum() / sizes.sum()
    std_val = np.sqrt(var_val)

    return mean_val, std_val


def std(data, block_shape=None, n_threads=None, mask=None):
    """
    """
    return mean_and_std(data, block_shape, n_threads, mask)[1]
