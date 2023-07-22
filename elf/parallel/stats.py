# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits

import multiprocessing
# would be nice to use dask for all of this instead of concurrent.futures
# so that this could be used on a cluster as well
from concurrent import futures
from tqdm import tqdm

from .common import get_blocking

import numpy as np


def mean(data, block_shape=None, n_threads=None, mask=None, verbose=False, roi=None):
    """ Compute the mean of the data in parallel.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        float - mean of the data
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi)
    n_blocks = blocking.numberOfBlocks

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def _mean(block_id):
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

        return np.mean(d)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            means = list(tqdm(tp.map(_mean, range(n_blocks)), total=n_blocks))
        else:
            means = tp.map(_mean, range(n_blocks))
    means = [m for m in means if m is not None]

    return np.mean(means)


def mean_and_std(data, block_shape=None, n_threads=None, mask=None, verbose=False, roi=None):
    """ Compute the mean and the standard deviation of the data in parallel.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        float - mean of the data
        float - standard deviation of the data
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi)
    n_blocks = blocking.numberOfBlocks

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def _mean_and_std(block_id):
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

        return np.mean(d), np.var(d), d.size

    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            results = list(tqdm(tp.map(_mean_and_std, range(n_blocks)), total=n_blocks))
        else:
            results = tp.map(_mean_and_std, range(n_blocks))
    results = [res for res in results if res is not None]

    means = np.array([res[0] for res in results])
    variances = np.array([res[1] for res in results])
    sizes = np.array([res[2] for res in results])

    mean_val = np.mean(means)
    # compute the new variance value and the new standard deviation
    var_val = (sizes * (variances + (means - mean_val) ** 2)).sum() / sizes.sum()
    std_val = np.sqrt(var_val)

    return mean_val, std_val


def std(data, block_shape=None, n_threads=None, mask=None, verbose=False, roi=None):
    """ Compute the standard deviation of the data in parallel.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        float - standard deviation of the data
    """
    return mean_and_std(data, block_shape, n_threads, mask, verbose, roi)[1]


def min_and_max(data, block_shape=None, n_threads=None, mask=None, verbose=False, roi=None):
    """ Compute the minimum and maximum of the data in parallel.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        scalar - minimum value of the data
        scalar - maximum value of the data
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi)
    n_blocks = blocking.numberOfBlocks

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def _min_and_max(block_id):
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

        return np.min(d), np.max(d)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            results = list(tqdm(tp.map(_min_and_max, range(n_blocks)), total=n_blocks))
        else:
            results = tp.map(_min_and_max, range(n_blocks))
    results = [res for res in results if res is not None]

    mins = np.array([res[0] for res in results])
    maxs = np.array([res[1] for res in results])

    return mins.min(), maxs.max()


def min(data, block_shape=None, n_threads=None, mask=None, verbose=False, roi=None):
    """ Compute the minimum of the data in parallel.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        scalar - minimum value of the data
    """
    return min_and_max(data, block_shape, n_threads, mask, verbose, roi)[0]


def max(data, block_shape=None, n_threads=None, mask=None, verbose=False, roi=None):
    """ Compute the maximum of the data in parallel.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        scalar - maximum value of the data
    """
    return min_and_max(data, block_shape, n_threads, mask, verbose, roi)[1]
