import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures
from tqdm import tqdm

from .common import get_blocking
from .unique import unique
from ..util import set_numpy_threads
set_numpy_threads(1)
import numpy as np


def segmentation_filter(data, out, filter_function, block_shape=None,
                        n_threads=None, mask=None, verbose=False, roi=None):

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi)
    n_blocks = blocking.numberOfBlocks

    def apply_filter(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        if mask is None:
            block_mask = None
        else:
            block_mask = mask[bb]
            if block_mask.sum() == 0:
                return

        seg = data[bb]
        seg = filter_function(seg, block_mask)
        out[bb] = seg

    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            list(tqdm(tp.map(apply_filter, range(n_blocks)), total=n_blocks))
        else:
            list(tp.map(apply_filter, range(n_blocks)))

    return out


def size_filter(data, out, min_size=None, max_size=None,
                block_shape=None, n_threads=None, mask=None,
                verbose=False, roi=None):
    """
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        out [array_like] - output data, numpy array or similar like h5py or zarr dataset
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        np.ndarray -
    """
    assert (min_size is not None) or (max_size is not None)
    ids, counts = unique(data, return_counts=True, block_shape=block_shape,
                         n_threads=n_threads, mask=mask, verbose=verbose, roi=roi)

    filter_ids = []
    if min_size is not None:
        filter_ids.extend(ids[counts < min_size])
    if max_size is not None:
        filter_ids.extend(ids[counts > max_size])
    filter_ids = np.array(filter_ids)

    def filter_function(block_seg, block_mask):
        bg_mask = np.isin(block_seg, filter_ids)
        if block_mask is not None:
            bg_mask[~block_mask] = False
        block_seg[bg_mask] = 0
        return block_seg

    out = segmentation_filter(data, out, filter_function,
                              block_shape=block_shape, n_threads=n_threads,
                              mask=mask, verbose=verbose, roi=roi)
    return out
