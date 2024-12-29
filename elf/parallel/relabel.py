# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits

import multiprocessing

# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures
from typing import Dict, Optional, Tuple

import nifty.tools as nt
from tqdm import tqdm

from .unique import unique
from .common import get_blocking

import numpy as np
from numpy.typing import ArrayLike


def relabel_consecutive(
    data: ArrayLike,
    start_label: int = 0,
    keep_zeros: bool = True,
    out: Optional[ArrayLike] = None,
    block_shape: Tuple[int, ...] = None,
    n_threads: Optional[int] = None,
    mask: Optional[ArrayLike] = None,
    verbose: bool = False,
    roi: Tuple[slice, ...] = None,
) -> Tuple[ArrayLike, int, Dict[int, int]]:
    """Compute the unique values of the data.

    Args:
        data: Input data, numpy array or similar like h5py or zarr dataset.
        start_label: Start value for relabeling.
        keep_zeros: Whether to always keep zero mapped to zero.
        out: Output, by default the relabeling is done inplace.
        block_shape: Shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available.
        n_threads: Number of threads, by default all are used.
        mask: Mask to exclude data from the computation.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.

    Returns:
        The relabeled output data.
        The max id after relabeling.
        The mapping of old to new labels.
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi, n_threads)
    block_shape = blocking.blockShape

    unique_values = unique(data, block_shape=block_shape,
                           mask=mask, n_threads=n_threads)
    mapping = {unv: ii for ii, unv in enumerate(unique_values, start_label)}
    if 0 in mapping and keep_zeros:
        mapping[0] = 0
    max_id = len(mapping) - 1

    n_blocks = blocking.numberOfBlocks
    if out is None:
        out = data
    elif data.shape != out.shape:
        raise ValueError("Expect data and out of same shape, got %s and %s" % (str(data.shape),
                                                                               str(out.shape)))

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def _relabel(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype("bool")
            if m.sum() == 0:
                return None

        d = data[bb]
        if mask is None or m.sum() == m.size:
            un_block = np.unique(d)
            mapping_block = {un: mapping[un] for un in un_block}
            o = nt.takeDict(mapping_block, d)
        else:
            v = d[m]
            un_block = np.unique(v)
            mapping_block = {un: mapping[un] for un in un_block}
            o = d.copy()
            o[m] = nt.takedDict(mapping_block, v)
        out[bb] = o

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(_relabel, range(n_blocks)), total=n_blocks, disable=not verbose))

    return out, max_id, mapping
