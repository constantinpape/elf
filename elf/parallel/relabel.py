import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures
from tqdm import tqdm

import nifty.tools as nt
from .unique import unique
from .common import get_block_shape
from ..util import set_numpy_threads
set_numpy_threads(1)
import numpy as np


def relabel_consecutive(data, start_label=0, keep_zeros=True, out=None,
                        block_shape=None, n_threads=None,
                        mask=None, verbose=False):
    """Compute the unique values of the data.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        start_label [int] - start value for relabeling (default: 0)
        keep_zeros [bool] - whether to always keep zero mapped to zero (default: True)
        out [array_like] - output, by default the relabeling is done inplace (default: None)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - the output data
        int - the max id after relabeling
        dict - mapping of old to new labels
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    block_shape = get_block_shape(data, block_shape)

    unique_values = unique(data, block_shape=block_shape,
                           mask=mask, n_threads=n_threads)
    mapping = {unv: ii for ii, unv in enumerate(unique_values, start_label)}
    if 0 in mapping and keep_zeros:
        mapping[0] = 0
    max_id = len(mapping) - 1

    # TODO support roi and use python blocking implementation
    shape = data.shape
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    n_blocks = blocking.numberOfBlocks

    if out is None:
        out = data

    def _relabel(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype('bool')
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
        if verbose:
            list(tqdm(tp.map(_relabel, range(n_blocks)), total=n_blocks))
        else:
            tp.map(_relabel, range(n_blocks))

    return out, max_id, mapping
