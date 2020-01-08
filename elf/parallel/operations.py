import multiprocessing
# would be nice to use dask for all of this instead of concurrent.futures
# so that this could be used on a cluster as well
from concurrent import futures
from numbers import Number

# TODO use python blocking implementation
import nifty.tools as nt

from .common import get_block_shape
from ..util import set_numpy_threads
set_numpy_threads(1)
import numpy as np


# TODO support broadcasting
def apply_operation(x, y, operation, out=None,
                    block_shape=None, n_threads=None, mask=None):
    """ Apply operation to two operands in parallel.

    Arguments:
        x [array_like] - operand 1, numpy array or similar like h5py or zarr dataset
        y [array_like or scalar] - operand 2, numpy array or similar like h5py or zarr dataset
            or scalar
        operation [callable] - operation applied to the two operands
        out [array_like] - output, by default the operation
            is done inplace in the first operand (default: None)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
    Returns:
        array_like - output
    """

    if out is None:
        out = x

    # TODO proper check if y is a scalar
    scalar_operand = isinstance(y, Number)

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    block_shape = get_block_shape(x, block_shape)

    # TODO support roi and use python blocking implementation
    shape = x.shape
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    n_blocks = blocking.numberOfBlocks

    def _apply_scalar(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype('bool')
            if m.sum() == 0:
                return None

        # load the data and apply the mask if given
        xx = x[bb]
        if mask is None:
            xx = operation(xx, y)
        else:
            xx[m] = operation(xx[m], y)
        out[bb] = xx

    def _apply_array(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype('bool')
            if m.sum() == 0:
                return None

        # load the data and apply the mask if given
        xx = x[bb]
        yy = y[bb]
        if mask is None:
            xx = operation(xx, yy)
        else:
            xx[m] = operation(xx[m], yy[m])
        out[bb] = xx

    _apply = _apply_scalar if scalar_operand else _apply_array
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_apply, block_id) for block_id in range(n_blocks)]
        [t.result() for t in tasks]

    return out


def add(x, y, out=None, block_shape=None, n_threads=None, mask=None):
    """ Add y to x in parallel.

    Arguments:
        x [array_like] - operand 1, numpy array or similar like h5py or zarr dataset
        y [array_like or scalar] - operand 2, numpy array or similar like h5py or zarr dataset
            or scalar
        out [array_like] - output, by default the operation
            is done inplace in the first operand (default: None)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.add, block_shape=block_shape,
                           n_threads=n_threads, mask=mask)


def subtract(x, y, out=None, block_shape=None, n_threads=None, mask=None):
    """ Subtract x from y in parallel.

    Arguments:
        x [array_like] - operand 1, numpy array or similar like h5py or zarr dataset
        y [array_like or scalar] - operand 2, numpy array or similar like h5py or zarr dataset
            or scalar
        out [array_like] - output, by default the operation
            is done inplace in the first operand (default: None)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.subtract, block_shape=block_shape,
                           n_threads=n_threads, mask=mask)


def multiply(x, y, out=None, block_shape=None, n_threads=None, mask=None):
    """ Multiply x and y in parallel.

    Arguments:
        x [array_like] - operand 1, numpy array or similar like h5py or zarr dataset
        y [array_like or scalar] - operand 2, numpy array or similar like h5py or zarr dataset
            or scalar
        out [array_like] - output, by default the operation
            is done inplace in the first operand (default: None)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.multiply, block_shape=block_shape,
                           n_threads=n_threads, mask=mask)


def divide(x, y, out=None, block_shape=None, n_threads=None, mask=None):
    """ Divide x by y in parallel.

    Arguments:
        x [array_like] - operand 1, numpy array or similar like h5py or zarr dataset
        y [array_like or scalar] - operand 2, numpy array or similar like h5py or zarr dataset
            or scalar
        out [array_like] - output, by default the operation
            is done inplace in the first operand (default: None)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.divide, block_shape=block_shape,
                           n_threads=n_threads, mask=mask)
