import multiprocessing
# would be nice to use dask for all of this instead of concurrent.futures
# so that this could be used on a cluster as well
from concurrent import futures
from numbers import Number
from tqdm import tqdm

# TODO use python blocking implementation
import nifty.tools as nt

from .common import get_block_shape
from ..util import set_numpy_threads
set_numpy_threads(1)
import numpy as np


# TODO support broadcasting
def apply_operation(x, y, operation, out=None,
                    block_shape=None, n_threads=None,
                    mask=None, verbose=False):
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
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """

    if out is None:
        out = x

    scalar_operand = isinstance(y, Number)
    # check the second operand
    if not scalar_operand:
        if not isinstance(y, np.ndarray):
            raise ValueError("Expected second operand to be scalar or numpy array, got %s" % type(y))
        # check that the shapes match (need to adapt this to support broadcasting)
        if x.shape != y.shape:
            raise ValueError("Shapes of operands do not match.")

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
        if verbose:
            list(tqdm(tp.map(_apply, range(n_blocks)), total=n_blocks))
        else:
            tp.map(_apply, range(n_blocks))

    return out


def add(x, y, out=None, block_shape=None, n_threads=None, mask=None, verbose=False):
    """ Add y to x in parallel.

    Arguments:
        x [array_like] - operand 1, numpy array or similar like h5py or zarr dataset
        y [array_like or scalar] - operand 2, numpy array, h5py or zarr dataset or scalar
        out [array_like] - output, by default the operation
            is done inplace in the first operand (default: None)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.add, block_shape=block_shape,
                           n_threads=n_threads, mask=mask, verbose=verbose,
                           out=out)


def subtract(x, y, out=None, block_shape=None, n_threads=None, mask=None, verbose=False):
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
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.subtract, block_shape=block_shape,
                           n_threads=n_threads, mask=mask, verbose=verbose,
                           out=out)


def multiply(x, y, out=None, block_shape=None, n_threads=None, mask=None, verbose=False):
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
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.multiply, block_shape=block_shape,
                           n_threads=n_threads, mask=mask, verbose=verbose,
                           out=out)


def divide(x, y, out=None, block_shape=None, n_threads=None, mask=None, verbose=False):
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
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.divide, block_shape=block_shape,
                           n_threads=n_threads, mask=mask, verbose=verbose,
                           out=out)


def greater(x, y, out=None, block_shape=None, n_threads=None, mask=None, verbose=False):
    """ Compute greater operation in parallel.

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
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.greater, block_shape=block_shape,
                           n_threads=n_threads, mask=mask, verbose=verbose,
                           out=out)


def greater_equal(x, y, out=None, block_shape=None, n_threads=None, mask=None, verbose=False):
    """ Compute greater operation in parallel.

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
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.greater_equal, block_shape=block_shape,
                           n_threads=n_threads, mask=mask, verbose=verbose,
                           out=out)


def less(x, y, out=None, block_shape=None, n_threads=None, mask=None, verbose=False):
    """ Compute greater operation in parallel.

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
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.less, block_shape=block_shape,
                           n_threads=n_threads, mask=mask, verbose=verbose,
                           out=out)


def less_equal(x, y, out=None, block_shape=None, n_threads=None, mask=None, verbose=False):
    """ Compute less_equal operation in parallel.

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
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.less_equal, block_shape=block_shape,
                           n_threads=n_threads, mask=mask, verbose=verbose,
                           out=out)


def minimum(x, y, out=None, block_shape=None, n_threads=None, mask=None, verbose=False):
    """ Compute minimum operation in parallel.

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
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.minimum, block_shape=block_shape,
                           n_threads=n_threads, mask=mask, verbose=verbose,
                           out=out)


def maximum(x, y, out=None, block_shape=None, n_threads=None, mask=None, verbose=False):
    """ Compute maximum operation in parallel.

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
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """
    return apply_operation(x, y, np.maximum, block_shape=block_shape,
                           n_threads=n_threads, mask=mask, verbose=verbose,
                           out=out)


# TODO with some magic we can autogenerate the parallel ops
# def _generate_operation(op_name, doc_intro):
#     doc_str =\
#     """%s
#
#     Arguments:
#         x [array_like] - operand 1, numpy array or similar like h5py or zarr dataset
#         y [array_like or scalar] - operand 2, numpy array, h5py or zarr dataset or scalar
#         out [array_like] - output, by default the operation
#             is done inplace in the first operand (default: None)
#         block_shape [tuple] - shape of the blocks used for parallelisation,
#             by default chunks of the input will be used, if available (default: None)
#         n_threads [int] - number of threads, by default all are used (default: None)
#         mask [array_like] - mask to exclude data from the computation (default: None)
#         verbose [bool] - verbosity flag (default: False)
#     Returns:
#         array_like - output
#     """ % doc_intro
#
#     def
