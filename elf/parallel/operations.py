import multiprocessing
# would be nice to use dask for all of this instead of concurrent.futures
# so that this could be used on a cluster as well
from concurrent import futures
from numbers import Number
from tqdm import tqdm

from .common import get_blocking
from ..util import set_numpy_threads
set_numpy_threads(1)
import numpy as np


def _compute_broadcast(shapex, shapey):
    broadcast = []
    for shx, shy in zip(shapex, shapey):
        if shx == shy:
            broadcast.append(False)
        elif shy == 1:
            broadcast.append(True)
        else:
            raise ValueError("Cannot broadcast shapes %s and %s" % (str(shapex, str(shapey))))
    return broadcast


def isin(x, y, out=None,
         block_shape=None, n_threads=None,
         mask=None, verbose=False, roi=None):
    """ Compute np.isin in parallel.

    Arguments:
        x [array_like] - operand 1, numpy array or similar like h5py or zarr dataset
        y [array_like or scalar] - operand 2, scalar, numpy array or list
        out [array_like] - output, by default the operation
            is done inplace in the first operand (default: None)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        array_like - output
    """

    # check the mask if given
    if mask is not None and mask.shape != x.shape:
        raise ValueError("Invalid mask shape, got %s, expected %s (= shape of first operand)" % (str(mask.shape),
                                                                                                 str(x.shape)))

    if out is None:
        out = x
    elif x.shape != out.shape:
        raise ValueError("Expect x and out of same shape, got %s and %s" % (str(x.shape),
                                                                            str(out.shape)))

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(x, block_shape, roi)
    n_blocks = blocking.numberOfBlocks

    def _isin(block_id):
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
            xx = np.isin(xx, y)
        else:
            xx[m] = np.isin(xx[m], y)
        out[bb] = xx

    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            list(tqdm(tp.map(_isin, range(n_blocks)), total=n_blocks))
        else:
            tp.map(_isin, range(n_blocks))

    return out


def apply_operation(x, y, operation, out=None,
                    block_shape=None, n_threads=None,
                    mask=None, verbose=False, roi=None):
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
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        array_like - output
    """

    # check type and dimension of the second operand and check if we need to broadcast
    scalar_operand = isinstance(y, Number)
    if scalar_operand:
        broadcast = False
    else:
        # TODO we ned to check for array_like here to also allow h5py, z5py, etc.
        if not isinstance(y, np.ndarray):
            raise ValueError("Expected second operand to be scalar or numpy array, got %s" % type(y))
        # check that the dimensions of operators
        if x.ndim != y.ndim:
            raise ValueError("Dimensions of operands do not match: %i, %i" % (x.ndim, y.ndim))
        # if the shapes disagree, check if we can broadcast
        broadcast = False if x.shape == y.shape else _compute_broadcast(x.shape, y.shape)

    # broadcasting and masking is not supported yet
    if mask is not None and broadcast:
        raise NotImplementedError("Broadcasting and masking is not implemented yet")

    # check the mask if given
    if mask is not None and mask.shape != x.shape:
        raise ValueError("Invalid mask shape, got %s, expected %s (= shape of first operand)" % (str(mask.shape),
                                                                                                 str(x.shape)))

    if out is None:
        out = x
    elif x.shape != out.shape:
        raise ValueError("Expect x and out of same shape, got %s and %s" % (str(x.shape),
                                                                            str(out.shape)))

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(x, block_shape, roi)
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
        # change the bounding boxes if inputs need to be broadcast
        if broadcast:
            bby = tuple(slice(None) if bcast else b for bcast, b in zip(broadcast, bb))
        else:
            bby = bb

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype('bool')
            if m.sum() == 0:
                return None

        # load the data and apply the mask if given
        xx = x[bb]
        yy = y[bby]
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


# helper function to autogenerate parallel impls of common numpy operations
def _generate_operation(op_name):

    doc_str =\
        """Apply np.%s block-wise and in parallel.

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
            roi [tuple[slice]] - region of interest for this computation (default: None)
        Returns:
            array_like - output
        """ % op_name

    def op(x, y, out=None, block_shape=None, n_threads=None,
           mask=None, verbose=False, roi=None):
        return apply_operation(x, y, getattr(np, op_name), block_shape=block_shape,
                               n_threads=n_threads, mask=mask, verbose=verbose,
                               out=out, roi=roi)

    op.__doc__ = doc_str
    op.__name__ = op_name
    globals()[op_name] = op


# autogenerate parallel implementation for common numpy operations
_op_names = ['add', 'subtract', 'multiply', 'divide',
             'greater', 'greater_equal', 'less', 'less_equal',
             'minimum', 'maximum']


for op_name in _op_names:
    _generate_operation(op_name)

del _generate_operation
del _op_names
