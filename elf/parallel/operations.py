# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits

import multiprocessing
# would be nice to use dask for all of this instead of concurrent.futures
# so that this could be used on a cluster as well
from concurrent import futures
from functools import partial
from numbers import Number
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from .common import get_blocking


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


def isin(
    x: ArrayLike,
    y: Union[ArrayLike, Number],
    out: Optional[ArrayLike] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    mask: Optional[ArrayLike] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
) -> ArrayLike:
    """Compute np.isin in parallel.

    Args:
        x: Operand 1, numpy array or similar, like h5py or zarr dataset.
        y: Operand 2, scalar, numpy array or list.
        out: Output, by default the operation is done inplace in the first operand.
        block_shape: Shape of the blocks to use for parallelisation,
            by default chunks of the input will be used, if available.
        n_threads: Number of threads, by default all are used.
        mask: Mask to exclude data from the computation.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.

    Returns:
        The output.
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
    blocking = get_blocking(x, block_shape, roi, n_threads)
    n_blocks = blocking.numberOfBlocks

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def _isin(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype("bool")
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
        list(tqdm(tp.map(_isin, range(n_blocks)), total=n_blocks, disable=not verbose))

    return out


def apply_operation(
    x: ArrayLike,
    y: Union[ArrayLike, Number],
    operation: callable,
    out: Optional[ArrayLike] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    mask: Optional[ArrayLike] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
) -> ArrayLike:
    """Apply operation to two operands in parallel.

    Args:
        x: Operand 1, numpy array or similar like h5py or zarr dataset.
        y: Operand 2, numpy array or similar like h5py or zarr dataset or scalar.
        operation: Operation applied to the two operands.
        out: Output, by default the operation is done inplace in the first operand.
        block_shape: Shape of the blocks to use for parallelisation,
            by default chunks of the input will be used, if available.
        n_threads: Number of threads, by default all are used.
        mask: Mask to exclude data from the computation.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.

    Returns:
        The output.
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
    blocking = get_blocking(x, block_shape, roi, n_threads)
    n_blocks = blocking.numberOfBlocks

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def _apply_scalar(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype("bool")
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
            m = mask[bb].astype("bool")
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
        list(tqdm(tp.map(_apply, range(n_blocks)), total=n_blocks, disable=not verbose))

    return out


def apply_operation_single(
    x: ArrayLike,
    operation: callable,
    axis: Optional[int] = None,
    out: Optional[ArrayLike] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    mask: Optional[ArrayLike] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
) -> ArrayLike:
    """Apply operation to single operand in parallel.

    Args:
        x: Operand 1, numpy array or similar like h5py or zarr dataset.
        operation: Operation applied to the two operands.
        axis: Axis along which to apply the operation.
        out: Output, by default the operation is done inplace in the first operand.
        block_shape: Shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available.
        n_threads: Number of threads, by default all are used.
        mask: Mask to exclude data from the computation.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.

    Returns:
        The output.
    """

    shape = x.shape
    if axis is not None:
        operation = partial(operation, axis=axis)
        shape = tuple(sh for ii, sh in enumerate(shape) if ii != axis)

    # check the mask if given
    if mask is not None and mask.shape != shape:
        raise ValueError("Invalid mask shape, got %s, expected %s (= shape of first operand)" % (str(mask.shape),
                                                                                                 str(shape)))
    # if no output is given, apply this operation inplace
    if out is None:
        out = x

    # check the shape against the output shape
    if shape != out.shape:
        raise ValueError("Expect x and out of same shape, got %s and %s" % (str(shape), str(out.shape)))

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(out, block_shape, roi, n_threads)
    n_blocks = blocking.numberOfBlocks

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def _apply(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype("bool")
            if m.sum() == 0:
                return None

        if axis is None:
            bb_in = bb
        else:
            bb_in = bb[:axis] + (slice(None),) + bb[axis:]

        # load the data and apply the mask if given
        xx = operation(x[bb_in])
        if mask is not None:
            xx[m] = 0
        out[bb] = xx

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(_apply, range(n_blocks)), total=n_blocks, disable=not verbose))

    return out


# helper function to autogenerate parallel impls of common numpy operations
def _generate_operation(op_name):

    doc_str =\
        """Apply np.%s block-wise and in parallel.

        Args:
            x: Operand 1, numpy array or similar like h5py or zarr dataset.
            y: Operand 2, numpy array, h5py or zarr dataset or scalar.
            out: Output, by default the operation is done inplace in the first operand.
            block_shape: Shape of the blocks to use for parallelisation,
                by default chunks of the input will be used, if available.
            n_threads: Number of threads, by default all are used.
            mask: Mask to exclude data from the computation.
            verbose: Verbosity flag.
            roi: Region of interest for this computation.

        Returns:
            The output.
        """ % op_name

    def op(
        x: ArrayLike,
        y: Union[ArrayLike, Number],
        out: Optional[ArrayLike] = None,
        block_shape: Optional[Tuple[int, ...]] = None,
        n_threads: Optional[int] = None,
        mask: Optional[ArrayLike] = None,
        verbose: bool = False,
        roi: Optional[Tuple[slice, ...]] = None,
    ) -> ArrayLike:
        return apply_operation(x, y, getattr(np, op_name), block_shape=block_shape,
                               n_threads=n_threads, mask=mask, verbose=verbose,
                               out=out, roi=roi)

    op.__doc__ = doc_str
    op.__name__ = op_name
    globals()[op_name] = op


# autogenerate parallel implementation for common numpy operations
_op_names = ["add", "subtract", "multiply", "divide",
             "greater", "greater_equal", "less", "less_equal",
             "minimum", "maximum"]


for op_name in _op_names:
    _generate_operation(op_name)

del _generate_operation
del _op_names
