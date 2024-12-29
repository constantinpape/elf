import multiprocessing
from concurrent import futures
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm
from .common import get_blocking


def copy(
    data: ArrayLike,
    out: ArrayLike,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    mask: Optional[ArrayLike] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
) -> ArrayLike:
    """Copy a dataset or array-like object in parallel.

    Args:
        data: Input data, numpy array or similar like h5py or zarr dataset.
        out: Output dataset or array-like object.
        block_shape: Shape of the blocks to use for parallelisation,
            by default chunks of the output will be used, if available.
        n_threads: Number of threads, by default all are used.
        mask: Mask to exclude data from the computation.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.

    Returns:
        The copied object.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads

    if out.shape != data.shape:
        raise ValueError(f"Output shape {out.shape} does not match input shape {data.shape}")
    if mask is not None and mask.shape != data.shape:
        raise ValueError(f"Invalid mask shape, got {mask.shape}, expected {data.shape} (= shape of first operand)")
    if block_shape is None:
        block_shape = out.chunks

    blocking = get_blocking(data, block_shape, roi, n_threads)

    def _copy_block(block_id):
        block = blocking.getBlock(blockIndex=block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        if mask is not None:
            m = mask[bb]
            if m.sum() == 0:
                return

        block_data = data[bb]
        if mask is not None:
            block_data[m] = 0
        out[bb] = block_data

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(_copy_block, range(n_blocks)), total=n_blocks, disable=not verbose))

    return out


def _ds_block_reduce(data, out_shape, func):
    pass


mean_downscaling = partial(_ds_block_reduce, func=np.mean)
max_downscaling = partial(_ds_block_reduce, func=np.max)
min_downscaling = partial(_ds_block_reduce, func=np.min)


def _ds_interpolate(data, out_shape, order):
    pass


nearest_downscaling = partial(_ds_interpolate, order=0)
linear_downscaling = partial(_ds_interpolate, order=1)
quadratic_downscaling = partial(_ds_interpolate, order=2)
cubic_downscaling = partial(_ds_interpolate, order=3)


def downscale(
    data: ArrayLike,
    out: ArrayLike,
    downscaling_function: Optional[Union[str, callable]] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    mask: Optional[ArrayLike] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
) -> ArrayLike:
    """Downscale a dataset in parallel.

    This functionality is not yet implemented. Calling it will raise a NotImplementedError.

    Args:
        data: Input data, numpy array or similar like h5py or zarr dataset.
        out: Output dataset / array-like object.
        downscaling_function: The function used for downscaling the blocks.
            By default mean downscaling is used.
        block_shape: Shape of the blocks to use for parallelisation,
            by default chunks of the output will be used, if available.
        n_threads: Number of threads, by default all are used.
        mask: Mask to exclude data from the computation.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.

    Returns:
        The downscaled output.
    """
    raise NotImplementedError

    ds_function_dict = {"mean_downscaling": mean_downscaling,
                        "max_downscaling": max_downscaling,
                        "min_downscaling": min_downscaling,
                        "nearest_downscaling": nearest_downscaling,
                        "linear_downscaling": linear_downscaling,
                        "quadratic_downscaling": quadratic_downscaling,
                        "cubic_downscaling": cubic_downscaling}
    ds_function_dict.update({name.replace("_downscaling", ""): func
                             for name, func in ds_function_dict.items()})

    if downscaling_function is None:
        downscaling_function = mean_downscaling
    elif isinstance(downscaling_function, str):
        downscaling_function = ds_function_dict[downscaling_function]
    elif not callable(downscaling_function):
        raise ValueError(f"Invalid downscaling function of type {type(downscaling_function)}")

    if block_shape is None:
        block_shape = out.chunks
    blocking = get_blocking(data, block_shape, roi, n_threads)

    def _downscale_block(block_id):
        pass

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(_downscale_block, range(n_blocks)), total=n_blocks, disable=not verbose))

    return out
