import multiprocessing
from concurrent import futures
from functools import partial
from tqdm import tqdm

import numpy as np
from .common import get_blocking


def copy(data, out,
         block_shape=None, n_threads=None,
         mask=None, verbose=False, roi=None):
    """ Copy a dataset in parallel.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        out [array_like] - output dataset
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the output will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        array_like - the copied dataset
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads

    if out.shape != data.shape:
        raise ValueError(f"Output shape {out.shape} does not match input shape {data.shape}")
    if mask is not None and mask.shape != data.shape:
        raise ValueError(f"Invalid mask shape, got {mask.shape}, expected {data.shape} (= shape of first operand)")
    if block_shape is None:
        block_shape = out.chunks

    blocking = get_blocking(data, block_shape, roi)

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
        if verbose:
            list(tqdm(tp.map(_copy_block, range(n_blocks)), total=n_blocks))
        else:
            list(tp.map(_copy_block, range(n_blocks)))

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


def downscale(data, out, downscaling_function=None,
              block_shape=None, n_threads=None,
              mask=None, verbose=False, roi=None):
    """ Downscale a dataset in parallel.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        out [array_like] - output dataset
        downscaling_function [str or callable] - the function used for downscaling the blocks.
            By default mean downscaling is used (default:  fNone)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the output will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
    Returns:
        array_like - the downscaled dataset
    """
    ds_function_dict = {'mean_downscaling': mean_downscaling,
                        'max_downscaling': max_downscaling,
                        'min_downscaling': min_downscaling,
                        'nearest_downscaling': nearest_downscaling,
                        'linear_downscaling': linear_downscaling,
                        'quadratic_downscaling': quadratic_downscaling,
                        'cubic_downscaling': cubic_downscaling}
    ds_function_dict.update({name.replace('_downscaling', ''): func
                             for name, func in ds_function_dict.items()})

    if downscaling_function is None:
        downscaling_function = mean_downscaling
    elif isinstance(downscaling_function, str):
        downscaling_function = ds_function_dict[downscaling_function]
    elif not callable(downscaling_function):
        raise ValueError(f"Invalid downscaling function of type {type(downscaling_function)}")

    if block_shape is None:
        block_shape = out.chunks
    blocking = get_blocking(data, block_shape, roi)

    def _downscale_block(block_id):
        pass

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            list(tqdm(tp.map(_downscale_block, range(n_blocks)), total=n_blocks))
        else:
            list(tp.map(_downscale_block, range(n_blocks)))

    return out
