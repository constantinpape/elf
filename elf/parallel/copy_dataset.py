import multiprocessing
from concurrent import futures
from typing import Optional, Tuple

from numpy.typing import ArrayLike
from tqdm import tqdm
from .common import get_blocking


def copy_dataset(
    ds_in: ArrayLike,
    ds_out: ArrayLike,
    roi_in: Optional[Tuple[slice, ...]] = None,
    roi_out: Optional[Tuple[slice, ...]] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    verbose: bool = False,
) -> ArrayLike:
    """Copy input to an output dataset or other array-like object in parallel.

    Args:
        ds_in: The input dataset or array-like object, like h5py, z5py or zarr dataset.
        ds_out: The output dataset, like h5py, z5py or zarr dataset.
        roi_in: Region of interest for the input dataset.
        roi_out: Region of interest for the output dataset.
        block_shape: Shape of the blocks to use for parallelisation,
            by default chunks of the output dataset will be used.
        n_threads: Number of threads, by default all available ones are used.
        verbose: Verbosity flag.

    Returns:
        The output dataset / array-like object.
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking_out = get_blocking(ds_out, block_shape, roi_out, n_threads)
    out_shape = tuple(re - rb for rb, re in zip(blocking_out.roiBegin, blocking_out.roiEnd))

    block_shape = tuple(blocking_out.blockShape)
    blocking_in = get_blocking(ds_in, block_shape, roi_in)
    in_shape = tuple(re - rb for rb, re in zip(blocking_in.roiBegin, blocking_in.roiEnd))

    if in_shape != out_shape:
        raise ValueError(f"Invalid roi shapes {in_shape}, {out_shape}")

    n_blocks = blocking_out.numberOfBlocks
    if blocking_in.numberOfBlocks != n_blocks:
        raise ValueError(f"Invalid number of blocks {blocking_in.numberOfBlocks}, {n_blocks}")

    def _copy_block(block_id):
        block_in = blocking_in.getBlock(block_id)
        bb_in = tuple(slice(beg, end) for beg, end in zip(block_in.begin, block_in.end))

        block_out = blocking_out.getBlock(block_id)
        bb_out = tuple(slice(beg, end) for beg, end in zip(block_out.begin, block_out.end))

        ds_out[bb_out] = ds_in[bb_in]

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(_copy_block, range(n_blocks)), total=n_blocks, disable=not verbose))

    return ds_out
