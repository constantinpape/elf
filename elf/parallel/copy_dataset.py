import multiprocessing
from concurrent import futures

from tqdm import tqdm
from .common import get_blocking


# FIXME this seems to be not thread-safe yet
def copy_dataset(ds_in, ds_out,
                 roi_in=None,
                 roi_out=None,
                 block_shape=None,
                 n_threads=None,
                 verbose=False):
    """ Copy input to output dataset in parallel.

    Arguments:
        ds_in [dataset] - input dataset (h5py, z5py or zarr dataset)
        ds_out [dataset] - output dataset (h5py, z5py or zarr dataset)
        roi_in [tuple[slice]] - region of interest (for the input dataset) (default: None)
        roi_out [tuple[slice]] - region of interest (for the output dataset) (default: None)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the output dataset will be used (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - output
    """

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking_out = get_blocking(ds_out, block_shape, roi_out)
    out_shape = tuple(re - rb for rb, re in zip(blocking_out.roiBegin, blocking_out.roiEnd))

    # print(blocking_out.blockShape)
    # return ds_out

    block_shape = tuple(blocking_out.blockShape)
    blocking_in = get_blocking(ds_in, block_shape, roi_in)
    in_shape = tuple(re - rb for rb, re in zip(blocking_in.roiBegin, blocking_in.roiEnd))

    if in_shape != out_shape:
        print(roi_out)
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
        if verbose:
            list(tqdm(tp.map(_copy_block, range(n_blocks)), total=n_blocks))
        else:
            tp.map(_copy_block, range(n_blocks))

    return ds_out
