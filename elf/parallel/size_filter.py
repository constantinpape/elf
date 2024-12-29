# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits

import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures
from typing import Optional, Tuple

import nifty.tools as nt
from tqdm import tqdm

from .common import get_blocking
from .unique import unique

import numpy as np
from numpy.typing import ArrayLike


def segmentation_filter(
    data: ArrayLike,
    out: ArrayLike,
    filter_function: callable,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    mask: Optional[ArrayLike] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
    relabel: Optional[callable] = None,
) -> ArrayLike:
    """Filter a segmentation based on a custom criterion.

    Args:
        data: Input data, numpy array or similar like h5py or zarr dataset.
        out: Output data, numpy array or similar like h5py or zarr dataset.
        filter_function: The function to express the custom filter criterion.
        block_shape: Shape of the blocks to use for parallelisation,
            by default chunks of the input will be used, if available.
        n_threads: Number of threads, by default all are used.
        mask: Mask to exclude data from the computation.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.
        relabel: Function for relabeling the segmentation.

    Returns:
        The filtered segmentation.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi, n_threads)
    n_blocks = blocking.numberOfBlocks

    def apply_filter(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        if mask is None:
            block_mask = None
        else:
            block_mask = mask[bb].astype("bool")
            if block_mask.sum() == 0:
                return

        seg = data[bb]
        seg = filter_function(seg, block_mask)

        if relabel is not None:
            seg = relabel(seg, block_mask)

        out[bb] = seg

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(apply_filter, range(n_blocks)), total=n_blocks, disable=not verbose))

    return out


def size_filter(
    data: ArrayLike,
    out: ArrayLike,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    mask: Optional[ArrayLike] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
    relabel: bool = True,
) -> ArrayLike:
    """Filter small objects and/or large objects from a segmentation and set them to 0.

    By default this functions relabels the segmentation result consecutively.
    Set relabel=False to avoid this behavior.

    Args:
        data: Input data, numpy array or similar like h5py or zarr dataset.
        out: Output data, numpy array or similar like h5py or zarr dataset.
        min_size: The minimal object size.
        max_size: The maximum object size.
        block_shape: Shape of the blocks to use for parallelisation,
            by default chunks of the input will be used, if available.
        n_threads: Number of threads, by default all are used.
        mask: Mask to exclude data from the computation.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.
        relabel: Whether to relabel the segmentation consecutively after filtering.

    Returns:
        The filtered segmentation.
    """
    assert (min_size is not None) or (max_size is not None)
    ids, counts = unique(data, return_counts=True, block_shape=block_shape,
                         n_threads=n_threads, mask=mask, verbose=verbose, roi=roi)

    filter_ids = []
    if min_size is not None:
        filter_ids.extend(ids[counts < min_size].tolist())
    if max_size is not None:
        filter_ids.extend(ids[counts > max_size].tolist())
    filter_ids = np.array(filter_ids)

    if relabel:
        remaining_ids = np.setdiff1d(ids, filter_ids)
        mapping = {idx: ii for ii, idx in enumerate(remaining_ids)}
        if 0 in mapping:
            assert mapping[0] == 0
        else:
            mapping[0] = 0

        @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
        def _relabel(seg, block_mask):
            if block_mask is None or block_mask.sum() == block_mask.size:
                ids_in_block = np.unique(seg)
                mapping_block = {idx: mapping[idx] for idx in ids_in_block}
                relabeled = nt.takeDict(mapping_block, seg)
            else:
                seg_in_mask = seg[block_mask]
                ids_in_block = np.unique(seg_in_mask)
                mapping_block = {idx: mapping[idx] for idx in ids_in_block}
                relabeled = seg.copy()
                relabeled[block_mask] = nt.takeDict(mapping_block, seg_in_mask)
            return relabeled

    else:
        _relabel = None

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def filter_function(block_seg, block_mask):
        bg_mask = np.isin(block_seg, filter_ids)
        if block_mask is not None:
            bg_mask[~block_mask] = False
        block_seg[bg_mask] = 0
        return block_seg

    out = segmentation_filter(data, out, filter_function,
                              block_shape=block_shape, n_threads=n_threads,
                              mask=mask, verbose=verbose, roi=roi, relabel=_relabel)
    return out
