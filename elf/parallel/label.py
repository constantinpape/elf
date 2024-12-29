# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits
from typing import Optional, Tuple

import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures

from tqdm import tqdm
from skimage.measure import label as label_impl
from skimage.segmentation import relabel_sequential

import nifty.tools as nt
import nifty.ufd as nufd
from .common import get_blocking

import numpy as np
from numpy.typing import ArrayLike


def cc_blocks(data, out, mask, blocking, with_background, n_threads, verbose):
    """@private
    """
    n_blocks = blocking.numberOfBlocks

    # Compute the connected component for one block.
    @threadpool_limits.wrap(limits=1)  # Restrict the numpy threadpool to 1 to avoid oversubscription.
    def _cc_block(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # Check if we have a mask and if we do, if we have pixels in the mask.
        if mask is not None:
            m = mask[bb].astype("bool")
            if m.sum() == 0:
                return 0

        # Load the data from this block.
        d = data[bb].copy()

        # Determine the background value.
        bg_val = 0 if with_background else int(d.max() + 1)

        # Set mask to background value.
        if mask is not None:
            d[~m] = bg_val

        d = label_impl(d, background=bg_val, connectivity=1)

        out[bb] = d
        return int(d.max())

    # Compute connected components for all blocks in parallel.
    with futures.ThreadPoolExecutor(n_threads) as tp:
        block_max_labels = list(tqdm(
           tp.map(_cc_block, range(n_blocks)), total=n_blocks, desc="Label all sub-blocks", disable=not verbose
        ))

    return out, block_max_labels


def merge_blocks(data, out, mask, offsets, blocking, max_id, with_background, n_threads, verbose):
    """@private
    """
    n_blocks = blocking.numberOfBlocks
    ndim = out.ndim

    @threadpool_limits.wrap(limits=1)  # Restrict the numpy threadpool to 1 to avoid oversubscription.
    def _merge_block_faces(block_id):
        block = blocking.getBlock(block_id)
        offset_block = offsets[block_id]

        merge_labels = []
        # For each axis, load the face with the lower block neighbor and compute the merge labels.
        for axis in range(ndim):
            ngb_id = blocking.getNeighborId(block_id, axis, lower=True)
            if ngb_id == -1:
                continue
            ngb_block = blocking.getBlock(ngb_id)

            # Make the bounding box for both faces and load the segmentation for it.
            face = tuple(slice(beg, end) if d != axis else slice(beg, beg + 1)
                         for d, (beg, end) in enumerate(zip(block.begin, block.end)))
            ngb_face = tuple(slice(beg, end) if d != axis else slice(end - 1, end)
                             for d, (beg, end) in enumerate(zip(ngb_block.begin,
                                                                ngb_block.end)))

            # Load and combine the mask for bot faces.
            if mask is not None:
                m = np.logical_and(out[face], out[ngb_face])
                if m.sum() == 0:
                    continue

            # Load the initial labels for both faces.
            d, d_ngb = data[face], data[ngb_face]
            assert d.shape == d_ngb.shape

            # Load the intermediate result for both faces.
            o, o_ngb = out[face], out[ngb_face]
            assert o.shape == o_ngb.shape == d.shape

            # Allocate full mask if we don't have a mask dataset.
            if mask is None:
                m = np.ones_like(d, dtype="bool")

            # Mask zero label if we have background.
            if with_background:
                m[d == 0] = 0
                m[d_ngb == 0] = 0

            # Mask pixels of the face where d != d_ngb, these should not be merged.
            m[d != d_ngb] = 0

            # Is there anything left to merge?
            if m.sum() == 0:
                continue
            offset_ngb = offsets[ngb_id]

            # Apply the mask to the labels.
            o, o_ngb = o[m], o_ngb[m]

            # Apply the offsets.
            o += offset_block
            o_ngb += offset_ngb

            # Compute the merge labels for this face by concatenation and unique.
            to_merge = np.concatenate([o[:, None], o_ngb[:, None]], axis=1)
            to_merge = np.unique(to_merge, axis=0)
            if to_merge.size > 0:
                merge_labels.append(to_merge)

        if len(merge_labels) > 0:
            return np.concatenate(merge_labels, axis=0)
        else:
            return None

    # Compute the merge ids across all block faces.
    with futures.ThreadPoolExecutor(n_threads) as tp:
        merge_labels = list(tqdm(
            tp.map(_merge_block_faces, range(n_blocks)), total=n_blocks,
            desc="Merge labels across block faces", disable=not verbose
        ))

    n_elements = max_id + 1
    merge_labels = [res for res in merge_labels if res is not None]
    if len(merge_labels) == 0:
        return np.arange(n_elements, dtype=out.dtype)

    merge_labels = np.concatenate(merge_labels, axis=0)

    # Merge labels via union find.
    ufd = nufd.ufd(n_elements)
    ufd.merge(merge_labels)

    # Get the new labels from the ufd.
    old_labels = np.arange(n_elements, dtype=out.dtype)
    new_labels = ufd.find(old_labels)
    if with_background:
        assert new_labels[0] == 0
    # Relabel the new labels consecutively and return them.
    return relabel_sequential(new_labels)[0]


def write_mapping(out, mask, offsets, mapping, with_background, blocking, n_threads, verbose):
    """@private
    """
    n_blocks = blocking.numberOfBlocks

    # Compute the connected component for one block.
    @threadpool_limits.wrap(limits=1)  # Restrict the numpy threadpool to 1 to avoid oversubscription.
    def _write_block(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # Check if we have a mask and if we do, if we have pixels in the mask.
        if mask is not None:
            m = mask[bb].astype("bool")
            if m.sum() == 0:
                return None
        offset = offsets[block_id]

        # Load the data from this block.
        d = out[bb]
        if mask is None:
            if with_background:
                d[d != 0] += offset
            else:
                d += offset
            d = nt.take(mapping, d)
        else:
            if with_background:
                m[d == 0] = 0
            values = (d[m] + offset)
            values = nt.take(mapping, values)
            d[m] = values

        out[bb] = d

    # Compute connected components for all blocks in parallel.
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_write_block, range(n_blocks)), total=n_blocks, desc="Write blocks", disable=not verbose
        ))

    return out


def label(
    data: ArrayLike,
    out: Optional[ArrayLike] = None,
    with_background: bool = True,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    mask: Optional[ArrayLike] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
    connectivity: int = 1,
) -> ArrayLike:
    """Label the input data in parallel.

    Applies blockwise connected component and merges the results over block boundaries.

    Args:
        data: Input data, numpy array or similar like h5py or zarr dataset.
        out: Output data. Note that `label` cannot be applied inplace.
        with_background: Whether to treat zero as background label.
        block_shape: Shape of the blocks to use for parallelisation,
            by default chunks of the input will be used, if available.
        n_threads: Number of threads, by default all available threads are used.
        mask: Mask to exclude data from the computation.
            Data not in the mask will be set to zero in the result.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.
        connectivity: The number of nearest neighbor hops to consider for connection.
            Currently only supports connectivity of 1.

    Returns:
        The labeled data.
    """
    if connectivity != 1:
        raise NotImplementedError(
            f"The only value for connectivity currently supported is 1, you passed {connectivity}."
        )

    if out is None:
        out = np.zeros(data.shape, dtype="uint64")

    if data.shape != out.shape:
        raise ValueError(f"Expect data and out of same shape, got {data.shape} and {out.shape}")

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi, n_threads)

    # 1.) Compute connected components for all blocks.
    out, offsets = cc_blocks(data, out, mask, blocking, with_background, n_threads=n_threads, verbose=verbose)

    # Turn block max labels into offsets.
    last_block_val = offsets[-1]
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    max_id = offsets[-1] + last_block_val

    # 2.) Merge the connected components along block boundaries.
    mapping = merge_blocks(data, out, mask, offsets,
                           blocking, max_id, with_background,
                           n_threads=n_threads, verbose=verbose)

    # 3.) Write the new new pixel labeling.
    out = write_mapping(out, mask, offsets,
                        mapping, with_background,
                        blocking, n_threads=n_threads, verbose=verbose)

    return out
