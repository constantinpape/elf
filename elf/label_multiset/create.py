from typing import Sequence, Tuple

import numpy as np
import nifty.tools as nt
from .label_multiset import LabelMultiset
from ..util import downscale_shape


def create_multiset_from_labels(labels: np.ndarray) -> LabelMultiset:
    """Create label multiset from a regular label array.

    Args:
        labels: Label array to summarize in the label multiset.

    Returns:
        The label multiset.
    """
    # argmaxs per block = labels in our case
    argmax = labels.flatten()

    # ids and offsets
    ids, offsets = np.unique(labels, return_inverse=True)

    # counts (1 by definiition)
    counts = np.ones(len(ids), dtype="int32")

    return LabelMultiset(argmax, offsets, ids, counts, labels.shape)


def downsample_multiset(
    multiset: LabelMultiset, scale_factor: Tuple[int, ...], restrict_set: int = -1
) -> LabelMultiset:
    """Downsample a label multiset.

    Args:
        multiset: The input label multiset.
        scale_factor: The scale factor for downsampling.
        restrict_set: The maximum entry length of the downsampled multiset.
            The default value (-1) means that the entry length is not restricted.

    Returns:
        The downsampled label multiset.
    """
    if not isinstance(multiset, LabelMultiset):
        raise ValueError("Expect input derived from MultisetBase, got %s" % type(multiset))

    shape = multiset.shape
    blocking = nt.blocking([0] * len(shape), shape, scale_factor)

    argmax, offsets, ids, counts = nt.downsampleMultiset(
        blocking, multiset.offsets, multiset.entry_sizes, multiset.entry_offsets,
        multiset.ids, multiset.counts, restrict_set
    )
    new_shape = downscale_shape(shape, scale_factor)
    return LabelMultiset(argmax, offsets, ids, counts, new_shape)


def merge_multisets(
    multisets: Sequence[LabelMultiset],
    grid_positions: Sequence[Tuple[int, ...]],
    shape: Tuple[int, ...],
    chunks: Tuple[int, ...],
) -> LabelMultiset:
    """Merge label multisets aranged in grid.

    Args:
        multisets: List of label multisets aranged in grid that will be merged.
        grid_positions: Grid coordinates of the input multisets.
        shape: Shape of the resulting multiset / grid.
        chunks: Chunk shape = default shape of input multiset.

    Returns:
        The merged label multiset.
    """
    if not isinstance(multisets, (tuple, list)) and\
       not all(isinstance(ms, LabelMultiset) for ms in multisets):
        raise ValueError("Expect list or tuple of LabelMultiset")

    # arrange multisets according to the grid
    multisets, blocking = _compute_multiset_vector(multisets, grid_positions, shape, chunks)

    new_size = int(np.prod(shape))
    argmax = np.zeros(new_size, dtype="uint64")
    offsets = np.zeros(new_size, dtype="uint64")

    def get_indices(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        new_indices = np.array([ax.flatten() for ax in np.mgrid[bb]])
        new_indices = np.ravel_multi_index(new_indices, shape)
        return new_indices

    # create merge helper initialized with multisets[0]
    ms = multisets[0]
    merge_helper = nt.MultisetMerger(np.unique(ms.offsets), ms.entry_sizes, ms.ids, ms.counts)
    # map offsets and argmax for first multiset
    new_indices = get_indices(0)
    argmax[new_indices] = ms.argmax
    offsets[new_indices] = ms.offsets

    for block_id, ms in enumerate(multisets[1:], 1):
        # map to the new indices
        new_indices = get_indices(block_id)
        # map argmax
        argmax[new_indices] = ms.argmax

        # update the merge helper
        new_offsets = merge_helper.update(np.unique(ms.offsets), ms.entry_sizes,
                                          ms.ids, ms.counts, ms.entry_offsets)
        offsets[new_indices] = new_offsets

    ids = merge_helper.get_ids()
    counts = merge_helper.get_counts()
    return LabelMultiset(argmax, offsets, ids, counts, shape)


def _compute_multiset_vector(multisets, grid_positions, shape, chunks):
    """Arange the multisets in c-order.
    """
    n_sets = len(multisets)
    ndim = len(shape)
    multiset_vector = n_sets * [None]

    blocking = nt.blocking(ndim * [0], shape, list(chunks))
    n_blocks = blocking.numberOfBlocks
    if n_blocks != n_sets:
        raise ValueError("Invalid grid: %i, %i" % (n_blocks, n_sets))

    # get the c-order positions
    positions = np.array([[gp[i] for gp in grid_positions] for i in range(ndim)], dtype="int")
    grid_shape = tuple(blocking.blocksPerAxis)
    positions = np.ravel_multi_index(positions, grid_shape)
    if any(pos >= n_sets for pos in positions):
        raise ValueError("Invalid grid positions")

    # put multi-sets into vector and check shapes
    for pos in positions:
        mset = multisets[pos]
        block_shape = tuple(blocking.getBlock(pos).shape)
        if mset.shape != block_shape:
            raise ValueError("Invalid multiset shape: %s, %s" % (str(mset.shape), str(block_shape)))
        multiset_vector[pos] = mset

    if any(ms is None for ms in multiset_vector):
        raise ValueError("Not all grid-positions filled")
    return multiset_vector, blocking
