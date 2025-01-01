import numbers
import sys

from itertools import product
from math import ceil
from typing import List, Literal, Sequence, Tuple, Union, TYPE_CHECKING

# Ellipsis as type annontation is only supported for python >= 3.11.
# To support earlier version we use Literal[...] as type annotation, according to ChatGPT that should work.
if sys.version_info.minor >= 11:
    EllipsisType = Ellipsis
else:
    EllipsisType = Literal[...]

if TYPE_CHECKING:
    import numpy as np


def slice_to_start_stop(s: slice, size: int) -> slice:
    """Normalize a slice so that its start and stop correspond to the positive coordinates for indexing.

    Returns slice(None, 0) if the slice is invalid.

    Args:
        s: The input slice.
        size: The size of the axis of the slice.

    Returns:
        The normalized slice with positive coordinates for start and stop.
    """
    if s.step not in (None, 1):
        raise ValueError("Nontrivial steps are not supported")

    if s.start is None:
        start = 0
    elif -size <= s.start < 0:
        start = size + s.start
    elif s.start < -size or s.start >= size:
        return slice(None, 0)
    else:
        start = s.start

    if s.stop is None or s.stop > size:
        stop = size
    elif s.stop < 0:
        stop = (size + s.stop)
    else:
        stop = s.stop

    if stop < 1:
        return slice(None, 0)

    return slice(start, stop)


def int_to_start_stop(i: int, size: int) -> slice:
    """Returns a slice with for corresponding to an integer coordinate.

    Args:
        i: The coordinate.
        size: The size of the axis for this coordinate.

    Returns:
        The slice corresponding to the input coordinate.
    """
    if -size < i < 0:
        start = i + size
    elif i >= size or i < -size:
        raise ValueError("Index ({}) out of range (0-{})".format(i, size - 1))
    else:
        start = i
    return slice(start, start + 1)


# For now, I have copied the z5 implementation:
# https://github.com/constantinpape/z5/blob/master/src/python/module/z5py/shape_utils.py#L126
# But it's worth taking a look at @clbarnes more general implementation too
# https://github.com/clbarnes/h5py_like
def normalize_index(
    index: Union[int, slice, EllipsisType, Tuple[Union[int, slice, EllipsisType], ...]],
    shape: Tuple[int, ...]
) -> Tuple[Tuple[slice, ...], Tuple[int, ...]]:
    """Normalize an index for a given shape, so that is expressed as tuple of slices with correct coordinates.

    The input index can be an integer coordinate, a slice or a tuple of slices / ellipsis.
    It will be returned as a tuple of slices of same length as the shape and will be normalized,
    so that its start and stop coordinates are positive and in bounds.

    Args:
        index: Index to be normalized.
        shape: The shape of the array-like object to be indexed.

    Returns:
        The normalized index.
        List containing singleton dimensions that should be squeezed after indexing.
    """
    type_msg = "Advanced selection inappropriate. " \
               "Only numbers, slices (`:`), and ellipsis (`...`) are valid indices (or tuples thereof)"

    if isinstance(index, tuple):
        slices_lst = list(index)
    elif isinstance(index, (numbers.Number, slice, type(Ellipsis))):
        slices_lst = [index]
    else:
        raise TypeError(type_msg)

    ndim = len(shape)
    if len([item for item in slices_lst if item != Ellipsis]) > ndim:
        raise TypeError("Argument sequence too long")
    elif len(slices_lst) < ndim and Ellipsis not in slices_lst:
        slices_lst.append(Ellipsis)

    normalized = []
    found_ellipsis = False
    squeeze = []
    for item in slices_lst:
        d = len(normalized)
        if isinstance(item, slice):
            normalized.append(slice_to_start_stop(item, shape[d]))
        elif isinstance(item, numbers.Number):
            squeeze.append(d)
            normalized.append(int_to_start_stop(int(item), shape[d]))
        elif isinstance(item, type(Ellipsis)):
            if found_ellipsis:
                raise ValueError("Only one ellipsis may be used")
            found_ellipsis = True
            while len(normalized) + (len(slices_lst) - d - 1) < ndim:
                normalized.append(slice(0, shape[len(normalized)]))
        else:
            raise TypeError(type_msg)
    return tuple(normalized), tuple(squeeze)


def squeeze_singletons(item: "np.ndarray", to_squeeze: Tuple[int, ...]) -> "np.ndarray":
    """Squeeze singleton dimensions in a numpy array.

    This should be used with `normalize_index` like so:
    ```
    index, to_squeeze = normalize_index(index, data.shape)
    out = data[index]
    out = squeeze_singletons(out, to_squeeze)
    ```

    Args:
        item: The input data.
        to_squeeze: The axes to squeeze.

    Returns:
        The data with squeezed dimensions.
    """
    if len(to_squeeze) == len(item.shape):
        return item.flatten()[0]
    elif to_squeeze:
        return item.squeeze(to_squeeze)
    else:
        return item


def map_chunk_to_roi(
    chunk_id: Sequence[int], roi: Tuple[slice, ...], chunks: Tuple[int, ...]
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    """Computes the overlap of a chunk with a region of interest.

    The overlap will be returned both in the (global) coordinate system that the
    ROI referes to and in the local coordinates system of the chunk.

    Args:
        chunk_id: The index of the chunk index, corresponding to its grid index.
        roi: The region of interest.
        chunks: The chunk shape.

    Returns:
        Overlap of the chunk and roi in chunk coordinates.
        Overlap of the chunk and roi in roi coordinates.
    """
    # block begins and ends
    block_begin = [cid * ch for cid, ch in zip(chunk_id, chunks)]
    block_end = [beg + ch for beg, ch in zip(block_begin, chunks)]

    # get roi begins and ends
    roi_begin = [rr.start for rr in roi]
    roi_end = [rr.stop for rr in roi]

    chunk_bb, roi_bb = [], []
    # iterate over dimensions and find the bb coordinates
    ndim = len(chunk_id)
    for dim in range(ndim):
        # calculate the difference between the block begin / end
        # and the roi begin / end
        off_diff = block_begin[dim] - roi_begin[dim]
        end_diff = roi_end[dim] - block_end[dim]

        # if the offset difference is negative, we are at a starting block
        # that is not completely overlapping
        # -> set all values accordingly
        if off_diff < 0:
            begin_in_roi = 0  # start block -> no local offset
            begin_in_block = -off_diff
            # if this block is the beginning block as well as the end block,
            # we need to adjust the local shape accordingly
            shape_in_roi = block_end[dim] - roi_begin[dim]\
                if block_end[dim] <= roi_end[dim] else roi_end[dim] - roi_begin[dim]

        # if the end difference is negative, we are at a last block
        # that is not completely overlapping
        # -> set all values accordingly
        elif end_diff < 0:
            begin_in_roi = block_begin[dim] - roi_begin[dim]
            begin_in_block = 0
            shape_in_roi = roi_end[dim] - block_begin[dim]

        # otherwise we are at a completely overlapping block
        else:
            begin_in_roi = block_begin[dim] - roi_begin[dim]
            begin_in_block = 0
            shape_in_roi = chunks[dim]

        # append to bbs
        chunk_bb.append(slice(begin_in_block, begin_in_block + shape_in_roi))
        roi_bb.append(slice(begin_in_roi, begin_in_roi + shape_in_roi))

    return tuple(chunk_bb), tuple(roi_bb)


def chunks_overlapping_roi(roi: Tuple[slice, ...], chunks: Tuple[int, ...]) -> Sequence[Tuple[int, ...]]:
    """Find all the chunk ids overlapping with a region of interest.

    Args:
        roi: The region of interest.
        chunks: The chunk shape.

    Returns:
        Sequence of chunk ids, where each chunk id is the nd grid coordinate of the chunk.
    """
    ranges = [range(rr.start // ch, rr.stop // ch if rr.stop % ch == 0 else rr.stop // ch + 1)
              for rr, ch in zip(roi, chunks)]
    return product(*ranges)


def downscale_shape(
    shape: Tuple[int, ...],
    scale_factor: Union[Tuple[int, ...], int],
    ceil_mode: bool = True,
) -> Tuple[int, ...]:
    """Compute new shape after downscaling a volume by given scale factor.

    Args:
        shape: The input shape.
        scale_factor: The scale factor used for downscaling.
        ceil_mode: Whether to apply ceil to output shape.

    Returns:
        The shape after downscaling.
    """
    scale_ = (scale_factor,) * len(shape) if isinstance(scale_factor, int) else scale_factor
    if ceil_mode:
        return tuple(sh // sf + int((sh % sf) != 0) for sh, sf in zip(shape, scale_))
    else:
        return tuple(sh // sf for sh, sf in zip(shape, scale_))


def sigma_to_halo(sigma: Union[float, Sequence[float]], order: int) -> Union[int, List[int]]:
    """Compute the halo value for applying an image filter in parallel.

    Based on:
    https://github.com/ukoethe/vigra/blob/master/include/vigra/multi_blockwise.hxx#L408

    Args:
        sigma: The sigma value of the filter.
        order: The order of the filter.

    Returns:
        The halo for enlarging blocks used for parallelization.
    """
    # NOTE it seems like the halo given here is not sufficient and the test in ilastik
    # for the reference implementation do not catch this because of insufficient block-shape:
    # https://github.com/ilastik/ilastik/blob/master/tests/test_workflows/carving/testCarvingTools.py
    # one way to deal with this is to introduce another multiplier to increase the halo,
    # but this should be investigated further!
    multiplier = 2
    if isinstance(sigma, numbers.Number):
        halo = multiplier * int(ceil(3.0 * sigma + 0.5 * order + 0.5))
    else:
        halo = [multiplier * int(ceil(3.0 * sig + 0.5 * order + 0.5)) for sig in sigma]
    return halo


def _make_checkerboard(blocking):
    blocks_a = [0]
    blocks_b = []
    processed_blocks = [0]
    ndim = len(blocking.blockShape)

    def recurse(current_block, insert_list):
        other_list = blocks_a if insert_list is blocks_b else blocks_b
        for dim in range(ndim):
            ngb_id = blocking.getNeighborId(current_block, dim, False)
            if ngb_id != -1 and (ngb_id not in processed_blocks):
                insert_list.append(ngb_id)
                processed_blocks.append(ngb_id)
                recurse(ngb_id, other_list)

    recurse(0, blocks_b)
    all_blocks = blocks_a + blocks_b
    expected = set(range(blocking.numberOfBlocks))
    assert len(all_blocks) == len(expected), "%i, %i" % (len(all_blocks), len(expected))
    assert len(set(all_blocks) - expected) == 0
    assert len(blocks_a) == len(blocks_b), "%i, %i" % (len(blocks_a), len(blocks_b))
    return blocks_a, blocks_b


def _make_checkerboard_with_roi(blocking, roi_begin, roi_end):

    # find the smallest roi coordinate
    block0 = blocking.coordinatesToBlockId(roi_begin)

    blocks_a = [block0]
    blocks_b = []
    processed_blocks = [block0]
    ndim = len(blocking.blockShape)

    blocks_in_roi = blocking.getBlockIdsOverlappingBoundingBox(roi_begin, roi_end)
    assert block0 in blocks_in_roi

    def recurse(current_block, insert_list):
        other_list = blocks_a if insert_list is blocks_b else blocks_b
        for dim in range(ndim):
            ngb_id = blocking.getNeighborId(current_block, dim, False)
            if (ngb_id != -1) and (ngb_id in blocks_in_roi) and (ngb_id not in processed_blocks):
                insert_list.append(ngb_id)
                processed_blocks.append(ngb_id)
                recurse(ngb_id, other_list)

    recurse(block0, blocks_b)
    all_blocks = blocks_a + blocks_b
    expected = set(blocks_in_roi)
    assert len(all_blocks) == len(expected), "%i, %i" % (len(all_blocks), len(expected))
    assert len(set(all_blocks) - expected) == 0
    assert len(blocks_a) == len(blocks_b), "%i, %i" % (len(blocks_a), len(blocks_b))
    return blocks_a, blocks_b


def divide_blocks_into_checkerboard(blocking, roi_begin=None, roi_end=None):
    """@private
    """
    assert (roi_begin is None) == (roi_end is None)
    if roi_begin is None:
        return _make_checkerboard(blocking)
    else:
        return _make_checkerboard_with_roi(blocking, roi_begin, roi_end)
