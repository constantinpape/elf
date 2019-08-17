import numbers


def slice_to_start_stop(s, size):
    """For a single dimension with a given size, normalize slice to size.
     Returns slice(None, 0) if slice is invalid."""
    if s.step not in (None, 1):
        raise ValueError('Nontrivial steps are not supported')

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


def int_to_start_stop(i, size):
    """For a single dimension with a given size, turn an int into slice(start, stop)
    pair."""
    if -size < i < 0:
        start = i + size
    elif i >= size or i < -size:
        raise ValueError('Index ({}) out of range (0-{})'.format(i, size - 1))
    else:
        start = i
    return slice(start, start + 1)


# For now, I have copied the z5 implementation:
# https://github.com/constantinpape/z5/blob/master/src/python/module/z5py/shape_utils.py#L126
# But it's worth taking a look at @clbarnes more general implementation too
# https://github.com/clbarnes/h5py_like
def normalize_index(index, shape):
    """ Normalize index to shape.

    Normalize input, which can be a slice or a tuple of slices / ellipsis to
    be of same length as shape and be in bounds of shape.

    Argumentss:
        index [int or slice or ellipsis or tuple[int or slice or ellipsis]]: slices to be normalized

    Returns:
        tuple[slice]: normalized slices (start and stop are both non-None)
        tuple[int]: which singleton dimensions should be squeezed out
    """
    type_msg = 'Advanced selection inappropriate. ' \
               'Only numbers, slices (`:`), and ellipsis (`...`) are valid indices (or tuples thereof)'

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


def squeeze_singletons(item, to_squeeze):
    """ Squeeze singletons in item.

    This should be used with `normalize_index` like so:
    ```
    index, to_squeeze = normalize_index(index, data.shape)
    out = data[index]
    out = squeeze_singletons(out, to_squeeze)
    ```
    """
    if len(to_squeeze) == len(item.shape):
        return item.flatten()[0]
    elif to_squeeze:
        return item.squeeze(to_squeeze)
    else:
        return item


def map_chunk_to_roi(chunk_id, roi, chunks):
    """ Given a chunk id, roi and the chunk shape, determine the coordinate
    overlap, both in the roi's and the chunk's coordinate sytem.

    Arguments:
        chunk_id [listlike[int]] - the nd chunk index.
        roi [tuple[slices]] - the region of interest
        chunks [tuple] - the chunk shape
    Returns:
        tuple[slice] - overlap of the chuk and roi in chunk coordinates
        tuple[slice] - overlap of the chuk and roi in roi coordinates
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
