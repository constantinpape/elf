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

    Args:
        index (int or slice or ellipsis or tuple[int or slice or ellipsis]): slices to be normalized

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
