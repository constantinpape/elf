

# TODO add support for
# - ellipses
# - negative indices
# - ???
# could be inspired by
# https://github.com/constantinpape/z5/blob/master/src/python/module/z5py/shape_utils.py#L126
# https://github.com/clbarnes/h5py_like
def normalize_index(index, shape):
    """ Normalize index to be of same dimension as shape
    """
    if isinstance(index, slice):
        index = (index,)
    else:
        assert isinstance(index, tuple)
        assert len(index) <= len(shape)
        assert all(isinstance(ind, slice) for ind in index)

    if len(index) < len(shape):
        n_missing = len(shape) - len(index)
        index = index + n_missing * (slice(None),)
    index = tuple(slice(0 if ind.start is None else ind.start,
                        sh if ind.stop is None else ind.stop)
                  for ind, sh in zip(index, shape))
    return index
