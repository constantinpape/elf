import nifty.tools as nt
from ..util import normalize_index


def get_blocking(data, block_shape, roi):
    if block_shape is None:
        try:
            block_shape = data.chunks
        except AttributeError:
            msg = "If block_shape is not given, the data needs to have a chunk attribute."
            raise ValueError(msg)

    shape, ndim = data.shape, data.ndim
    if roi is None:
        roi_begin = ndim * [0]
        roi_end = shape
    else:
        roi_normalized, to_squeeze = normalize_index(roi, shape)
        if any(to_squeeze):
            raise ValueError("Invalid roi")
        roi_begin = [bb.start for bb in roi_normalized]
        roi_end = [bb.stop for bb in roi_normalized]
    blocking = nt.blocking(roi_begin, roi_end, block_shape)
    return blocking
