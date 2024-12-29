import nifty.tools as nt
from ..util import normalize_index


def _is_chunk_aligned(shape, chunks):
    return all(sh % ch == 0 for sh, ch in zip(shape, chunks))


def get_blocking(data, block_shape, roi, n_threads):
    """@private
    """
    # check if we have a chunked data storage
    try:
        chunks = data.chunks
    except AttributeError:
        chunks = None

    # if block_shape was not passed, then set it to the chunks (if we have chunks)
    if block_shape is None:
        if chunks is None:
            raise ValueError("If block_shape is not given, the data needs to have a chunk attribute.")
        else:
            block_shape = data.chunks

    # if we have chunks then check if the blocks are chunk aligned.
    # otherwise we can only run single-threaded because write access would not be thread-safe
    if chunks is not None and not _is_chunk_aligned(block_shape, chunks):
        if n_threads > 1:
            raise ValueError(
                f"You have chosen the block shape {block_shape}, which is not aligned with the chunks {chunks}. "
                f"And you're trying to parallelize with {n_threads} threads. This is not thread-safe. "
                "Please choose a chunk-aligned block-shape or run single threaded."
            )

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

        # if we have a chunks then check if the roi is chunk algined.
        # otherwise we can only run single-threaded because write access would not be thread-safe.
        if chunks is not None and not _is_chunk_aligned(roi_begin, chunks):
            if n_threads > 1:
                raise ValueError(
                    f"You have chosen a roi with start {roi_begin}, which is not aligned with the chunks {chunks}. "
                    f"And you're trying to parallelize with {n_threads} threads. This is not thread-safe. "
                    "Please choose a chunk-aligned roi or run single threaded."
                )

    blocking = nt.blocking(roi_begin, roi_end, block_shape)
    return blocking
