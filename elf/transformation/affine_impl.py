from functools import partial
import numpy as np
import nifty.tools as nt
from .affine import transform_coordinate
# try:
#     import fastfilters as ff
# except ImportError:
#     import vigra.filters as ff


def apply_affine_chunked_2d(data, out, matrix, start, stop,
                            order, blocking, sigma):
    # TODO in a more serious impl, would need to limit the cache size,
    # ideally using lru cache
    chunk_cache = {}
    chunks = blocking.blockShape

    for ii in range(start[0], stop[0]):
        for jj in range(start[1], stop[1]):
            old_coord = (ii, jj)
            coord = transform_coordinate(old_coord, matrix)
            # TODO support more than nearest neighbor assignment
            coord = [int(round(co, 0)) for co in coord]

            # range check
            if any(co < 0 for co in coord) or any(co >= sh for co, sh in zip(coord, data.shape)):
                continue

            chunk_id = blocking.coordinatesToBlockId(coord)
            chunk, offset = chunk_cache.get(chunk_id, (None, None))
            if chunk is None:
                # NOTE only works for z5py, need to implement this for other backends
                # also, it's problematic to put this into c++ if we want to support arbitrary
                # backends, try numba / cython first?
                chunk_pos = blocking.blockGridPosition(chunk_id)
                chunk = data.read_chunk(chunk_pos)
                # TODO apply pre-smoothing to chunk if sigma is not None
                offset = [cp * ch for cp, ch in zip(chunk_pos, chunks)]
                chunk_cache[chunk_id] = (chunk, offset)

            chunk_coord = tuple(oc - of for oc, of in zip(coord, offset))
            val = chunk[chunk_coord]
            out_coord = tuple(co - of for co, of in zip(old_coord, start))
            out[out_coord] = val
    return out


def apply_affine_chunked_3d(data, out, matrix, start, stop,
                            order, blocking, sigma):
    # TODO in a more serious impl, would need to limit the cache size
    # ideally using lru cache
    chunk_cache = {}
    chunks = blocking.blockShape

    for ii in range(start[0], stop[0]):
        for jj in range(start[1], stop[1]):
            for kk in range(start[2], stop[2]):
                old_coord = (ii, jj, kk)
                coord = transform_coordinate(old_coord, matrix)

                # TODO support more than nearest neighbor assignment
                coord = [int(round(co, 0)) for co in coord]

                # range check
                if any(co < 0 for co in coord) or any(co >= sh
                                                      for co, sh in zip(coord, data.shape)):
                    continue

                chunk_id = blocking.coordinatesToBlockId(coord)
                chunk, offset = chunk_cache.get(chunk_id, None)
                if chunk_id is None:
                    # NOTE only works for z5py, need to implement this for other backends
                    # also, it's problematic to put this into c++ if we want to support arbitrary
                    # backends, try numba / cython first?
                    chunk_pos = blocking.blockGridPosition(chunk_id)
                    chunk = data.read_chunk(chunk_pos)
                    # TODO apply pre-smoothing to chunk if sigma is not None
                    offset = [cp * ch for cp, ch in zip(chunk_pos, chunks)]
                    chunk_cache[chunk_id] = (chunk, offset)

                chunk_coord = tuple(oc - of for oc, of in zip(coord, offset))
                val = chunk[chunk_coord]
                out_coord = tuple(co - of for co, of in zip(old_coord, start))
                out[out_coord] = val
    return out


def apply_affine_2d(data, out, matrix, start, stop, order):
    for ii in range(start[0], stop[0]):
        for jj in range(start[1], stop[1]):
            old_coord = (ii, jj)
            coord = transform_coordinate(old_coord, matrix)

            # TODO support more than nearest neighbor assignment
            coord = tuple(int(round(co, 0)) for co in coord)

            # range check
            if any(co < 0 for co in coord) or any(co >= sh for co, sh in zip(coord, data.shape)):
                continue

            val = data[coord]
            out_coord = tuple(co - of for co, of in zip(old_coord, start))
            out[out_coord] = val
    return out


def apply_affine_3d(data, out, matrix, start, stop, order):
    for ii in range(start[0], stop[0]):
        for jj in range(start[1], stop[1]):
            for kk in range(start[2], stop[2]):
                old_coord = (ii, jj, kk)
                coord = transform_coordinate(old_coord, matrix)

                # TODO support more than nearest neighbor assignment
                coord = tuple(int(round(co, 0)) for co in coord)

                # range check
                if any(co < 0 for co in coord) or any(co >= sh
                                                      for co, sh in zip(coord, data.shape)):
                    continue

                val = data[coord]
                out_coord = tuple(co - of for co, of in zip(old_coord, start))
                out[out_coord] = val
    return out


# TODO support order > 0 and smoothing with sigma for anti-aliasing
# prototype impl for on the fly affine transformation of
# sub-volumes / bounding boxes
def affine_transform_for_subvolume(data, matrix, bb,
                                   order=0, fill_value=0, sigma=None):
    """ Apply affine transformation to subvolume.

    Arguments:
        data [array_like] - input data
        matrix [np.ndarray] - 4x4 matrix defining the affine transformation
        bb [tuple[slice]] - bounding box into the output data
        order [int] - interpolation order (default: 0)
        fill_value [scalar] - output value for invald coordinates (default: 0)
        sigma [float] - sigma value used for pre-smoothing the input
            in order to avoid aliasing effects (default: None)
    """
    ndim = data.ndim
    if ndim not in (2, 3):
        raise ValueError("Only support 2 or 3 dimensional data, not %i dimensions" % ndim)

    chunks = getattr(data, 'chunks', None)
    if order > 0 or sigma is not None:
        raise NotImplementedError()

    start = tuple(b.start for b in bb)
    stop = tuple(b.stop for b in bb)
    sub_shape = tuple(sto - sta for sta, sto in zip(start, stop))

    if chunks is None:
        # TODO apply smoothing to input if sigma is not None
        _apply = apply_affine_2d if ndim == 2 else apply_affine_3d
    else:
        blocking = nt.blocking([0] * ndim, data.shape, chunks)
        _apply = apply_affine_chunked_2d if ndim == 2 else apply_affine_chunked_3d
        _apply = partial(_apply, blocking=blocking, sigma=sigma)

    out = np.full(sub_shape, fill_value, dtype=data.dtype)
    out = _apply(data, out, matrix, start, stop, order)
    return out
