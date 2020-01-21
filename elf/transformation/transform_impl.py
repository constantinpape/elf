from functools import partial
import numpy as np
import nifty.tools as nt
# fom ..util import sigma_to_halo
# try:
#     import fastfilters as ff
# except ImportError:
#     import vigra.filters as ff


def apply_transform_chunked_2d(data, out,
                               transform_coordinate, sample_coordinates,
                               start, stop, blocking, sigma):
    # TODO in a more serious impl, would need to limit the cache size,
    # ideally using lru cache
    chunk_cache = {}
    chunks = blocking.blockShape

    # precompute for range check
    max_range = tuple(sh - 1 for sh in data.shape)

    # if sigma is not None:
    #   halo = sigma_to_halo(sigma, 0)

    for ii in range(start[0], stop[0]):
        for jj in range(start[1], stop[1]):
            old_coord = (ii, jj)
            coord = transform_coordinate(old_coord)

            # range check
            if any(co < 0 or co >= maxr for co, maxr in zip(coord, max_range)):
                continue

            # get the coordinates to iterate over and the interpolation weights
            coords, weights = sample_coordinates(coord)

            # iterate over coordinates and compute the output value
            val = 0.
            for coord, weight in zip(coords, weights):
                chunk_id = blocking.coordinatesToBlockId(list(coord))
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
                val += weight * chunk[chunk_coord]

            out_coord = tuple(co - of for co, of in zip(old_coord, start))
            out[out_coord] = val
    return out


def apply_transform_chunked_3d(data, out,
                               transform_coordinate, sample_coordinates,
                               start, stop, blocking, sigma):
    # TODO in a more serious impl, would need to limit the cache size
    # ideally using lru cache
    chunk_cache = {}
    chunks = blocking.blockShape

    # precompute for range check
    max_range = tuple(sh - 1 for sh in data.shape)

    for ii in range(start[0], stop[0]):
        for jj in range(start[1], stop[1]):
            for kk in range(start[2], stop[2]):
                old_coord = (ii, jj, kk)
                coord = transform_coordinate(old_coord)

                # range check
                if any(co < 0 or co >= maxr for co, maxr in zip(coord, max_range)):
                    continue

                # get the coordinates to iterate over and the interpolation weights
                coords, weights = sample_coordinates(coord)

                # iterate over coordinates and compute the output value
                val = 0.
                for coord, weight in zip(coords, weights):
                    chunk_id = blocking.coordinatesToBlockId(list(coord))
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
                    val += weight * chunk[chunk_coord]
                out_coord = tuple(co - of for co, of in zip(old_coord, start))
                out[out_coord] = val
    return out


def apply_transform_2d(data, out,
                       transform_coordinate, sample_coordinates,
                       start, stop):

    # precompute for range check
    max_range = tuple(sh - 1 for sh in data.shape)

    for ii in range(start[0], stop[0]):
        for jj in range(start[1], stop[1]):
            old_coord = (ii, jj)
            coord = transform_coordinate(old_coord)

            # range check
            if any(co < 0 or co >= maxr for co, maxr in zip(coord, max_range)):
                continue

            # get the coordinates to iterate over and the interpolation weights
            coords, weights = sample_coordinates(coord, where_format=True)
            # compute the output value
            val = (weights * data[coords]).sum()

            out_coord = tuple(co - of for co, of in zip(old_coord, start))
            out[out_coord] = val
    return out


def apply_transform_3d(data, out,
                       transform_coordinate, sample_coordinates,
                       start, stop):

    # precompute for range check
    max_range = tuple(sh - 1 for sh in data.shape)

    for ii in range(start[0], stop[0]):
        for jj in range(start[1], stop[1]):
            for kk in range(start[2], stop[2]):
                old_coord = (ii, jj, kk)
                coord = transform_coordinate(old_coord)

                # range check
                if any(co < 0 or co >= maxr for co, maxr in zip(coord, max_range)):
                    continue

                # get the coordinates to iterate over and the interpolation weights
                coords, weights = sample_coordinates(coord, where_format=True)
                # compute the output value
                val = (weights * data[coords]).sum()

                out_coord = tuple(co - of for co, of in zip(old_coord, start))
                out[out_coord] = val
    return out


# nearest neighbor sampling / order 0
def sample_nn(coord, where_format=False):
    if where_format:
        return tuple(np.array([round(co, 0)], dtype='uint64') for co in coord), np.ones(1)
    else:
        return [tuple(int(round(co, 0)) for co in coord)], [1.]


# TODO implement the higher orders
# linear sampling / order 1
def sample_linear(coord, where_format=False):
    pass


# quadratic sampling / order 2
def sample_quadratic(coord, where_format=False):
    pass


# cubic sampling / order 3
def sample_cubic(coord, where_format=False):
    pass


# TODO support multichannel transformation -> don't transform first coordinate dimension
# TODO support order > 0 and smoothing with sigma for anti-aliasing
# prototype impl for on the fly coordinate, transformation of sub-volumes / bounding boxes
def transform_subvolume(data, transform, bb,
                        order=0, fill_value=0, sigma=None):
    """ Apply transform transformation to subvolume.

    Arguments:
        data [array_like] - input data
        transform [callable] - coordinate transfotmation
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
    if sigma is not None:
        raise NotImplementedError("Pre-smoothing is currently not implemented.")

    start = tuple(b.start for b in bb)
    stop = tuple(b.stop for b in bb)
    sub_shape = tuple(sto - sta for sta, sto in zip(start, stop))

    if chunks is None:
        # TODO apply smoothing to input if sigma is not None
        _apply = apply_transform_2d if ndim == 2 else apply_transform_3d
    else:
        blocking = nt.blocking([0] * ndim, data.shape, chunks)
        _apply = apply_transform_chunked_2d if ndim == 2 else apply_transform_chunked_3d
        _apply = partial(_apply, blocking=blocking, sigma=sigma)

    if order == 0:
        sampler = sample_nn
    # elif order == 1:
    #     sampler = sample_linear
    else:
        raise NotImplementedError("Only interpolation with order <= 0 is currently implemented.")

    out = np.full(sub_shape, fill_value, dtype=data.dtype)
    out = _apply(data, out, transform, sampler, start, stop)
    return out
