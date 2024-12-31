from functools import partial
from numbers import Number
from typing import Optional, Tuple

import numpy as np
import nifty.tools as nt

from numpy.typing import ArrayLike

from ..util import sigma_to_halo
try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff


#
# transformation impls
#


def apply_transform_chunked(data, out, transform_coordinate, interpolate_coordinates, start, stop, blocking, sigma):
    """@private
    """
    ndim = data.ndim
    # initialise the chunk cache
    chunk_cache = {}
    chunks = blocking.blockShape

    # precompute for range check
    max_range = tuple(sh - 1 for sh in data.shape)

    if sigma is not None:
        halo = sigma_to_halo(sigma, 0)
        if isinstance(halo, Number):
            halo = ndim * (halo,)

    def _apply_coord(old_coord):
        coord = transform_coordinate(old_coord)

        # range check
        if any(co < 0 or co >= maxr for co, maxr in zip(coord, max_range)):
            return

        # get the coordinates to iterate over and the interpolation weights
        coords, weights = interpolate_coordinates(coord)

        # iterate over coordinates and compute the output value
        val = 0.
        for coord, weight in zip(coords, weights):
            chunk_id = blocking.coordinatesToBlockId(list(coord))
            chunk, offset = chunk_cache.get(chunk_id, (None, None))
            if chunk is None:

                chunk_pos = blocking.blockGridPosition(chunk_id)
                if sigma is None:
                    block = blocking.getBlock(chunk_id)
                    chunk_bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
                else:
                    block = blocking.getBlockWithHalo(chunk_id, list(halo))
                    chunk_bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin,
                                                                         block.outerBlock.end))

                chunk = data[chunk_bb]
                if sigma is not None:
                    chunk = ff.gaussianSmoothing(chunk, sigma)
                    inner_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlockLocal.begin,
                                                                         block.innerBlockLocal.end))
                    chunk = chunk[inner_bb]

                offset = [cp * ch for cp, ch in zip(chunk_pos, chunks)]
                chunk_cache[chunk_id] = (chunk, offset)

            chunk_coord = tuple(oc - of for oc, of in zip(coord, offset))
            val += weight * chunk[chunk_coord]

        out_coord = tuple(co - of for co, of in zip(old_coord, start))
        out[out_coord] = val

    if ndim == 2:
        for ii in range(start[0], stop[0]):
            for jj in range(start[1], stop[1]):
                old_coord = (ii, jj)
                _apply_coord(old_coord)
    elif ndim == 3:
        for ii in range(start[0], stop[0]):
            for jj in range(start[1], stop[1]):
                for kk in range(start[2], stop[2]):
                    old_coord = (ii, jj, kk)
                    _apply_coord(old_coord)
    else:
        raise ValueError("Invalid dimension %i" % ndim)

    return out


def apply_transform(data, out, transform_coordinate, interpolate_coordinates, start, stop):
    """@private
    """

    ndim = data.ndim
    # precompute for range check
    max_range = tuple(sh - 1 for sh in data.shape)

    def _apply_coord(old_coord):
        coord = transform_coordinate(old_coord)

        # range check
        if any(co < 0 or co >= maxr for co, maxr in zip(coord, max_range)):
            return

        # get the coordinates to iterate over and the interpolation weights
        coords, weights = interpolate_coordinates(coord, where_format=True)
        # compute the output value
        val = (weights * data[coords]).sum()

        out_coord = tuple(co - of for co, of in zip(old_coord, start))
        out[out_coord] = val

    if ndim == 2:
        for ii in range(start[0], stop[0]):
            for jj in range(start[1], stop[1]):
                old_coord = (ii, jj)
                _apply_coord(old_coord)
    elif ndim == 3:
        for ii in range(start[0], stop[0]):
            for jj in range(start[1], stop[1]):
                for kk in range(start[2], stop[2]):
                    old_coord = (ii, jj, kk)
                    _apply_coord(old_coord)
    else:
        raise ValueError("Invalid dimension %i" % ndim)

    return out


#
# Coordinate interpolation
#


# nearest neighbor sampling / order 0
def interpolate_nn(coord, where_format=False):
    """@private
    """
    if where_format:
        return tuple(np.array([round(co, 0)], dtype='uint64') for co in coord), np.ones(1)
    else:
        return [tuple(int(round(co, 0)) for co in coord)], [1.]


# linear sampling / order 1
def interpolate_linear(coord, where_format=False):
    """@private
    """
    # determine upper and lower coordinate bound
    ndim = len(coord)
    lower = [int(co) for co in coord]
    upper = [lo + 1 for lo in lower]

    # find all next neighbor coordiantes by choosing all upper/lower
    # combinations via bitshifts of 2 ** ndim
    coords = [[lower[d] if (i & (1 << (k - 1))) else upper[d]
               for d, k in enumerate(range(1, ndim + 1))]
              for i in range(2 ** ndim)]
    # compute the weights for the different coordinates
    weights = [np.prod([abs(1. - abs(co - sa)) for sa, co in zip(sampled, coord)])
               for sampled in coords]

    # cast to np.where coordinate format
    if where_format:
        coords = tuple(np.array([coo[d] for coo in coords], dtype='uint64')
                       for d in range(ndim))
        weights = np.array(weights)
    return coords, weights


# TODO implement the higher orders
# quadratic sampling / order 2
def interpolate_quadratic(coord, where_format=False):
    """@private
    """
    pass


# cubic sampling / order 3
def interpolate_cubic(coord, where_format=False):
    """@private
    """
    pass


# TODO support multichannel transformation -> don't transform first coordinate dimension
# prototype impl for on the fly coordinate, transformation of sub-volumes / bounding boxes
def transform_subvolume(
    data: ArrayLike,
    transform: callable,
    bb: Tuple[slice, ...],
    order: int = 0,
    fill_value: Number = 0,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """Apply transform transformation to subvolume.

    Args:
        data: The input data, can be a numpy array or another array-like object.
        transform: The coordinate transfotmation.
        bb: The bounding box into the output data.
        order: The interpolation order.
        fill_value: The output value for invald coordinates.
        sigma: The sigma value used for pre-smoothing the input in order to avoid aliasing effects.

    Returns:
        The transformed subvolume.
    """
    ndim = data.ndim
    if ndim not in (2, 3):
        raise ValueError("Only support 2 or 3 dimensional data, not %i dimensions" % ndim)
    chunks = getattr(data, "chunks", None)

    start = tuple(b.start for b in bb)
    stop = tuple(b.stop for b in bb)
    sub_shape = tuple(sto - sta for sta, sto in zip(start, stop))

    if chunks is None:
        # TODO apply smoothing to input if sigma is not None
        if sigma is not None:
            raise NotImplementedError("Pre-smoothing is currently not implemented.")
        _apply = apply_transform
    else:
        blocking = nt.blocking([0] * ndim, data.shape, chunks)
        _apply = partial(apply_transform_chunked, blocking=blocking, sigma=sigma)

    if order == 0:
        interpolate = interpolate_nn
    elif order == 1:
        interpolate = interpolate_linear
    else:
        raise NotImplementedError("Only interpolation with order < 2 is currently implemented.")

    out = np.full(sub_shape, fill_value, dtype=data.dtype)
    return _apply(data, out, transform, interpolate, start, stop)
