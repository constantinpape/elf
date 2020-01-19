import numpy as np
import nifty.tools as nt
from .affine import transform_coordinate


def apply_affine_2d(data, out, blocking, matrix, start, stop):
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
                offset = [cp * ch for cp, ch in zip(chunk_pos, chunks)]
                chunk_cache[chunk_id] = (chunk, offset)

            chunk_coord = tuple(oc - of for oc, of in zip(coord, offset))
            val = chunk[chunk_coord]
            out[old_coord] = val
    return out


def apply_affine_3d(data, out, blocking, matrix, start, stop):
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
                coord = tuple(int(round(co, 0)) for co in coord)

                # range check
                if any(co < 0 for co in coord) or any(co >= sh for co, sh in zip(coord, data.shape)):
                    continue

                chunk_id = blocking.coordinatesToBlockId(coord)
                chunk, offset = chunk_cache.get(chunk_id, None)
                if chunk_id is None:
                    # NOTE only works for z5py, need to implement this for other backends
                    # also, it's problematic to put this into c++ if we want to support arbitrary
                    # backends, try numba / cython first?
                    chunk_pos = blocking.blockGridPosition(chunk_id)
                    chunk = data.read_chunk(chunk_pos)
                    offset = [cp * ch for cp, ch in zip(chunk_pos, chunks)]
                    chunk_cache[chunk_id] = (chunk, offset)

                chunk_coord = tuple(oc - of for oc, of in zip(coord, offset))
                val = chunk[chunk_coord]
                out[old_coord] = val
    return out


def apply_with_chunks(data, matrix, out, chunks, start, stop):
    ndim = data.ndim
    blocking = nt.blocking([0] * ndim, data.shape, chunks)
    # TODO nd loop ?
    if ndim == 2:
        out = apply_affine_2d(data, out, blocking, matrix, start, stop)
    elif ndim == 3:
        out = apply_affine_3d(data, out, blocking, matrix, start, stop)
    else:
        raise ValueError
    return out


# prototype impl for on the fly affine transformation of
# sub-volumes / bounding boxes
def apply_affine_for_subvolume(data, matrix, bb):
    chunks = getattr(data, 'chunks', None)
    assert chunks is not None
    start = tuple(b.start for b in bb)
    stop = tuple(b.stop for b in bb)
    sub_shape = tuple(sto - sta for sta, sto in zip(start, stop))
    out = np.zeros(sub_shape, dtype=data.dtype)
    out = apply_with_chunks(data, matrix, out, chunks, start, stop)
    return out
