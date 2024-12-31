from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

# import blosc for chunk compression
try:
    import blosc
except ImportError:
    blosc = None

from .base import WrapperBase
from ..util import normalize_index, squeeze_singletons, map_chunk_to_roi, chunks_overlapping_roi


# TODO implement compression
# TODO specify cache size in MB instead of num elements?
class Cache(MutableMapping):
    """ Base class for chunk cache.

    A child class must define `_store` (dict-like) and implement `delete_item_from_cache`.

    Args:
        max_cache_size: The maximal number of chunks to cache.
        compression: The compression method.
    """
    def __init__(self, max_cache_size: int, compression: Optional[str] = None):
        self._max_cache_size = max_cache_size
        if compression is not None:
            raise NotImplementedError

    @property
    def max_cache_size(self):
        return self._max_cache_size

    def __delitem__(self, key):
        del self._store[key]

    def __getitem__(self, key):
        return self._store[key]

    def __iter__(self):
        for k in self._store:
            yield k

    def __len__(self):
        return len(self._store)

    def __setitem__(self, key, item):
        if len(self._store) == self.max_cache_size:
            self.delete_item_from_cache()
        self._store[key] = item


class FIFOCache(Cache):
    """FIFO cache implementation, using an OrderedDict internally.

    Args:
        max_cache_size: The maximal number of chunks to cache.
        compression: The compression method.
    """
    def __init__(self, max_cache_size: int, compression: Optional[str] = None):
        super().__init__(max_cache_size, compression)
        self._store = OrderedDict()

    def delete_item_from_cache(self):
        last_key = next(reversed(self._store))
        del self._store[last_key]


# some cache implementations out there:
# dask:
# https://docs.dask.org/en/latest/caching.html
# zarr LRU cache:
# https://github.com/zarr-developers/zarr-python/blob/d330cdf69d5fe21604aaba8f93cd1cfaed486d58/zarr/storage.py#L1709
# and the ilastik / lazyflow implementation:
# https://github.com/ilastik/lazyflow/blob/master/lazyflow/operators/opBlockedArrayCache.py

# further improvement options:
# - chunk compression with blosc
#
# - more cache replacement strategies (LIFO, FIFO, LRU), see also:
#   https://en.wikipedia.org/wiki/Cache_replacement_policies
#
# - performance
# -- use dask delayed to parallelize __getitem__
# -- OR use daemon process to compres chunks
# -- mover impl to c++
class CachedVolume(WrapperBase):
    """Wrapper around array-like data to cache loaded chunks.

    This can improve the performance for data with high latency,
    for example data streamed from S3 or similar web protocols.

    Args:
        volume: The data to wrap.
        cache: The cache implementation.
        chunks: The chunk shape. By default the chunks of the wrapped data are used.
    """
    preload_internal_chunks = False  # TODO implement this and make param

    def __init__(self, volume: ArrayLike, cache: Cache, chunks: Optional[Tuple[int, ...]] = None):
        super().__init__(volume)

        try:
            self._internal_chunks = volume.chunks
        except TypeError:
            raise ValueError("""Expected a volume type that exposes the attribute 'chunks',
                             like h5py.Dataset or z5py.Dataset.""")
        self._chunks = self.internal_chunks if chunks is None else chunks

        # initialize the actual chunk cache
        if not isinstance(cache, Cache):
            raise ValueError("Expect cache to be of type elf.wrapper.Cache, not %s" % type(cache))
        self._cache = cache

    @property
    def chunks(self):
        return self._chunks

    @property
    def internal_chunks(self):
        return self._internal_chunks

    @property
    def cache(self):
        return self._cache

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
        roi_start, roi_stop = [b.start for b in index], [b.stop for b in index]

        # initialize the data
        out_shape = tuple(sto - sta for sta, sto in zip(roi_start, roi_stop))
        out = np.empty(out_shape, dtype=self.dtype)

        # determine all chunks overlapping with `index`
        overlapping_chunks = chunks_overlapping_roi(index, self.chunks)

        # get all chunks that are in the cache and collect chunks not in the cache
        not_cached = []
        for chunk_id in overlapping_chunks:
            chunk_data = self.cache.get(chunk_id, None)
            if chunk_data is None:
                not_cached.append(chunk_id)
            else:
                # get corresponding bounding boxes for out and chunk
                # chunk_index = self._blocking.blockGridPosition(chunk_id)
                chunk_bb, out_bb = map_chunk_to_roi(chunk_id, index, self.chunks)
                out[out_bb] = chunk_data[chunk_bb]

        # check if we preload internal chunks
        if self.preload_internal_chunks:
            # find all the internal chunks we need to load
            internal_chunks = []
            for chunk_id in not_cached:
                chunk_roi = tuple(slice(cid * ch, min((cid + 1) * ch, sh))
                                  for cid, ch, sh in zip(chunk_id, self.chunks, self.shape))
                internal_chunks.extend(list(chunks_overlapping_roi(chunk_roi, self.internal_chunks)))
            # TODO
            # read all internal chunks, map them to chunks, write to data and put chunks on the cache

        # otherwise, we just load the missing chunks
        else:
            for chunk_id in not_cached:
                chunk_roi = tuple(slice(cid * ch, min((cid + 1) * ch, sh))
                                  for cid, ch, sh in zip(chunk_id, self.chunks, self.shape))
                chunk_data = self.volume[chunk_roi]
                self.cache[chunk_id] = chunk_data

                # get corresponding bounding boxes for out and chunk
                chunk_bb, out_bb = map_chunk_to_roi(chunk_id, index, self.chunks)
                out[out_bb] = chunk_data[chunk_bb]

        return squeeze_singletons(out, to_squeeze)
