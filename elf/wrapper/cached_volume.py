import numpy as np
import nifty.tools as nt
# import blosc for chunk compression
try:
    import blosc
except ImportError:
    blosc = None

from .wrapper_base import WrapperBase
from ..util import normalize_index, squeeze_singletons


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
    replacement_strategies = ('fifo', 'lifo')

    """
    """
    def __init__(self, volume, chunks=None,
                 replacement_strategy='fifo', compression=None):
        super().__init__(volume)

        if replacement_strategy.lower() not in self.replacement_strategies:
            raise ValueError("Replacement strategy %s not supported" % replacement_strategy)
        self._replacement_strategy = replacement_strategy.lower()

        try:
            self._internal_chunks = volume.chunks
        except TypeError:
            raise ValueError("""Expected a volume type that exposes the attribute 'chunks',
                             like h5py.Dataset or z5py.Dataset.""")
        self._chunks = self.internal_chunks if chunks is None else chunks

        # TODO is there something better then a dict?
        # initialize the actual chunk cache
        self._cache = {}

        # TODO we use nifty blocking for now, but it would be
        # nice to have this functionality independent of nifty
        # and use a blocking implemented in elf.util (check with @k-dominik)
        # blockings for chunks and internal chunks
        self._blocking = nt.blocking([0] * self.ndim, self.shape, self.chunks)
        self._internal_blocking = nt.blocking([0] * self.ndim, self.shape, self.internal_chunks)

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
        overlapping_chunks = self._blocking.getBlockIdsOverlappingBoundingBox(roi_start,
                                                                              roi_stop)

        # get all chunks that are in the cache and collect chunks not in the cache
        not_cached = []
        for chunk_id in overlapping_chunks:
            chunk_data = self.cache.get(chunk_id, None)
            if chunk_data is None:
                not_cached.append(chunk_id)
            else:
                # TODO get corresponding bounding boxes in out and chunk
                out[''] = chunk_data['']

        # find all the internal chunks we need to load
        internal_chunks = []
        for chunk_id in not_cached:
            chunk_block = self._blocking.getBlock(chunk_id)
            chunk_start, chunk_stop = [start for start in chunk_block.begin], [stop for stop in chunk_block.end]
            internal_chunks.extend(self._internal_blocking.getBlockIdsOverlappingBoundingBox(chunk_start,
                                                                                             chunk_stop))

        # TODO read_chunk only works for z5py
        # TODO
        # read all internal chunks, map them to chunks, write to data and put chunks on the cache

        return squeeze_singletons(out, to_squeeze)
