from concurrent import futures

import numpy as np
from ..label_multiset import deserialize_labels
from ..util import (normalize_index, squeeze_singletons,
                    map_chunk_to_roi, chunks_overlapping_roi)


class LabelMultisetWrapper:
    """Wrapper class for a label multiset dataset so that it can be accessed via arbitrary slices.
    """
    def __init__(self, dataset):
        self._dataset = dataset
        self.n_threads = 1

    @property
    def dtype(self):
        return np.dtype("uint64")

    @property
    def ndim(self):
        return self._dataset.ndim

    @property
    def chunks(self):
        return self._dataset.chunks

    @property
    def shape(self):
        return self._dataset.shape

    @property
    def size(self):
        return self._dataset.size

    @property
    def attrs(self):
        return self._dataset.attrs

    def _load_roi(self, roi):
        # snap roi to grid
        chunk_ids = chunks_overlapping_roi(roi, self.chunks)

        # init data (dtype is hard-coded to uint64)
        roi_shape = tuple(rr.stop - rr.start for rr in roi)
        data = np.zeros(roi_shape, dtype="uint64")

        def load_chunk(chunk_id):
            chunk_shape = self._dataset.get_chunk_shape(chunk_id)
            chunk_data = self._dataset.read_chunk(chunk_id)
            if chunk_data is None:
                chunk_data = np.zeros(chunk_shape, dtype="uint64")
            else:
                chunk_data = deserialize_labels(chunk_data, chunk_shape)

            chunk_bb, out_bb = map_chunk_to_roi(chunk_id, roi, self.chunks)
            data[out_bb] = chunk_data[chunk_bb]

        if self.n_threads > 1:
            with futures.ThreadPoolExecutor(self.n_threads) as tp:
                tasks = [tp.submit(load_chunk, cid) for cid in chunk_ids]
                [t.result() for t in tasks]
        else:
            [load_chunk(cid) for cid in chunk_ids]

        return data

    def __getitem__(self, key):
        roi, to_squeeze = normalize_index(key, self.shape)
        return squeeze_singletons(self._load_roi(roi), to_squeeze)
