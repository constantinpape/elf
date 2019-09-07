import numpy as np
import nifty.tools as nt  # use other blocking than nifty.tools
from ..util import normalize_index, chunks_overlapping_roi, map_chunk_to_roi


class MultisetBase:
    def __init__(self, shape):
        self._shape = tuple(shape)
        self._size = int(np.prod(list(shape)))

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return self._size


class LabelMultiset(MultisetBase):
    """ Implement label multiset similar to
    https://github.com/saalfeldlab/imglib2-label-multisets.

    TODO explain member variables (esp. n_entries and n_elements) and usage
    """

    def __init__(self, argmax, offsets, ids, counts, shape):
        super().__init__(shape)

        if len(argmax) != len(offsets) != self.size:
            raise ValueError("Shape, argmax and offset do not match: %i %i %i" % (len(argmax),
                                                                                  len(offsets),
                                                                                  self.size))
        self.argmax = argmax
        self.offsets = offsets.astype('uint64')

        if len(ids) != len(counts):
            raise ValueError("Ids and counts do not match: %i, %i" % (len(ids), len(counts)))
        self.ids = ids
        self.counts = counts
        self.n_elements = len(self.ids)

        # compute the unique-offsets (= corresponding to entries) and the offsets
        # w.r.t entries instead of elements
        unique_offsets, self.entry_offsets = np.unique(self.offsets, return_inverse=True)
        if unique_offsets[-1] >= self.n_elements:
            raise ValueError("Elements and offsets do not match: %i, %i" % (self.n_elements,
                                                                            unique_offsets[-1]))
        self.n_entries = len(unique_offsets)
        # compute size of the entries from unique offsets
        unique_offsets = np.concatenate([unique_offsets,
                                         np.array([self.n_elements])]).astype('uint64')
        self.entry_sizes = np.diff(unique_offsets)

    def __getitem__(self, key):
        index = normalize_index(key, self._shape)[0]
        # need to convert the nd slice into vector of flat indices for all points
        # of the grid defined by the slice

        index = np.array([ax.flatten() for ax in np.mgrid[index]])
        index = np.ravel_multi_index(index, self._shape)

        # get offsets and sizes of the entries
        offsets = self.offsets[index]
        sizes = self.entry_sizes[self.entry_offsets[index]]

        # get the ids and counts of the entries
        # TODO vectorize, maybe with scipy.sparse?
        id_dict = {}
        for off, size in zip(offsets, sizes):
            mids = self.ids[off:off+size]
            mcounts = self.counts[off:off+size]
            for i, c in zip(mids, mcounts):
                if i in id_dict:
                    id_dict[i] += c
                else:
                    id_dict[i] = c
        ids = np.array(list(id_dict.keys()), dtype='uint64')
        counts = np.array(list(id_dict.values()), dtype='int32')

        sorter = np.argsort(ids)
        ids = ids[sorter]
        counts = counts[sorter]

        return ids, counts


class LabelMultisetGrid(MultisetBase):
    def __init__(self, multisets, grid_positions, shape, chunks):
        super().__init__(shape)
        self._chunks = tuple(chunks)

        n_sets = len(multisets)
        if len(grid_positions) != n_sets:
            raise ValueError("Multisets and grid-positions do not match: %i, %i" % (len(grid_positions),
                                                                                    n_sets))
        # check and process the grid positions
        if n_sets > 1 and n_sets % 2 != 0:
            raise ValueError("Expect even number of multisets: %i" % n_sets)
        self.multisets, self.grid_shape = self.compute_multiset_vector(multisets, grid_positions)

    def compute_multiset_vector(self, multisets, grid_positions):
        """ Store multiset in list with C-Order.
        """
        n_sets = len(multisets)
        multiset_vector = n_sets * [None]

        blocking = nt.blocking(self.ndim * [0], self.shape, list(self.chunks))
        n_blocks = blocking.numberOfBlocks
        if n_blocks != n_sets:
            raise ValueError("Invalid grid: %i, %i" % (n_blocks, n_sets))

        # get the c-order positions
        positions = np.array([[gp[i] for gp in grid_positions] for i in range(self.ndim)],
                             dtype='int')
        grid_shape = tuple(blocking.blocksPerAxis)
        positions = np.ravel_multi_index(positions, grid_shape)
        if any(pos >= n_sets for pos in positions):
            raise ValueError("Invalid grid positions")

        # put multi-sets into vector and check shapes
        for pos in positions:
            mset = multisets[pos]
            block_shape = tuple(blocking.getBlock(pos).shape)
            if mset.shape != block_shape:
                raise ValueError("Invalid multiset shape: %s, %s" % (str(mset.shape),
                                                                     str(block_shape)))
            multiset_vector[pos] = mset

        if any(ms is None for ms in multiset_vector):
            raise ValueError("Not all grid-positions filled")
        return multiset_vector, grid_shape

    def __getitem__(self, key):
        index = normalize_index(key, self.shape)[0]
        grid_points = chunks_overlapping_roi(index, self.chunks)

        # TODO vectorize, maybe with scipy.sparse?
        id_dict = {}
        for grid_point in grid_points:
            bb = map_chunk_to_roi(grid_point, index, self.chunks)[0]
            grid_id = np.ravel_multi_index(np.array([[grid_point[i]] for i in range(self.ndim)]),
                                           self.grid_shape)[0]
            mids, mcounts = self.multisets[grid_id][bb]
            for i, c in zip(mids, mcounts):
                if i in id_dict:
                    id_dict[i] += c
                else:
                    id_dict[i] = c
        ids = np.array(list(id_dict.keys()), dtype='uint64')
        counts = np.array(list(id_dict.values()), dtype='int32')

        sorter = np.argsort(ids)
        ids = ids[sorter]
        counts = counts[sorter]

        return ids, counts

    @property
    def chunks(self):
        return self._chunks
