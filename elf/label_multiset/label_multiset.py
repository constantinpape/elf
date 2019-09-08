import numpy as np
import nifty.tools as nt
from ..util import normalize_index


class LabelMultiset:
    """ Implement label multiset similar to
    https://github.com/saalfeldlab/imglib2-label-multisets.

    Label multisets summarize the ids and counts of label arrays.
    This implementation uses flat arrays to store `ids` and `counts`.
    The member variables `shape` and `size` refer to the summarized label array.
    of pixels. Further, `n_elements` refers to the number of elements
    (= len(ids) / len(counts)), `n_entries` refers to the number of unique multi-set entries.

    Arguments:
        argmax [np.ndarray] - flat array of len `size` holding max labels per set.
        offsets [np.ndarray] - flat array of len `size` holding offsets into
            `ids`/`counts` for each set.
        ids [np.ndarray] - flat array holding the summarized label ids.
        counts [np.ndarray] - flat array holding the summarized label counts.
    """

    def __init__(self, argmax, offsets, ids, counts, shape):
        self._shape = tuple(shape)
        self._size = int(np.prod(list(shape)))

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

        # get the flattened entry indices
        index = normalize_index(key, self._shape)[0]
        index = np.array([ax.flatten() for ax in np.mgrid[index]])
        index = np.ravel_multi_index(index, self._shape)

        # get offsets and sizes of the entries
        offsets = self.offsets[index]
        sizes = self.entry_sizes[self.entry_offsets[index]]

        # merge ids and counts with c++ helper from nifty
        ids, counts = nt.readSubset(offsets, sizes, self.ids, self.counts,
                                    argsort=True)
        return ids, counts

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return self._size
