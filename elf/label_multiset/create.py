from math import ceil
import numpy as np
import nifty.tools as nt  # use other blocking than nifty.tools
from .label_multiset import MultisetBase, LabelMultiset


def create_multiset_from_labels(labels):
    """ Create label multiset from a regular label array.
    """
    # argmaxs per block = labels in our case
    argmax = labels.flatten()

    # ids and offsets
    ids, offsets = np.unique(labels, return_inverse=True)

    # counts (1 by definiition)
    counts = np.ones(len(ids), dtype='int32')

    multiset = LabelMultiset(argmax, offsets, ids, counts, labels.shape)
    return multiset


def create_multiset_from_multiset(multiset, scale_factor, restrict_set=-1):
    """ Create label multiset from another multiset or a multiset grid.
    """
    if not isinstance(multiset, MultisetBase):
        raise ValueError("Expect input derived from MultisetBase, got %s" % type(multiset))

    shape = multiset.shape
    blocking = nt.blocking([0] * len(shape), shape, scale_factor)
    n_blocks = blocking.numberOfBlocks

    ids, counts = [], []
    hashed = {}

    argmax, offsets = [], []
    current_offset = 0

    for block_id in range(n_blocks):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        bids, bcounts = multiset[bb]

        # apply restrict_set if specified
        if restrict_set > 0:
            count_sorted = np.argsort(bcounts)[::-1][:restrict_set]
            bids, bcounts = bids[count_sorted], bcounts[count_sorted]
            id_sorted = np.argsort(bids)
            bids, bcounts = bids[id_sorted], bcounts[id_sorted]

        # compute the argmax-label
        max_id = np.argmax(bcounts)
        max_label = bids[max_id]
        max_count = bcounts[max_id]
        argmax.append(max_label)

        # we use the argmax label + count as block hash
        myhash = hash((max_label, max_count))
        candidates = hashed.get(myhash, [])
        add_entry = True

        # check if we have this entry in the candidates already
        for offset in candidates:

            cids, ccounts = ids[offset], counts[offset]

            # yes we have this entry already -> just store its offset
            if np.array_equal(bids, cids) and np.array_equal(bcounts, ccounts):
                add_entry = False
                offsets.append(offset)
                break

        # we haven't found this entry
        # -> make a new entry and increase the offset count
        if add_entry:
            offsets.append(current_offset)
            ids.append(bids)
            counts.append(bcounts)

            # TODO how do we do this efficiently in python without two look-ups?
            # add entry to the hash
            if myhash in hashed:
                hashed[myhash].append(current_offset)
            else:
                hashed[myhash] = [current_offset]

            # increase the offset by 1 (we count offsets in terms of entries, NOT elements here)
            current_offset += 1

    # we have computed offsets w.r.t entries and ids, counts as list of lists[int]
    # label multiset expects offsets w.r.t elements and ids and counts as list[int]

    # compute offsets w.r.t elements
    element_offsets = np.cumsum([0] + [len(ii) for ii in ids[:-1]])
    offsets = element_offsets[offsets]
    # flatten id and count vectors
    ids = np.array([i for ii in ids for i in ii], dtype='uint64')
    counts = np.array([c for cc in counts for c in cc], dtype='int32')

    argmax = np.array(argmax, dtype='uint64')

    new_shape = tuple(int(ceil(sh / sc)) for sh, sc in zip(shape, scale_factor))
    return LabelMultiset(argmax, offsets, ids, counts, new_shape)
