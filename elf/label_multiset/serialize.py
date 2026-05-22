import struct
from typing import Tuple

import numpy as np
from .label_multiset import LabelMultiset


def deserialize_labels(serialization: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Deserialize summarized label array from multiset serialization.

    Args:
        serialization: Flat byte array with multiset serialization.
        shape: Shape of the multiset.

    Returns:
        The labels that were summarized by the multiset.
    """

    # number of sets is encoded as integer in the first 4 bytes
    pos = 0
    next_pos = 4
    size = struct.unpack(">i", serialization[pos:next_pos].tobytes())[0]

    # the argmax vector is encoded as long in the next 8 * size bytes
    pos = next_pos
    next_pos += 8 * size
    argmax = serialization[pos:next_pos]
    argmax = np.frombuffer(argmax.tobytes(), dtype=">q")

    return argmax.reshape(shape)


def deserialize_multiset(serialization: np.ndarray, shape: Tuple[int, ...]) -> LabelMultiset:
    """Deserialize label multiset.

    Args:
        serialization: Flat byte array with multiset serialization.
        shape: Shape of the multiset.

    Returns:
        The deserialized label multiset.
    """

    # number of sets is encoded as integer in the first 4 bytes
    pos = 0
    next_pos = 4
    size = struct.unpack(">i", serialization[pos:next_pos].tobytes())[0]

    # the argmax vector is encoded as long in the next 8 * size bytes
    pos = next_pos
    next_pos += 8 * size
    argmax = serialization[pos:next_pos]
    argmax = np.frombuffer(argmax.tobytes(), dtype=">q")

    # the byte offset vector is encoded as long in the next 4 * size bytes
    pos = next_pos
    next_pos += 4 * size
    offsets = serialization[pos:next_pos]
    offsets = np.frombuffer(offsets.tobytes(), dtype=">i")

    # compute the unique byte offsets and the inverse mapping
    byte_offsets, inverse_offsets = np.unique(offsets, return_inverse=True)

    # the data is encoded as byte buffer,
    # storing ids as longs and counts as ints
    data = serialization[next_pos:]
    assert byte_offsets[-1] < len(data)

    data_offsets = np.concatenate(
        [byte_offsets, np.array([len(data)], dtype=byte_offsets.dtype)]
    ).astype(np.int64)

    # each entry is laid out as: 4-byte <i n_elements header, then n_elements * 12 bytes
    # of (8-byte <q id, 4-byte <i count). Derive n_elements per entry from byte lengths.
    entry_byte_lens = np.diff(data_offsets)
    n_per_entry = (entry_byte_lens - 4) // 12
    total_elements = int(n_per_entry.sum())

    # build a boolean mask over `data` that hides the 4-byte header of each entry,
    # leaving a contiguous sequence of element records to reinterpret.
    header_idx = (data_offsets[:-1, None] + np.arange(4)[None, :]).ravel()
    keep_mask = np.ones(len(data), dtype=bool)
    keep_mask[header_idx] = False
    records = data[keep_mask].reshape(total_elements, 12)
    ids = np.frombuffer(records[:, :8].tobytes(), dtype="<q").astype("uint64")
    counts = np.frombuffer(records[:, 8:].tobytes(), dtype="<i").astype("int32")

    # compute the set offsets from byte offsets and per-entry element counts
    entry_offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(n_per_entry)[:-1]])
    assert len(entry_offsets) == len(data_offsets) - 1
    offsets = entry_offsets[inverse_offsets]

    return LabelMultiset(argmax, offsets, ids, counts, shape)


# apparently, we do not need to switch to fortran order for the
# serialization, but that should be double checked.
def serialize_multiset(multiset: LabelMultiset) -> np.ndarray:
    """Serialize label multiset serialization in imglib format.

    The multiset is serialized as follows:
    1.) number of sets / cells encoded as integer (4 bytes)
    2.) max label id for each set encoded as long (8 bytes * num_cells)
    3.) offset in bytes into the data array for each set encoded as int (4 bytes * num cells)
    4.) the data storing label ids / counts encoded as long / int (datalen in bytes)
    See also:
    https://github.com/saalfeldlab/imglib2-label-multisets/blob/master/src/main/java/net/imglib2/type/label/LabelMultisetTypeDownscaler.java#L176

    Args:
        multiset: The label multiset to serialze.

    Returns:
        The serialized label multiset as flat binary array.
    """
    size, n_entries, n_elements = multiset.size, multiset.n_entries, multiset.n_elements
    argmax, offsets, ids, counts = (multiset.argmax, multiset.offsets,
                                    multiset.ids, multiset.counts)
    # encode the argmax vector
    argmax = np.array(argmax, dtype=">q").tobytes()

    # merge and encode ids and counts.
    # the ids are stored as long, the counts as int (both little endian).
    ids = [struct.pack("<q", i) for i in ids]
    counts = [struct.pack("<i", c) for c in counts]
    # get list of the unique offsets to delineate entries
    offset_list = np.concatenate([np.unique(offsets), np.array([n_elements])]).astype("uint64")
    assert offset_list[-2] < n_elements, "%i, %i" % (offset_list[-2], n_elements)

    # zip entry_sizesm ids and counts into one list.
    # given ids and counts:
    # ids = [id1, id2, id3, ..., idN]
    # counts = [c1, c2, c3, ..., cN]
    # where (1, 2), (3), ..., (N) form multiset entries,
    # we want to obtain:
    # data = [[id1, c1, id2, c2], [id3, c3] ..., [idN, cN]]
    data = [[x for t in zip(ids[beg:end], counts[beg:end]) for x in t]
            for beg, end in zip(offset_list[:-1], offset_list[1:])]

    # encode the data. we also prepend the entry size encoded as int for java
    entry_sizes = [struct.pack("<i", es) for es in multiset.entry_sizes]
    data = [b"".join([es] + elem) for es, elem in zip(entry_sizes, data)]
    assert len(data) == n_entries

    # comupute the byte offsets for each entry in data
    data_offsets = np.cumsum([0] + [len(entry) for entry in data[:-1]])
    assert len(data_offsets) == n_entries

    # encode the data
    data = b"".join(data)

    # get the offsets in bytes and encode
    offsets = [data_offsets[off] for off in multiset.entry_offsets]
    offsets = np.array(offsets, dtype=">i").tobytes()

    # encode the number of sets
    size = struct.pack(">i", size)

    # combine to byte buffer for serialization
    serialization = size + argmax + offsets + data
    serialization = np.frombuffer(serialization, dtype="uint8")
    return serialization
