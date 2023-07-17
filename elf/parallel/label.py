import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures

from tqdm import tqdm
from skimage.measure import label as label_impl
from skimage.segmentation import relabel_sequential

import nifty.tools as nt
import nifty.ufd as nufd
from .common import get_blocking

from elf.util import set_numpy_threads
set_numpy_threads(1)
import numpy as np


def cc_blocks(data, out, mask, blocking, with_background,
              n_threads, verbose):
    n_blocks = blocking.numberOfBlocks

    # compute the connected component for one block
    def _cc_block(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype("bool")
            if m.sum() == 0:
                return 0

        # load the data from this block
        d = data[bb].copy()

        # determine the background value
        bg_val = 0 if with_background else int(d.max() + 1)

        # set mask to background value
        if mask is not None:
            d[~m] = bg_val

        d = label_impl(d, background=bg_val, connectivity=1)

        out[bb] = d
        return int(d.max())

    # compute connected components for all blocks in parallel
    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            block_max_labels = list(tqdm(
                tp.map(_cc_block, range(n_blocks)),
                total=n_blocks, desc="Label all sub-blocks")
            )
        else:
            block_max_labels = list(tp.map(_cc_block, range(n_blocks)))

    return out, block_max_labels


def merge_blocks(data, out, mask, offsets,
                 blocking, max_id, with_background,
                 n_threads, verbose):
    n_blocks = blocking.numberOfBlocks
    ndim = out.ndim

    def _merge_block_faces(block_id):
        block = blocking.getBlock(block_id)
        offset_block = offsets[block_id]

        merge_labels = []
        # for each axis, load the face with the lower block neighbor and compute the merge labels
        for axis in range(ndim):
            ngb_id = blocking.getNeighborId(block_id, axis, lower=True)
            if ngb_id == -1:
                continue
            ngb_block = blocking.getBlock(ngb_id)

            # make the bounding box for both faces and load the segmentation for it
            face = tuple(slice(beg, end) if d != axis else slice(beg, beg + 1)
                         for d, (beg, end) in enumerate(zip(block.begin, block.end)))
            ngb_face = tuple(slice(beg, end) if d != axis else slice(end - 1, end)
                             for d, (beg, end) in enumerate(zip(ngb_block.begin,
                                                                ngb_block.end)))

            # load and combine the mask for bot faces
            if mask is not None:
                m = np.logical_and(out[face], out[ngb_face])
                if m.sum() == 0:
                    continue

            # load the initial labels for both faces
            d, d_ngb = data[face], data[ngb_face]
            assert d.shape == d_ngb.shape

            # load the intermediate result for both faces
            o, o_ngb = out[face], out[ngb_face]
            assert o.shape == o_ngb.shape == d.shape

            # allocate full mask if we don't have a mask dataset
            if mask is None:
                m = np.ones_like(d, dtype="bool")

            # mask zero label if we have background
            if with_background:
                m[d == 0] = 0
                m[d_ngb == 0] = 0

            # mask pixels of the face where d != d_ngb, these should not be merged!
            m[d != d_ngb] = 0

            # is there anything left to merge?
            if m.sum() == 0:
                continue
            offset_ngb = offsets[ngb_id]

            # apply the mask to the labels
            o, o_ngb = o[m], o_ngb[m]

            # apply the offsets
            o += offset_block
            o_ngb += offset_ngb

            # compute the merge labels for this face
            # by concatenation and unique
            to_merge = np.concatenate([o[:, None], o_ngb[:, None]], axis=1)
            to_merge = np.unique(to_merge, axis=0)
            if to_merge.size > 0:
                merge_labels.append(to_merge)

        if len(merge_labels) > 0:
            return np.concatenate(merge_labels, axis=0)
        else:
            return None

    # compute the merge ids across all block faces
    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            merge_labels = list(tqdm(tp.map(_merge_block_faces, range(n_blocks)),
                                     total=n_blocks, desc="Merge labels across block faces"))
        else:
            merge_labels = tp.map(_merge_block_faces, range(n_blocks))
    merge_labels = [res for res in merge_labels if res is not None]
    if len(merge_labels) == 0:
        return np.arange(max_id + 1, dtype=out.dtype)

    merge_labels = np.concatenate(merge_labels, axis=0)
    # merge labels via union find
    old_labels = np.arange(max_id + 1, dtype=out.dtype)
    ufd = nufd.boost_ufd(old_labels)
    ufd.merge(merge_labels)

    # get the new labels from the ufd
    new_labels = ufd.find(old_labels)
    if with_background:
        assert new_labels[0] == 0
    # relabel the new labels consecutively and return them
    return relabel_sequential(new_labels)[0]


def write_mapping(out, mask, offsets, mapping,
                  with_background, blocking,
                  n_threads, verbose):
    n_blocks = blocking.numberOfBlocks

    # compute the connected component for one block
    def _write_block(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        # check if we have a mask and if we do if we
        # have pixels in the mask
        if mask is not None:
            m = mask[bb].astype("bool")
            if m.sum() == 0:
                return None
        offset = offsets[block_id]

        # load the data from this block
        d = out[bb]
        if mask is None:
            if with_background:
                d[d != 0] += offset
            else:
                d += offset
            d = nt.take(mapping, d)
        else:
            if with_background:
                m[d == 0] = 0
            values = (d[m] + offset)
            values = nt.take(mapping, values)
            d[m] = values

        out[bb] = d

    # compute connected components for all blocks in parallel
    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            list(tqdm(tp.map(_write_block, range(n_blocks)), total=n_blocks, desc="Write blocks"))
        else:
            tp.map(_write_block, range(n_blocks))

    return out


def label(data, out, with_background=True, block_shape=None,
          n_threads=None, mask=None, verbose=False, roi=None, connectivity=1):
    """Label the data in parallel by applying blockwise connected component and
    merging the results over block boundaries.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        out [array_like] - output data (label cannot be applied inplace)
        with_background [bool] - whether to treat zero as background label (default: True)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation.
            Data not in the mask will be set to zero in the result. (default: None)
        verbose [bool] - verbosity flag (default: False)
        roi [tuple[slice]] - region of interest for this computation (default: None)
        connectivity [int] - the number of nearest neighbor hops to consider for connection.
            Currently only supports connectivity of 1. (default: 1)
    Returns:
        array_like - the output data
    """
    if connectivity != 1:
        raise NotImplementedError(
            f"The only value for connectivity currently supported is 1, you passed {connectivity}."
        )

    if data.shape != out.shape:
        raise ValueError(f"Expect data and out of same shape, got {data.shape} and {out.shape}")

    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(data, block_shape, roi)

    # 1.) compute connected components for all blocks
    out, offsets = cc_blocks(data, out, mask, blocking, with_background, n_threads, verbose)

    # turn block max labels into offsets
    last_block_val = offsets[-1]
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    max_id = offsets[-1] + last_block_val

    # 2.) merge the connected components along block boundaries
    mapping = merge_blocks(data, out, mask, offsets,
                           blocking, max_id, with_background,
                           n_threads, verbose)

    # 3.) write the new new pixel labeling
    out = write_mapping(out, mask, offsets,
                        mapping, with_background,
                        blocking, n_threads, verbose)

    return out
