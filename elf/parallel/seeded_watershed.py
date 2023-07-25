import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures

from skimage.segmentation import watershed
from tqdm import tqdm

from elf.parallel.common import get_blocking


def seeded_watershed(hmap, seeds, out, block_shape, halo,
                     mask=None, n_threads=None, verbose=False, roi=None):
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(hmap, block_shape, roi, n_threads)

    def process_block(block_id):
        block = blocking.getBlockWithHalo(block_id, list(halo))

        outer_bb = tuple(slice(
            beg, end
        ) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))

        local_bb = tuple(slice(
            beg, end
        ) for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))

        if mask is not None:
            block_mask = mask[outer_bb]
            inner_mask = block_mask[local_bb]
            if inner_mask.sum() == 0:
                return
        else:
            block_mask = None

        block_hmap, block_seeds = hmap[outer_bb], seeds[outer_bb]
        ws = watershed(block_hmap, block_seeds, mask=block_mask)

        inner_bb = tuple(slice(
            beg, end
        ) for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))

        out[inner_bb] = ws[local_bb]

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(process_block, range(n_blocks)), total=n_blocks, desc="Seeded watershed", disable=not verbose
        ))

    return out
