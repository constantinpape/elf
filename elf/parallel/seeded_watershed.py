import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures
from typing import Optional, Tuple

from numpy.typing import ArrayLike
from skimage.segmentation import watershed
from tqdm import tqdm

from elf.parallel.common import get_blocking


def seeded_watershed(
    hmap: ArrayLike,
    seeds: ArrayLike,
    out: ArrayLike,
    block_shape: Tuple[int, ...],
    halo: Tuple[int, ...],
    mask: Optional[ArrayLike] = None,
    n_threads: Optional[int] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
) -> ArrayLike:
    """Compute seeded watershed in parallel over blocks.

    Args:
        hmap: The heightmap for the watershed.
        seeds: The seeds for the watershed.
        out: The output for the watershed.
        block_shape: Shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available.
        halo: The halo for enlarging the blocks used for parallelization.
        mask: Mask to exclude data from the watershed.
        n_threads: Number of threads, by default all are used.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.

    Returns:
        The watershed output.
    """
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
