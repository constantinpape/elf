import multiprocessing
# would be nice to use dask, so that we can also run this on the cluster
from concurrent import futures
from typing import Optional, Tuple

import bioimage_cpp as bic
import numpy as np
from numpy.typing import ArrayLike
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
        block = blocking.get_block_with_halo(block_id, list(halo))

        outer_bb = tuple(slice(
            beg, end
        ) for beg, end in zip(block.outer_block.begin, block.outer_block.end))

        local_bb = tuple(slice(
            beg, end
        ) for beg, end in zip(block.inner_block_local.begin, block.inner_block_local.end))

        if mask is not None:
            block_mask = mask[outer_bb]
            inner_mask = block_mask[local_bb]
            if inner_mask.sum() == 0:
                return
        else:
            block_mask = None

        block_hmap, block_seeds = hmap[outer_bb], seeds[outer_bb]
        # bic.segmentation.watershed requires float32/float64 hmap and integer markers
        if block_hmap.dtype not in (np.float32, np.float64):
            block_hmap = block_hmap.astype("float32")
        if block_seeds.dtype not in (np.uint32, np.uint64, np.int32, np.int64):
            block_seeds = block_seeds.astype("uint32")
        ws = bic.segmentation.watershed(block_hmap, block_seeds, mask=block_mask)

        inner_bb = tuple(slice(
            beg, end
        ) for beg, end in zip(block.inner_block.begin, block.inner_block.end))

        out[inner_bb] = ws[local_bb]

    n_blocks = blocking.number_of_blocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(process_block, range(n_blocks)), total=n_blocks, desc="Seeded watershed", disable=not verbose
        ))

    return out
