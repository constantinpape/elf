import multiprocessing
from concurrent import futures
from typing import List, Optional, Tuple

import nifty.tools as nt
import numpy as np
import vigra

from nifty.filters import nonMaximumDistanceSuppression
from numpy.typing import ArrayLike
from tqdm import tqdm

from ..util import divide_blocks_into_checkerboard

try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff


def watershed(
    input_: np.ndarray, seeds: np.ndarray, size_filter: int = 0, exclude: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    """Compute seeded watershed.

    Args:
        input_: The input height map.
        seeds: The seed map.
        size_filter: The minimal segment size.
        exclude: List of seed ids that will not be size filtered.

    Returns:
        The watershed segmentation.
        The max id of the watershed segmentation.
    """
    ws, max_id = vigra.analysis.watershedsNew(input_, seeds=seeds)
    if size_filter > 0:
        ws, max_id = apply_size_filter(ws, input_, size_filter, exclude=exclude)
    return ws, max_id


def apply_size_filter(
    segmentation: np.ndarray, input_: np.ndarray, size_filter: int, exclude: Optional[List[int]] = None
) -> Tuple[np.ndarray, int]:
    """Apply size filter to a segmentation.

    The segments removed by the size filtering will be filled via a seeded watershed.

    Args:
        segmentation: The input segmentation.
        input_: The innput height map.
        size_filter: The minimal segment size.
        exclude: List of segment ids that will not be filtered.

    Returns:
        The size filtered segmentation.
        The max id of the filtered segmentation.
    """
    ids, sizes = np.unique(segmentation, return_counts=True)
    filter_ids = ids[sizes < size_filter]
    if exclude is not None:
        filter_ids = filter_ids[np.logical_not(np.in1d(filter_ids, exclude))]
    filter_mask = np.in1d(segmentation, filter_ids).reshape(segmentation.shape)
    segmentation[filter_mask] = 0
    _, max_id = vigra.analysis.watershedsNew(input_, seeds=segmentation, out=segmentation)
    return segmentation, max_id


def non_maximum_suppression(dt, seeds):
    """@private
    """
    # Apply non maximum distance suppression to seeds.
    seeds = np.array(np.where(seeds)).transpose()
    seeds = nonMaximumDistanceSuppression(dt, seeds)
    vol = np.zeros(dt.shape, dtype="bool")
    coords = tuple(seeds[:, i] for i in range(seeds.shape[1]))
    vol[coords] = 1
    return vol


def distance_transform_watershed(
    input_: np.ndarray,
    threshold: float,
    sigma_seeds: float,
    sigma_weights: float = 2.0,
    min_size: int = 100,
    alpha: float = 0.9,
    pixel_pitch: Optional[List[int]] = None,
    apply_nonmax_suppression: bool = False,
    mask: Optional[np.ndarray] = None,
    seeds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Compute watershed segmentation based on distance transform seeds.

    Following the procedure outlined in "Multicut brings automated neurite segmentation closer to human performance":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/217205318/beier_17_multicut.pdf

    Args:
        input_: The input height map.
        threshold: The value for the threshold applied before the distance transform.
        sigma_seeds: The smoothing factor for the watershed seed map.
        sigma_weigths: The smoothing factor for the watershed weight map.
        min_size: The minimal size of watershed segments.
        alpha: Value used to blend input_ and distance_transform in order to obtain the watershed weight map.
        pixel_pitch: Anisotropy factor used to compute the distance transform.
        apply_nonmax_suppression: Whetther to apply non-maxmimum suppression to filter out seeds.
        mask: Mask to exclude from segmentation.
        seeds: The initial seeds.

    Returns:
        The watershed segmentation.
        The max id of watershed segmentation.
    """
    if apply_nonmax_suppression and nonMaximumDistanceSuppression is None:
        raise ValueError("Non-maximum suppression is only available with nifty.")

    # check the mask if it was passed
    if mask is not None:
        if mask.shape != input_.shape or mask.dtype != np.dtype("bool"):
            raise ValueError("Invalid mask")

        # return all zeros for empty mask
        if mask.sum() == 0:
            return np.zeros_like(mask, dtype="uint64"), 0

    # threshold the input and compute distance transform
    thresholded = (input_ > threshold).astype("uint32")

    dt = vigra.filters.distanceTransform(thresholded, pixel_pitch=pixel_pitch)

    # shield of the masked area if given
    if mask is not None:
        inv_mask = np.logical_not(mask)
        dt[inv_mask] = 0.

    # compute seeds from maxima of the (smoothed) distance transform
    if sigma_seeds:
        dt = ff.gaussianSmoothing(dt, sigma_seeds)

    # preprpocess initial seeds (if given)
    if seeds is None:
        initial_seeds = None
    else:
        initial_seed_ids = np.unique(seeds)
        seed_max = initial_seed_ids[-1]
        assert len(initial_seed_ids) == seed_max + 1, \
            "The seeds passed to distance_transform_watershed must have consecutive ids and start from 1"
        initial_seeds = seeds

    compute_maxima = vigra.analysis.localMaxima if dt.ndim == 2 else vigra.analysis.localMaxima3D
    seeds = compute_maxima(dt, marker=np.nan, allowAtBorder=True, allowPlateaus=True)
    seeds = np.isnan(seeds)
    if apply_nonmax_suppression:
        seeds = non_maximum_suppression(dt, seeds)
    seeds = vigra.analysis.labelMultiArrayWithBackground(seeds.view("uint8"))

    if initial_seeds is not None:
        seeds[seeds != 0] += seed_max
        initial_seed_mask = initial_seeds != 0
        seeds[initial_seed_mask] = initial_seeds[initial_seed_mask]

    # normalize and invert distance transform
    dt = 1. - (dt - dt.min()) / dt.max()

    # compute weights from input and distance transform
    if sigma_weights:
        hmap = alpha * ff.gaussianSmoothing(input_, sigma_weights) + (1. - alpha) * dt
    else:
        hmap = alpha * input_ + (1. - alpha) * dt

    # compute watershed
    ws, max_id = watershed(hmap, seeds, size_filter=min_size)
    if mask is not None:
        ws[inv_mask] = 0

    return ws.astype("uint64"), max_id


def stacked_watershed(
    input_: np.ndarray,
    ws_function: callable = distance_transform_watershed,
    mask: Optional[np.ndarray] = None,
    n_threads: Optional[int] = None,
    verbose: bool = False,
    output: Optional[ArrayLike] = None,
    **ws_kwargs,
) -> Tuple[ArrayLike, int]:
    """Run 2d watershed stacked along z-axis.

    Args:
        input_: The input height map.
        ws_function: The watershed function.
        mask: Mmask to exclude from segmentation.
        n_threads: The number of threads to use for parallelization.
        verbose: Whether to print progress.
        output: The output for the watershed, will be a newly allocated numpy array by default.
        ws_kwargs: Keyword arguments for the watershed function.

    Returns:
        The watershed segmentation.
        The max id of the watershed segmentation.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if output is None:
        output = np.zeros(input_.shape, dtype="uint64")
    assert output.shape == input_.shape

    if mask is not None and (mask.shape != input_.shape or mask.dtype != np.dtype("bool")):
        raise ValueError("Invalid mask")

    def _wsz(z):
        zmask = None if mask is None else mask[z]
        wsz, max_id = ws_function(input_[z], mask=zmask, **ws_kwargs)
        output[z] = wsz
        return max_id

    nz = len(input_)
    slices = range(nz)
    with futures.ThreadPoolExecutor(n_threads) as tp:
        max_ids = list(tqdm(
            tp.map(_wsz, slices), total=nz, desc="Run stacked watershed"
        )) if verbose else list(tp.map(_wsz, slices))
    offsets = np.array(max_ids, dtype="uint64")

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)

    if mask is None:
        output += offsets[:, None, None]

    else:

        def _add_offset(z):
            output[z][mask[z]] += offsets[z]

        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(_add_offset, z) for z in range(len(input_))]
            [t.result() for t in tasks]

    max_id = int(output[-1].max())
    return output, max_id


def blockwise_two_pass_watershed(
    input_: np.ndarray,
    block_shape: Tuple[int, ...],
    halo: Tuple[int, ...],
    ws_function: callable = distance_transform_watershed,
    n_threads: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    verbose: bool = False,
    output: Optional[ArrayLike] = None,
    **kwargs,
) -> Tuple[ArrayLike, int]:
    """Run a 3d distance transform watershed blockwise, in two passes, to avoid block boundary artifacts.

    Args:
        input_: The input height map.
        block_shape: The block shape for parallelization.
        halo: The halo for extending blocks.
        ws_function: The watershed function.
        n_threads: The number of threads.
        mask: A mask to exclude from segmentation.
        verbose: Whether to print progress.
        output: Output for the watershed, will be a numpy array by default.
        kwargs: Keyword arguments for the watershed function.

    Returns:
        The watershed segmentation.
        The max id of the watershed segmentation.
    """
    assert input_.ndim == 3
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    if output is None:
        output = np.zeros(input_.shape, dtype="uint64")
    assert output.shape == input_.shape

    blocking = nt.blocking([0, 0, 0], list(input_.shape), list(block_shape))
    block_ids_pass_one, block_ids_pass_two = divide_blocks_into_checkerboard(blocking)

    # pass 1: run on the "white" fields of the checkerboard
    # run the watershed independently per block
    def run_block_one(block_id):
        block = blocking.getBlockWithHalo(block_id, list(halo))
        outer_bb = tuple(slice(start, stop) for start, stop in zip(block.outerBlock.begin, block.outerBlock.end))
        input_block = input_[outer_bb]
        mask_block = None if mask is None else mask[outer_bb]
        ws, _ = ws_function(input_block, mask=mask_block, **kwargs)

        inner_bb = tuple(slice(start, stop) for start, stop in zip(block.innerBlock.begin, block.innerBlock.end))
        local_bb = tuple(
            slice(start, stop) for start, stop in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end)
        )
        ws = vigra.analysis.labelMultiArrayWithBackground(ws[local_bb].astype("uint32")).astype("uint64")

        # use the lowest pixel id in this block as offset
        # in order to guarantee that superpixel ids are unique
        offset = block_id * np.prod(blocking.blockShape)
        if mask_block is None:
            ws += offset
        else:
            ws[mask_block[local_bb]] += offset
        output[inner_bb] = ws

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(run_block_one, block_ids_pass_one), total=len(block_ids_pass_one),
            desc="Run pass one of two-pass watershed"
        )) if verbose else list(tp.map(run_block_one, block_ids_pass_one))

    # pass 2: run on the "black" fields of the checkerboard
    # seed the watshed from the segments from "white"
    def run_block_two(block_id):
        block = blocking.getBlockWithHalo(block_id, list(halo))
        outer_bb = tuple(slice(start, stop) for start, stop in zip(block.outerBlock.begin, block.outerBlock.end))
        input_block = input_[outer_bb]
        mask_block = None if mask is None else mask[outer_bb]
        seeds_block = output[outer_bb]

        # relabel the seeds to be consecutive and start from 1
        seeds_block, seed_max, seed_id_mapping = vigra.analysis.relabelConsecutive(
            seeds_block, start_label=1, keep_zeros=True
        )

        ws, ws_max_id = ws_function(input_block, mask=mask_block, seeds=seeds_block, **kwargs)

        inner_bb = tuple(slice(start, stop) for start, stop in zip(block.innerBlock.begin, block.innerBlock.end))
        local_bb = tuple(
            slice(start, stop) for start, stop in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end)
        )
        ws = ws[local_bb]

        offset = block_id * np.prod(blocking.blockShape)
        # map ids corresponding to seeds back to the seed ids
        id_mapping = {v: k for k, v in seed_id_mapping.items()}
        assert 0 in id_mapping

        # map the other seeds to their value + the block offset
        id_mapping.update({seed_id: seed_id + offset for seed_id in range(seed_max + 1, ws_max_id + 1)})
        # apply the mapping
        ws = nt.takeDict(id_mapping, ws)

        output[inner_bb] = ws

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(run_block_two, block_ids_pass_two), total=len(block_ids_pass_two),
            desc="Run pass two of two-pass watershed"
        )) if verbose else list(tp.map(run_block_two, block_ids_pass_two))

    _, max_id, _ = vigra.analysis.relabelConsecutive(output, out=output)
    return output, max_id


def from_affinities_to_boundary_prob_map(
    affinities: np.ndarray,
    offsets: List[List[int]],
    used_offsets: Optional[List[int]] = None,
    offset_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Compute a boundary-probability map from merge affinities (1.0 indicates merge, 0.0 indicates split).

    For every pixel, the value of the probability map is given by the minimum affinity associated to that pixel.
    Only the offsets specified in `used_offsets` will be considered while taking the minimum
    (usually, best results are achieved when using only affinities associated to local or short-range offsets).

    It is also possible to weight different offsets differently when taking the minimum (`offset_weights`).
    By default, all affinities are used and weighted with equal weight = 1.0.

    Args:
        affinities: The affinity map.
        offsets: The offsets corresponding to the affinity channel.
        used_offsets: The offsets to use for accumulating boundary probabilities.
        offset_weights: The weights for probability accumulation.

    Returns:
        The boundary probability map.
    """
    if isinstance(offsets, list):
        offsets = np.array(offsets)

    inverted_affs = 1. - affinities
    if used_offsets is None:
        used_offsets = range(offsets.shape[0])
    if offset_weights is None:
        offset_weights = [1.0 for _ in range(len(used_offsets))]
    assert len(used_offsets) == len(offset_weights)
    rolled_affs = []
    for i, offs_idx in enumerate(used_offsets):
        offset = offsets[offs_idx]
        shifts = tuple([int(off / 2) for off in offset])

        padding = [[0, 0] for _ in range(len(shifts))]
        for ax, shf in enumerate(shifts):
            if shf < 0:
                padding[ax][1] = -shf
            elif shf > 0:
                padding[ax][0] = shf
        padded_inverted_affs = np.pad(inverted_affs, pad_width=((0, 0),) + tuple(padding), mode='constant')
        crop_slices = tuple(
            slice(padding[ax][0], padded_inverted_affs.shape[ax + 1] - padding[ax][1]) for ax in range(3))
        rolled_affs.append(
            np.roll(padded_inverted_affs[offs_idx], shifts, axis=(0, 1, 2))[crop_slices] * offset_weights[i])
    prob_map = np.stack(rolled_affs).max(axis=0)

    return prob_map


class WatershedOnDistanceTransformFromAffinities:
    """A wrapper around the `distance_transform_watershed` function, so that it can be used as superpixel generator.

    As input to the call function, it expects affinities with shape (nb_offsets, shape_x, shape_y, shape_z).
    See also `from_affinities_to_boundary_prob_map` for details on how the arguments are used.

    Args:
        offsets: The offsets corresponding to the affinity channel.
        used_offsets: The offsets to use for accumulating boundary probabilities.
        offset_weights: The weights for probability accumulation.
        return_hmap: Whether to return the computed boundary probability map together with the segmentation.
        invert_affinities: Whether the passed affinities should be inverted with (1. - given_affinities).
        stacked_2d: Whether to compute the WS segmentation for 2D slices
        watershed_kwargs: Arguments passed to the `distance_transform_watershed`.
    """
    def __init__(
        self,
        offsets: List[List[int]],
        used_offsets: Optional[List[int]] = None,
        offset_weights: Optional[List[float]] = None,
        return_hmap: bool = False,
        invert_affinities: bool = False,
        stacked_2d: bool = False,
        n_threads: Optional[int] = None,
        **watershed_kwargs
    ):
        if isinstance(offsets, list):
            offsets = np.array(offsets)
        else:
            assert isinstance(offsets, np.ndarray)

        self.offsets = offsets
        self.used_offsets = used_offsets
        self.offset_weights = offset_weights
        self.return_hmap = return_hmap
        self.invert_affinities = invert_affinities
        self.stacked_2d = stacked_2d
        self.watershed_kwargs = watershed_kwargs
        self.nb_threads = n_threads

    def __call__(self, affinities, foreground_mask=None):
        assert affinities.shape[0] == len(self.offsets)
        assert affinities.ndim == 4

        if self.invert_affinities:
            affinities = 1. - affinities
        hmap = from_affinities_to_boundary_prob_map(affinities, self.offsets, self.used_offsets, self.offset_weights)

        background_mask = None if foreground_mask is None else np.logical_not(foreground_mask)
        if self.stacked_2d:
            segmentation, _ = stacked_watershed(
                hmap, ws_function=distance_transform_watershed,
                mask=background_mask, n_threads=self.nb_threads, **self.watershed_kwargs
            )
        else:
            segmentation, _ = distance_transform_watershed(hmap, mask=background_mask, **self.watershed_kwargs)

        # Map ignored pixels to -1:
        if foreground_mask is not None:
            assert foreground_mask.shape == segmentation.shape
            segmentation = segmentation.astype("int64")
            segmentation = np.where(foreground_mask, segmentation, np.ones_like(segmentation) * (-1))

        if self.return_hmap:
            return segmentation, hmap
        else:
            return segmentation
