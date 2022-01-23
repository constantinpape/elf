import multiprocessing
from concurrent import futures
import numpy as np
import vigra
try:
    from nifty.filters import nonMaximumDistanceSuppression
except ImportError:
    nonMaximumDistanceSuppression = None
try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff


def watershed(input_, seeds, size_filter=0, exclude=None):
    """ Compute seeded watershed.

    Arguments:
        input_ [np.ndarray] - input height map.
        seeds [np.ndarray] - seed map.
        size_filter [int] - minimal segment size (default: 0).
        exclude [list] - list of segment ids that will not be size filtered (default: None).

    Returns:
        np.ndarray - watershed segmentation
        int - max id of watershed segmentation
    """
    ws, max_id = vigra.analysis.watershedsNew(input_, seeds=seeds)
    if size_filter > 0:
        ws, max_id = apply_size_filter(ws, input_, size_filter,
                                       exclude=exclude)
    return ws, max_id


def apply_size_filter(segmentation, input_, size_filter, exclude=None):
    """ Apply size filter to segmentation.

    Arguments:
        segmentation [np.ndarray] - input segmentation.
        input_ [np.ndarray] - input height map.
        size_filter [int] - minimal segment size.
        exclude [list] - list of segment ids that will not be size filtered (default: None).

    Returns:
        np.ndarray - size filtered segmentation
        int - max id of size filtered segmentation
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
    """ Apply non maximum distance suppression to seeds.
    """
    seeds = np.array(np.where(seeds)).transpose()
    seeds = nonMaximumDistanceSuppression(dt, seeds)
    vol = np.zeros(dt.shape, dtype="bool")
    coords = tuple(seeds[:, i] for i in range(seeds.shape[1]))
    vol[coords] = 1
    return vol


def distance_transform_watershed(input_, threshold, sigma_seeds,
                                 sigma_weights=2., min_size=100,
                                 alpha=.9, pixel_pitch=None,
                                 apply_nonmax_suppression=False,
                                 mask=None):
    """ Compute watershed segmentation based on distance transform seeds.

    Following the procedure outlined in "Multicut brings automated neurite segmentation closer to human performance":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/217205318/beier_17_multicut.pdf

    Arguments:
        input_ [np.ndarray] - input height map.
        threshold [float] - value for the threshold applied before distance tranform.
        sigma_seeds [float] - smoothing factor for the watershed seed map.
        sigma_weigths [float] - smoothing factor for the watershed weight map (default: 2).
        min_size [int] - minimal size of watershed segments (default: 100)
        alpha [float] - alpha used to blend input_ and distance_transform in order to obtain the
            watershed weight map (default: .9)
        pixel_pitch [listlike[int]] - anisotropy factor used to compute the distance transform (default: None)
        apply_nonmax_suppression [bool] - whetther to apply non-maxmimum suppression to filter out seeds.
            Needs nifty. (default: False)
        mask [np.ndarray] - mask to exclude from segmentation (default: None)

    Returns:
        np.ndarray - watershed segmentation
        int - max id of watershed segmentation
    """
    if apply_nonmax_suppression and nonMaximumDistanceSuppression is None:
        raise ValueError("Non-maximum suppression is only available with nifty.")

    # check the mask if it was passed
    if mask is not None:
        if mask.shape != input_.shape or mask.dtype != np.dtype("bool"):
            raise ValueError("Invalid mask")

        # return all zeros for empty mask
        if mask.sum() == 0:
            return np.zeros_like(mask, dtype="uint32"), 0

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

    compute_maxima = vigra.analysis.localMaxima if dt.ndim == 2 else vigra.analysis.localMaxima3D
    seeds = compute_maxima(dt, marker=np.nan, allowAtBorder=True, allowPlateaus=True)
    seeds = np.isnan(seeds)
    if apply_nonmax_suppression:
        seeds = non_maximum_suppression(dt, seeds)
    seeds = vigra.analysis.labelMultiArrayWithBackground(seeds.view("uint8"))

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

    return ws, max_id


def stacked_watershed(input_, ws_function=distance_transform_watershed,
                      mask=None, n_threads=None, **ws_kwargs):
    """ Run 2d watershed stacked along z-axis.

    Arguments:
        input_ [np.ndarray] - input height map.
        ws_function [callable] - watershed function (default: distance_transform_watershed)
        mask [np.ndarray] - mask to exclude from segmentation (default: None)
        n_threads [int] - number of threads (default: None)
        ws_kwargs - keyworrd arguments for the watershed function

    Returns:
        np.ndarray - watershed segmentation
        int - max id of watershed segmentation
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    out = np.zeros(input_.shape, dtype="uint64")

    if mask is not None and (mask.shape != input_.shape or mask.dtype != np.dtype("bool")):
        raise ValueError("Invalid mask")

    def _wsz(z):
        zmask = None if mask is None else mask[z]
        wsz, max_id = ws_function(input_[z], mask=zmask, **ws_kwargs)
        out[z] = wsz
        return max_id

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_wsz, z) for z in range(len(input_))]
        offsets = np.array([t.result() for t in tasks], dtype="uint64")

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)

    if mask is None:
        out += offsets[:, None, None]

    else:

        def _add_offset(z):
            out[z][mask[z]] += offsets[z]

        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(_add_offset, z) for z in range(len(input_))]
            [t.result() for t in tasks]

    max_id = int(out[-1].max())
    return out, max_id


def from_affinities_to_boundary_prob_map(affinities, offsets, used_offsets=None, offset_weights=None):
    """
    Compute a boundary-probability map from merge affinities (1.0 indicates merge, 0.0 indicates split).
    For every pixel, the value of the probability map is given by the minimum affinity associated to that pixel.

    Only the offsets specified in `used_offsets` will be considered while taking the minimum (usually, best results are
    achieved when using only affinities associated to local or short-range offsets).

    It is also possible to weight different offsets differently when taking the minimum (`offset_weights`). By default,
    all affinities are used and weighted with equal weight = 1.0.

    :param affinities: Expected shape of the array: (channels,z,x,y) or (channels,x,y)
    :param offsets: np.array indicating the offsets associated to the passed affinities
    :param used_offsets: list of offset indices that will be actually used to compute the prob. map
    :param offset_weights: list of weights for the used offsets (by default all weights are 1.0)
    :return: Boundary-probability map
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


class WatershedOnDistanceTransformFromAffinities():
    def __init__(self, offsets,
                 used_offsets=None,
                 offset_weights=None,
                 return_hmap=False,
                 invert_affinities=False,
                 stacked_2d=False,
                 nb_threads=None,
                 **watershed_kwargs):
        """
        A wrapper around the `distance_transform_watershed` function, so that it can be used as superpixel generator for
        GASP or other segmentation workflows.

        As input, it expects affinities with shape (nb_offsets, shape_x, shape_y, shape_z).


        :param offsets:
        :param used_offsets: See arguments of `from_affinities_to_boundary_prob_map` function
        :param offset_weights: See arguments of `from_affinities_to_boundary_prob_map` function
        :param return_hmap: Whether to return the computed boundary probability map together with the resulting segmentation.
        :param invert_affinities: Whether the passed affinities should be inverted with (1. - given_affinities). False by default.
        :param stacked_2d: Whether to compute the WS segmentation for 2D slices
        :param watershed_kwargs: Arguments passed to the `distance_transform_watershed`
        """
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
        self.nb_threads = nb_threads

    def __call__(self, affinities, foreground_mask=None):
        assert affinities.shape[0] == len(self.offsets)
        assert affinities.ndim == 4

        if self.invert_affinities:
            affinities = 1. - affinities

        hmap = from_affinities_to_boundary_prob_map(affinities, self.offsets, self.used_offsets,
                                       self.offset_weights)

        background_mask = None if foreground_mask is None else np.logical_not(foreground_mask)
        if self.stacked_2d:
            segmentation, _ = stacked_watershed(hmap, ws_function=distance_transform_watershed,
                      mask=background_mask, n_threads=self.nb_threads, **self.watershed_kwargs)
        else:
            segmentation, _ = distance_transform_watershed(hmap, mask=background_mask,
                                                                           **self.watershed_kwargs)

        # Map ignored pixels to -1:
        if foreground_mask is not None:
            assert foreground_mask.shape == segmentation.shape
            segmentation = segmentation.astype('int64')
            segmentation = np.where(foreground_mask, segmentation, np.ones_like(segmentation) * (-1))

        if self.return_hmap:
            return segmentation, hmap
        else:
            return segmentation
