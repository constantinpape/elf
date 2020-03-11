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

    if input_.dtype != 'uint8' or seeds.dtype != 'uint32':
        # Vigra watershedsNew only supports input_ of type uint8 and seeds of type uint32.
        raise Exception("Expected input_, seeds of numpy array dtype uint8, uint32, but got {}, {} instead."
                        .format(input_.dtype, seeds.dtype))

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
    vol = np.zeros(dt.shape, dtype='bool')
    coords = tuple(seeds[:, i] for i in range(seeds.shape[1]))
    vol[coords] = 1
    return vol


def distance_transform_watershed(input_, threshold, sigma_seeds,
                                 sigma_weights=2., min_size=100,
                                 alpha=.9, pixel_pitch=None,
                                 apply_nonmax_suppression=False):
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

    Returns:
        np.ndarray - watershed segmentation
        int - max id of watershed segmentation
    """
    if apply_nonmax_suppression and nonMaximumDistanceSuppression is None:
        raise ValueError("Non-maximum suppression is only available with nifty.")

    # threshold the input and compute distance transform
    thresholded = (input_ > threshold).astype('uint32')
    dt = vigra.filters.distanceTransform(thresholded, pixel_pitch=pixel_pitch)

    # compute seeds from maxima of the (smoothed) distance transform
    if sigma_seeds:
        dt = ff.gaussianSmoothing(dt, sigma_seeds)
    compute_maxima = vigra.analysis.localMaxima if dt.ndim == 2 else vigra.analysis.localMaxima3D
    seeds = compute_maxima(dt, marker=np.nan, allowAtBorder=True, allowPlateaus=True)
    seeds = np.isnan(seeds)
    if apply_nonmax_suppression:
        seeds = non_maximum_suppression(dt, seeds)
    seeds = vigra.analysis.labelMultiArrayWithBackground(seeds.view('uint8'))

    # normalize and invert distance transform
    dt = 1. - (dt - dt.min()) / dt.max()

    # compute weights from input and distance transform
    if sigma_weights:
        hmap = alpha * ff.gaussianSmoothing(input_, sigma_weights) + (1. - alpha) * dt
    else:
        hmap = alpha * input_ + (1. - alpha) * dt

    hmap = hmap.astype('uint8') # change back to uint8, so as to work with Vigra watershedsNew

    # compute watershed
    ws, max_id = watershed(hmap, seeds, size_filter=min_size)
    return ws, max_id


def stacked_watershed(input_, ws_function=distance_transform_watershed,
                      n_threads=None, **ws_kwargs):
    """ Run 2d watershed stacked along z-axis.

    Arguments:
        input_ [np.ndarray] - input height map.
        ws_function [callable] - watershed function (default: distance_transform_watershed)
        n_threads [int] - number of threads (default: None)
        ws_kwargs - keyworrd arguments for the watershed function

    Returns:
        np.ndarray - watershed segmentation
        int - max id of watershed segmentation
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    out = np.zeros(input_.shape, dtype='uint64')

    def _wsz(z):
        wsz, max_id = ws_function(input_[z], **ws_kwargs)
        out[z] = wsz
        return max_id

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_wsz, z) for z in range(len(input_))]
        offsets = np.array([t.result() for t in tasks], dtype='uint64')

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)

    out += offsets[:, None, None]
    max_id = int(out[-1].max())
    return out, max_id
