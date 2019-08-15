import numpy as np
import vigra
try:
    from nifty.filters import nonMaximumDistanceSuppression
except ImportError:
    nonMaximumDistanceSuppression = None


def watershed(input_, seeds, size_filter=0, exclude=None):
    """ Compute seeded watershed.

    Parameter:
        input_ [np.ndarray] - input height map.
        seeds [np.ndarray] - seed map.
        size_filter [int] - minimal segment size (default: 0).
        exclude [list] - list of segment ids that will not be size filtered (default: None).
    """
    ws, max_id = vigra.analysis.watershedsNew(input_, seeds=seeds)
    if size_filter > 0:
        ws, max_id = apply_size_filter(ws, input_, size_filter,
                                       exclude=exclude)
    return ws, max_id


def apply_size_filter(segmentation, input_, size_filter, exclude=None):
    """ Apply size filter to segmentation.

    Parameter:
        segmentation [np.ndarray] - input segmentation.
        input_ [np.ndarray] - input height map.
        size_filter [int] - minimal segment size.
        exclude [list] - list of segment ids that will not be size filtered (default: None).
    """
    ids, sizes = np.unique(segmentation, return_counts=True)
    filter_ids = ids[sizes < size_filter]
    if exclude is not None:
        filter_ids = filter_ids[np.logical_not(np.in1d(filter_ids, exclude))]
    filter_mask = np.in1d(segmentation, filter_ids).reshape(segmentation.shape)
    segmentation[filter_mask] = 0
    _, max_id = vigra.analysis.watershedsNew(input_, seeds=segmentation, out=segmentation)
    return segmentation, max_id


# TODO implement
def distance_transform_watershed(input_, threshold, sigma_seeds,
                                 sigma_weights=2., min_size=100,
                                 alpha=.9, pixel_pitch=None,
                                 apply_non_max_suppression=False):
    """ Compute watershed segmentation based on distance transform seeds.

    Parameter:
        input_ [np.ndarray] - input height map.
        threshold [float] - value for the threshold applied before distance tranform.
        sigma_seeds [float] - smoothing factor for the watershed seed map.
        sigma_weigths [float] - smoothing factor for the watershed weight map (default: 2).
        min_size [int] - minimal size of watershed segments (default: 100)
        alpha [float] - alpha used to blend input_ and distance_transform in order to obtain the
            watershed weight map (default: .9)
        pixel_pitch [listlike[int]] - anisotropy factor used to compute the distance transform (default: None)
        apply_non_max_suppression [bool] - whetther to apply non-maxmimum suppression to filter out seeds.
            Needs nifty. (default: False)
    """
    pass
