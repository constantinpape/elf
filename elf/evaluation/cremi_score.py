import numpy as np
from .util import compute_ignore_mask, contigency_table
from .rand_index import compute_rand_scores
from .variation_of_information import compute_vi_scores


def cremi_score(segmentation, groundtruth, ignore_seg=None, ignore_gt=None,
                ignore_gt_split=None, ignore_gt_merge=None):
    """ Computes cremi scores between two segmentations

    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth
        ignore_seg [listlike] - ignore ids for segmentation (default: None)
        ignore_gt [listlike] - ignore ids for groundtruth (default: None)
        ignore_gt_split [listlike] - ignore groundtruth ids for split contribution (default: None)
        ignore_gt_merge [listlike] - ignore groundtruth ids for merge contribution (default: None)
    Retuns:
        float - vi-split
        float - vi-merge
        float - adapted rand error
        float - cremi score
    """

    ignore_mask = compute_ignore_mask(segmentation, groundtruth,
                                      ignore_seg, ignore_gt)
    if ignore_mask is not None:
        segmentation = segmentation[ignore_mask]
        groundtruth = groundtruth[ignore_mask]
    else:
        # if we don't have a mask, we need to make sure the segmentations are 1d
        segmentation = segmentation.ravel()
        groundtruth = groundtruth.ravel()

    # compute ids, counts and overlaps making up the contigency table
    a_dict, b_dict, p_ids, p_counts = contigency_table(groundtruth, segmentation,
                                                       ignore_gt_split, ignore_gt_merge)
    n_points = segmentation.size

    # compute vi scores
    vis, vim = compute_vi_scores(a_dict, b_dict, p_ids, p_counts, n_points,
                                 use_log2=True)

    # compute and rand scores
    ari, _ = compute_rand_scores(a_dict, b_dict, p_counts, n_points)

    # compute the cremi score = geometric mean of voi and ari
    cs = np.sqrt(ari * (vis + vim))

    return vis, vim, ari, cs
