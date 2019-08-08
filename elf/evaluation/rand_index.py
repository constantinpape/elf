from .util import contigency_table, compute_ignore_mask


def compute_rand_scores(a_dict, b_dict, p_counts, n_points):

    # compute the rand-primitves
    a_counts = a_dict.values()
    sum_a = float(sum(c * c for c in a_counts))

    b_counts = b_dict.values()
    sum_b = float(sum(c * c for c in b_counts))

    sum_ab = float(sum(c * c for c in p_counts))

    prec = sum_ab / sum_b
    rec = sum_ab / sum_a

    # compute rand scores:
    # adapted rand index and randindex
    ari = (2 * prec * rec) / (prec + rec)
    ri = 1. - (sum_a + sum_b - 2 * sum_ab) / (n_points * n_points)
    ari = 1. - ari

    return ari, ri


def rand_index(segmentation, groundtruth, ignore_seg=None, ignore_gt=None):
    """ Compute rand index derived scores between two segmentations.

    Computes adapted rand error and rand index.

    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth
        ignore_seg [listlike] - ignore ids for segmentation (default: None)
        ignore_gt [listlike] - ignore ids for groundtruth (default: None)
    Retuns:
        float - adapted rand error
        float - rand index
    """
    ignore_mask = compute_ignore_mask(segmentation, groundtruth,
                                      ignore_seg, ignore_gt)
    if ignore_mask is not None:
        segmentation = segmentation[ignore_mask]
        groundtruth = groundtruth[ignore_mask]
    else:
        # if we don't have a mask, we need to make sure the segmentations are
        segmentation = segmentation.ravel()
        groundtruth = groundtruth.ravel()

    # compute ids, counts and overlaps making up the contigency table
    a_dict, b_dict, _, p_counts = contigency_table(groundtruth, segmentation)
    n_points = segmentation.size

    # compute and return rand scores
    ari, ri = compute_rand_scores(a_dict, b_dict, p_counts, n_points)
    return ari, ri
