from typing import Optional, Sequence, Tuple

import numpy as np

from .util import compute_ignore_mask, contigency_table
from .rand_index import compute_rand_scores
from .variation_of_information import compute_vi_scores


def cremi_score(
    segmentation: np.ndarray,
    groundtruth: np.ndarray,
    ignore_seg: Optional[Sequence[int]] = None,
    ignore_gt: Optional[Sequence[int]] = None,
) -> Tuple[float, float, float, float]:
    """Compute cremi score of two segmentations.

    This score was used as the evaluation metric for the CREMI challenge.
    It is defined as the geometric mean of the variation of information and the adapted rand score.

    Args:
        segmentation: Candidate segmentation to evaluate.
        groundtruth: Groundtruth segmentation.
        ignore_seg: Ignore ids for the segmentation.
        ignore_gt: Ignore ids for the groundtruth.

    Retuns:
        The variation of information split score.
        The variation of information merge score.
        The adapted rand error.
        The cremi score.
    """

    ignore_mask = compute_ignore_mask(segmentation, groundtruth, ignore_seg, ignore_gt)
    if ignore_mask is not None:
        segmentation = segmentation[ignore_mask]
        groundtruth = groundtruth[ignore_mask]
    else:
        segmentation = segmentation.ravel()
        groundtruth = groundtruth.ravel()

    # Compute ids, counts and overlaps making up the contigency table.
    a_dict, b_dict, p_ids, p_counts = contigency_table(groundtruth, segmentation)
    n_points = segmentation.size

    # Compute VI scores.
    vis, vim = compute_vi_scores(a_dict, b_dict, p_ids, p_counts, n_points, use_log2=True)

    # Compute rand score.
    ari, _ = compute_rand_scores(a_dict, b_dict, p_counts, n_points)

    # Compute the cremi score = geometric mean of voi and ari.
    cs = np.sqrt(ari * (vis + vim))

    return vis, vim, ari, cs
