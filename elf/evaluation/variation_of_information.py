from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from .util import contigency_table, compute_ignore_mask


def compute_vi_scores(a_dict, b_dict, p_ids, p_counts, n_points, use_log2):
    """@private
    """
    log = np.log2 if use_log2 else np.log

    # Compute the vi-primitves.
    a_counts = a_dict.values()
    sum_a = sum(-c / n_points * log(c / n_points) for c in a_counts)

    b_counts = b_dict.values()
    sum_b = sum(-c / n_points * log(c / n_points) for c in b_counts)

    sum_ab = np.sum([c / n_points * log(n_points * c / (a_dict[a] * b_dict[b]))
                     for (a, b), c in zip(p_ids, p_counts)])

    # Compute the actual vi-scores (split-vi, merge-vi).
    vis = sum_b - sum_ab
    vim = sum_a - sum_ab
    return vis, vim


def variation_of_information(
    segmentation: np.ndarray,
    groundtruth: np.ndarray,
    ignore_seg: Optional[Sequence[int]] = None,
    ignore_gt: Optional[Sequence[int]] = None,
    use_log2: bool = True
) -> Tuple[float, float]:
    """Compute variation of information between two segmentations.

    This function computes the split and merge variation of information scores
    You can add them up to get the overall variation of information.

    Args:
        segmentation: Candidate segmentation to evaluate.
        groundtruth: Groundtruth segmentation.
        ignore_seg: Ignore ids for the segmentation.
        ignore_gt: Ignore ids for the groundtruth.
        use_log2: Whether to use log_2 or log_e.

    Retuns:
        The split variation of information.
        The merge variation of information.
    """
    ignore_mask = compute_ignore_mask(segmentation, groundtruth,
                                      ignore_seg, ignore_gt)
    if ignore_mask is not None:
        segmentation = segmentation[ignore_mask]
        groundtruth = groundtruth[ignore_mask]
    else:
        segmentation = segmentation.ravel()
        groundtruth = groundtruth.ravel()

    # Compute ids, counts and overlaps making up the contigency table.
    a_dict, b_dict, p_ids, p_counts = contigency_table(groundtruth, segmentation)
    n_points = segmentation.size

    # Compute and return vi scores.
    vis, vim = compute_vi_scores(a_dict, b_dict, p_ids, p_counts, n_points, use_log2=use_log2)
    return vis, vim


def compute_object_vi_scores(a_dict, b_dict, p_ids, p_counts, use_log2):
    """@private
    """
    log = np.log2 if use_log2 else np.log

    object_scores = {}
    for gt_id, gt_count in a_dict.items():

        # Find all objects that overlap with this groundtruth id.
        overlap_mask = p_ids[:, 0] == gt_id
        overlap_ids = p_ids[:, 1][overlap_mask]
        overlap_counts = p_counts[overlap_mask]

        # Compute object scores according to
        # https://arxiv.org/pdf/1708.02599.pdf page 16
        vim = -sum(ocount / gt_count * log(ocount / gt_count) for ocount in overlap_counts)
        vis = -sum(ocount / gt_count * log(ocount / b_dict[ovlp_id])
                   for ocount, ovlp_id in zip(overlap_counts, overlap_ids))
        object_scores[gt_id] = (vis, vim)

    return object_scores


def object_vi(
    segmentation: np.ndarray,
    groundtruth: np.ndarray,
    ignore_seg: Optional[Sequence[int]] = None,
    ignore_gt: Optional[Sequence[int]] = None,
    use_log2: bool = True,
) -> Dict[int, Tuple[float, float]]:
    """Compute the per-object variation of information between two segmentations.

    Based on https://arxiv.org/pdf/1708.02599.pdf.

    Args:
        segmentation: Candidate segmentation to evaluate.
        groundtruth: The groundtruth segmentation.
        ignore_seg: Ignore ids for the segmentation.
        ignore_gt: Ignore ids for the groundtruth.
        use_log2: Whether to use log_2 or log_e.

    Returns:
        Per object split-vi and merge-vi for the groundtruth objects.
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

    object_scores = compute_object_vi_scores(a_dict, b_dict, p_ids, p_counts, use_log2=use_log2)
    return object_scores
