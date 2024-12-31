from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from .util import contigency_table


def intersection_over_union(overlap):
    """@private
    """
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    eps = 1e-7
    return overlap / np.maximum(n_pixels_pred + n_pixels_true - overlap, eps)


def intersection_over_true(overlap):
    """@private
    """
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return overlap / n_pixels_true


def intersection_over_pred(overlap):
    """@private
    """
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    return overlap / n_pixels_pred


MATCHING_CRITERIA = {"iou": intersection_over_union,
                     "iot": intersection_over_true,
                     "iop": intersection_over_pred}
"""@private
"""


def precision(tp, fp, fn):
    """@private
    """
    return tp/(tp+fp) if tp > 0 else 0


def recall(tp, fp, fn):
    """@private
    """
    return tp/(tp+fn) if tp > 0 else 0


def segmentation_accuracy(tp, fp, fn):
    """@private
    """
    # -> https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    return tp/(tp+fp+fn) if tp > 0 else 0


def f1(tp, fp, fn):
    """@private
    """
    return (2*tp)/(2*tp+fp+fn) if tp > 0 else 0


def label_overlap(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    ignore_label: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the number of overlapping elements for objects in two label images.

    Args:
        seg_a: Candidate segmentation to evaluate.
        seg_b: Segmentation to compare to.
        ignore_label: Overlap of any objects with this label are not
            taken into account in the output. `None` indicates that no label
            should be ignored. It is assumed that the `ignore_label` has the
            same meaning in both segmentations.

    Returns:
        Matrix with cells i,j containing the count of overlapping elements
            of object i in `seg_a` with obj j in `seg_b`.
            Note: indices in the returned matrix may not correspond to object ids anymore.
        Index of ignore label in label_overlap output matrix.
    """
    p_ids, p_counts = contigency_table(seg_a, seg_b)[2:]
    p_ids = p_ids.astype("uint64")

    # unique object_ids in a, b
    u_oids_a, u_oids_b = np.unique(p_ids[:, 0]), np.unique(p_ids[:, 1])
    n_objs_a, n_objs_b = len(u_oids_a), len(u_oids_b)

    # mapping from unique indexes to continuous 0...N
    idx_map_a = dict(zip(u_oids_a, range(n_objs_a)))
    idx_map_b = dict(zip(u_oids_b, range(n_objs_b)))

    # matrix with max_remapped index in each dimension
    overlap = np.zeros((n_objs_a, n_objs_b), dtype="uint64")

    # remap indices and set costs in matrix
    index = ([idx_map_a[idx] for idx in p_ids[:, 0]], [idx_map_b[idx] for idx in p_ids[:, 1]])
    overlap[index] = p_counts

    # determine remapped row, column indices of ignore_label
    ignore_idx = (None, None)
    if ignore_label is not None:
        ignore_idx = (idx_map_a.get(ignore_label), idx_map_b.get(ignore_label))

    return overlap, ignore_idx


def _compute_scores(segmentation, groundtruth, criterion, ignore_label):
    # compute overlap from the contingency table
    overlap, ignore_idx = label_overlap(segmentation, groundtruth, ignore_label)

    # compute scores with the matcher
    matcher = MATCHING_CRITERIA[criterion]
    scores = matcher(overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1, f"{np.min(scores)}, {np.max(scores)}"

    # remove ignore_label (remapped to continuous object_ids)
    if ignore_idx[0] is not None:
        scores = np.delete(scores, ignore_idx[0], axis=0)
    if ignore_idx[1] is not None:
        scores = np.delete(scores, ignore_idx[1], axis=1)

    n_pred, n_true = scores.shape
    n_matched = min(n_true, n_pred)

    return n_true, n_matched, n_pred, scores


def _compute_tps(scores, n_matched, threshold):
    not_trivial = n_matched > 0 and np.any(scores >= threshold)
    if not_trivial:
        # compute optimal matching with scores as tie-breaker
        costs = -(scores >= threshold).astype(float) - scores / (2*n_matched)
        pred_ind, true_ind = linear_sum_assignment(costs)
        assert n_matched == len(true_ind) == len(pred_ind)
        match_ok = scores[pred_ind, true_ind] >= threshold
        tp = np.count_nonzero(match_ok)
    else:
        tp = 0
    return tp


def matching(
    segmentation: np.ndarray,
    groundtruth: np.ndarray,
    threshold: float = 0.5,
    criterion: str = "iou",
    ignore_label: int = 0,
) -> Dict[str, float]:
    """Compute scores from matching objects in segmentation and groundtruth.

    Implementation based on:
    https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py

    Args:
        segmentation: Candidate segmentation to evaluate.
        groundtruth: Groundtruth segmentation.
        threshold: Overlap threshold.
        criterion: Matching criterion. Can be one of "iou", "iop", "iot".
        ignore_label: Overlap of any objects with this label are not
            taken into account in the output. `None` indicates that no label
            should be ignored. It is assumed that the `ignore_label` has the
            same meaning in both segmentations.

    Returns:
        Mapping of the names for different metrics to their respective scores.
    """

    n_true, n_matched, n_pred, scores = _compute_scores(segmentation, groundtruth, criterion, ignore_label)
    tp = _compute_tps(scores, n_matched, threshold)
    fp = n_pred - tp
    fn = n_true - tp
    stats = {"precision": precision(tp, fp, fn),
             "recall": recall(tp, fp, fn),
             "segmentation_accuracy": segmentation_accuracy(tp, fp, fn),
             "f1": f1(tp, fp, fn)}
    return stats


def mean_segmentation_accuracy(
    segmentation: np.ndarray,
    groundtruth: np.ndarray,
    thresholds: Optional[List[float]] = None,
    return_accuracies: bool = False,
    ignore_label: int = 0,
) -> Union[float, Tuple[float, List[float]]]:
    """Compute the mean segmentation accuracy metrics for comparing two segmentation results.

    This metric was introduced in the PascalVoc Challenge:
    https://link.springer.com/article/10.1007/s11263-009-0275-4
    The implementation used here follows the DSB 2018 Nucelus Segmentation Challenge.

    Args:
        segmentation: Candidate segmentation to evaluate.
        groundtruth: Groundtruth segmentation.
        thresholds: Overlap thresholds, by default np.arange(0.5, 1., 0.05) is used.
        return_accuracies: Whether to return intermediate scores.
        ignore_label: Overlap of any objects with this label are not
            taken into account in the output. `None` indicates that no label
            should be ignored. It is assumed that the `ignore_label` has the
            same meaning in both segmentations.

    Returns:
        The mean segmentation accuracy score.
        The segmentation accuracies for the individual overlap thresholds.
            Only returned if return_accuracies is set to True.
    """
    n_true, n_matched, n_pred, scores = _compute_scores(
        segmentation, groundtruth, criterion="iou", ignore_label=ignore_label
    )
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)

    tps = [_compute_tps(scores, n_matched, threshold) for threshold in thresholds]
    fps = [n_pred - tp for tp in tps]
    fns = [n_true - tp for tp in tps]
    accuracies = [segmentation_accuracy(tp, fp, fn) for tp, fp, fn in zip(tps, fps, fns)]
    mean_accuracy = np.mean(accuracies)

    if return_accuracies:
        return mean_accuracy, accuracies
    else:
        return mean_accuracy
