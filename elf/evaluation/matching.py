import numpy as np
from scipy.optimize import linear_sum_assignment
from .util import contigency_table


def intersection_over_union(overlap):
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return overlap / (n_pixels_pred + n_pixels_true - overlap)


def intersection_over_true(overlap):
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return overlap / n_pixels_true


def intersection_over_pred(overlap):
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    return overlap / n_pixels_pred


MATCHING_CRITERIA = {'iou': intersection_over_union,
                     'iot': intersection_over_true,
                     'iop': intersection_over_pred}


def precision(tp, fp, fn):
    return tp/(tp+fp) if tp > 0 else 0


def recall(tp, fp, fn):
    return tp/(tp+fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    # -> https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    return tp/(tp+fp+fn) if tp > 0 else 0


def f1(tp, fp, fn):
    return (2*tp)/(2*tp+fp+fn) if tp > 0 else 0


def label_overlap(seg_a, seg_b):
    p_ids, p_counts = contigency_table(seg_a, seg_b)[2:]
    p_ids = p_ids.astype('uint64')
    max_a, max_b = int(p_ids[:, 0].max()), int(p_ids[:, 1].max())
    overlap = np.zeros((max_a + 1, max_b + 1), dtype='uint64')
    index = (p_ids[:, 0], p_ids[:, 1])
    overlap[index] = p_counts
    return overlap


def _compute_scores(segmentation, groundtruth, criterion):
    # compute overlap from the contingency table
    overlap = label_overlap(segmentation, groundtruth)

    # compute scores with the matcher
    matcher = MATCHING_CRITERIA[criterion]
    scores = matcher(overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    # ignore background
    scores = scores[1:, 1:]
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


def matching(segmentation, groundtruth, threshold=0.5, criterion='iou'):
    """ Scores from matching objects in segmentation and groundtruth.

    Implementation based on:
    https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py

    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth segmentation
        threshold [float] - overlap threshold (default: 0.5)
        criterion [str] - matching criterion. Can be one of 'iou', 'iop', 'iot'. (default: 'iou')
    """

    n_true, n_matched, n_pred, scores = _compute_scores(segmentation, groundtruth, criterion)
    tp = _compute_tps(scores, n_matched, threshold)
    fp = n_pred - tp
    fn = n_true - tp
    stats = {'precision': precision(tp, fp, fn),
             'recall': recall(tp, fp, fn),
             'accuracy': accuracy(tp, fp, fn),
             'f1': f1(tp, fp, fn)}
    return stats


def mean_average_precision(segmentation, groundtruth,
                           thresholds=None, return_aps=False):
    """ Mean average precision metrics.

    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth segmentation
        thresholds [sequence of floats] - overlap thresholds,
            by default np.arange(0.5, 1., 0.05) is used (default: None)
        return_aps [bool] - whether to return intermediate aps (default: false)
    """
    n_true, n_matched, n_pred, scores = _compute_scores(segmentation, groundtruth,
                                                        criterion='iou')
    if thresholds is None:
        thresholds = np.arange(0.5, 1., 0.05)

    tps = [_compute_tps(scores, n_matched, threshold) for threshold in thresholds]
    fps = [n_pred - tp for tp in tps]
    fns = [n_true - tp for tp in tps]
    aps = [precision(tp, fp, fn) for tp, fp, fn in zip(tps, fps, fns)]
    m_ap = np.mean(aps)

    if return_aps:
        return m_ap, aps
    else:
        return m_ap
