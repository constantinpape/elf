import numpy as np
import nifty.ground_truth as ngt

# implementations based on:
# https://github.com/kreshuklab/sparse-object-embeddings/blob/master/pytorch3dunet/clustering/sbd.py


def dice_score(segmentation, groundtruth):
    """ Compute the dice score between binarized segmentation and ground-truth.

    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth

    Returns:
        float - the dice score
    """
    seg = segmentation > 0
    gt = groundtruth > 0

    nom = 2 * np.sum(gt * seg)
    denom = np.sum(gt) + np.sum(seg)

    eps = 1e-7
    score = float(nom) / float(denom + eps)
    return score


def _best_dice_numpy(gt, seg):
    gt_labels = np.setdiff1d(np.unique(gt), [0])
    seg_labels = np.setdiff1d(np.unique(seg), [0])

    if len(seg_labels) == 0 or len(gt_labels) == 0:
        return 0.0

    best_dices = []
    for gt_idx in gt_labels:
        _gt_seg = (gt == gt_idx).astype('uint8')
        dices = []
        for pred_idx in seg_labels:
            _pred_seg = (seg == pred_idx).astype('uint8')

            dice = dice_score(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    return np.mean(best_dices)


def _best_dice_nifty(gt, seg):
    gt_labels, gt_counts = np.unique(gt, return_counts=True)
    seg_labels, seg_counts = np.unique(seg, return_counts=True)
    seg_counts = {seg_id: cnt for seg_id, cnt in zip(seg_labels, seg_counts)}

    if gt_labels[0] == 0:
        gt_labels, gt_counts = gt_labels[1:], gt_counts[1:]
    if len(gt_labels) == 0:
        return 0.0

    eps = 1e-7
    overlaps = ngt.overlap(gt, seg)
    dice_scores = []
    for gt_id, gt_count in zip(gt_labels, gt_counts):
        ovlp_ids, ovlp_counts = overlaps.overlapArrays(gt_id, sorted=True)
        if ovlp_ids[0] == 0:
            ovlp_ids, ovlp_counts = ovlp_ids[1:], ovlp_counts[1:]
        if len(ovlp_ids) == 0:
            dice_scores.append(0.)
            continue
        seg_id, count = ovlp_ids[0], ovlp_counts[0]
        score = float(2 * count) / float(gt_count + seg_counts[seg_id] + eps)
        dice_scores.append(score)

    return np.mean(dice_scores)


def symmetric_best_dice_score(segmentation, groundtruth, impl='nifty'):
    """ Compute the best symmetric dice score between the objects in the groundtruth and segmentation.

    This metric is used in the CVPPP instance segmentation challenge.

    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth
        impl [str] - implementation used to compute the best dice score (default: 'nifty')

    Returns:
        float - the best symmetric dice score
    """
    assert impl in ('nifty', 'numpy')
    best_dice = _best_dice_nifty if impl == 'nifty' else _best_dice_numpy
    score1 = best_dice(segmentation, groundtruth)
    score2 = best_dice(groundtruth, segmentation)
    return min(score1, score2)
