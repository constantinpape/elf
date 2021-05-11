import numpy as np

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


def _best_dice(gt, seg):
    gt_lables = np.setdiff1d(np.unique(gt), [0])
    seg_labels = np.setdiff1d(np.unique(seg), [0])

    best_dices = []
    for gt_idx in gt_lables:
        _gt_seg = (gt == gt_idx).astype('uint8')
        dices = []
        for pred_idx in seg_labels:
            _pred_seg = (seg == pred_idx).astype('uint8')

            dice = dice_score(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    return np.mean(best_dices)


def symmetric_best_dice_score(segmentation, groundtruth):
    """ Compute the best symmetric dice score between the objects in the groundtruth and segmentation.

    This metric is used in the CVPPP instance segmentation challenge.

    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth

    Returns:
        float - the best symmetric dice score
    """
    score1 = _best_dice(segmentation, groundtruth)
    score2 = _best_dice(groundtruth, segmentation)
    return min(score1, score2)
