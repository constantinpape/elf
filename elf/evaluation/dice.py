from typing import Optional

import numpy as np
import bioimage_cpp as bic

# implementations based on:
# https://github.com/kreshuklab/sparse-object-embeddings/blob/master/pytorch3dunet/clustering/sbd.py


def dice_score(
    segmentation: np.ndarray,
    groundtruth: np.ndarray,
    threshold_seg: Optional[float] = 0,
    threshold_gt: Optional[float] = 0,
) -> float:
    """Compute the dice score between binarized segmentation and ground-truth.

    For comparing probaility maps (i.e. predictions in range [0, 1]) with this function
    you need to set the thresholds to None. Otherwise the results will be wrong.

    Args:
        segmentation: Candidate segmentation to evaluate.
        groundtruth: Groundtruth segmentation.
        threshold_seg: The threshold applied to the segmentation. If None, the segmentation is not thresholded.
        threshold_gt: The threshold applied to the ground-truth. If None, the groundtruth is not thresholded.

    Returns:
        The dice score.
    """
    assert segmentation.shape == groundtruth.shape, f"{segmentation.shape}, {groundtruth.shape}"
    if threshold_seg is None:
        seg = segmentation
    else:
        seg = segmentation > threshold_seg
    if threshold_gt is None:
        gt = groundtruth
    else:
        gt = groundtruth > threshold_gt

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
        _gt_seg = (gt == gt_idx).astype("uint8")
        dices = []
        for pred_idx in seg_labels:
            _pred_seg = (seg == pred_idx).astype("uint8")

            dice = dice_score(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    return np.mean(best_dices)


def _best_dice_bic(gt, seg, average_scores=True):
    overlaps = bic.utils.segmentation_overlap(gt, seg)

    gt_counts_table = overlaps.counts_a_table()
    seg_counts_table = overlaps.counts_b_table()
    gt_counts = {int(lab): int(cnt) for lab, cnt in zip(gt_counts_table["label"], gt_counts_table["count"])}
    seg_counts = {int(lab): int(cnt) for lab, cnt in zip(seg_counts_table["label"], seg_counts_table["count"])}

    gt_labels = [lab for lab in gt_counts if lab != 0]
    if len(gt_labels) == 0:
        return 0.0

    eps = 1e-7
    dice_scores = []
    for gt_id in gt_labels:
        gt_count = gt_counts[gt_id]
        ovlps = overlaps.overlaps_for_label_a(gt_id)
        ovlp_ids = ovlps["label"]
        ovlp_counts = ovlps["count"]
        zero_mask = ovlp_ids == 0
        if zero_mask.any():
            ovlp_ids, ovlp_counts = ovlp_ids[~zero_mask], ovlp_counts[~zero_mask]
        if len(ovlp_ids) == 0:
            dice_scores.append(0.)
            continue
        scores = [float(2 * count) / float(gt_count + seg_counts[int(seg_id)] + eps)
                  for seg_id, count in zip(ovlp_ids, ovlp_counts)]
        dice_scores.append(np.max(scores))

    if average_scores:
        return np.mean(dice_scores)
    else:
        return dice_scores


def symmetric_best_dice_score(
    segmentation: np.ndarray,
    groundtruth: np.ndarray,
    impl: str = "bic",
) -> float:
    """Compute the best symmetric dice score between the objects in the groundtruth and segmentation.

    This metric is used in the CVPPP instance segmentation challenge.

    Args:
        segmentation: Candidate segmentation to evaluate.
        groundtruth: Groundtruth segmentation.
        impl: Implementation used to compute the best dice score. The available implementations are 'bic' and 'numpy'.

    Returns:
        The best symmetric dice score.
    """
    assert impl in ("bic", "numpy")
    best_dice = _best_dice_bic if impl == "bic" else _best_dice_numpy
    score1 = best_dice(segmentation, groundtruth)
    score2 = best_dice(groundtruth, segmentation)
    return min(score1, score2)
