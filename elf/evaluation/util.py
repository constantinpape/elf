from typing import Dict, Tuple

import numpy as np
import nifty.ground_truth as ngt


def contigency_table(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> Tuple[Dict[int, int], Dict[int, int], np.ndarray, np.ndarray]:
    """Compute the pairs and counts in the contingency table for two segmentations.

    The contingency table counts the number of pixels that are shared between
    objects from seg_a and seg_b.

    Args:
        seg_a: the first segmentation.
        seg_b: the second segmentation.

    Returns:
        Dictionary that maps ids in seg_a to count.
        Dictionary that maps ids in seg_b to count.
        The pairs in the contigency table, giving first the id in seg_a and then the one in seg_b.
        The overlap count in the contigency table.
    """
    # compute the unique ids and couunts for seg a and seg b
    # and wrap them in a dict
    a_ids, a_counts = np.unique(seg_a, return_counts=True)
    b_ids, b_counts = np.unique(seg_b, return_counts=True)
    a_dict = dict(zip(a_ids, a_counts.astype("float64")))
    b_dict = dict(zip(b_ids, b_counts.astype("float64")))

    # compute the overlaps and overlap counts
    # use nifty gt functionality
    ovlp_comp = ngt.overlap(seg_a, seg_b)
    ovlps = [ovlp_comp.overlapArrays(ida, sorted=False) for ida in a_ids]
    p_ids = np.array([[ida, idb] for ida, ovlp in zip(a_ids, ovlps) for idb in ovlp[0]])
    p_counts = np.concatenate([ovlp[1] for ovlp in ovlps]).astype("float64")
    assert len(p_ids) == len(p_counts)

    # this is the alternative (naive) numpy impl, unfortunately this is very slow and
    # needs a lot of memory
    # pairs = np.concatenate((seg_a[:, None], seg_b[:, None]), axis=1)
    # p_ids_, p_counts_ = np.unique(pairs, return_counts=True, axis=0)

    return a_dict, b_dict, p_ids, p_counts


def compute_ignore_mask(seg_a, seg_b, ignore_a, ignore_b):
    """@private
    """
    if ignore_a is None and ignore_b is None:
        return None
    ignore_mask_a = None if ignore_a is None else np.isin(seg_a, ignore_a)
    ignore_mask_b = None if ignore_b is None else np.isin(seg_b, ignore_b)

    if ignore_mask_a is not None and ignore_mask_b is None:
        ignore_mask = ignore_mask_a
    elif ignore_mask_a is None and ignore_mask_b is not None:
        ignore_mask = ignore_mask_b
    elif ignore_mask_a is not None and ignore_mask_b is not None:
        ignore_mask = np.logical_and(ignore_mask_a, ignore_mask_b)

    # need to invert the mask
    return np.logical_not(ignore_mask)
