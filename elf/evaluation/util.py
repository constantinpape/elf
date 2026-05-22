from typing import Dict, Tuple

import numpy as np
import bioimage_cpp as bic


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
    overlap = bic.utils.segmentation_overlap(seg_a, seg_b)

    counts_a = overlap.counts_a_table()
    counts_b = overlap.counts_b_table()
    a_dict = {int(lab): float(cnt) for lab, cnt in zip(counts_a["label"], counts_a["count"])}
    b_dict = {int(lab): float(cnt) for lab, cnt in zip(counts_b["label"], counts_b["count"])}

    table = overlap.overlap_table()
    p_ids = np.stack([table["label_a"], table["label_b"]], axis=1)
    p_counts = table["count"].astype("float64")

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
