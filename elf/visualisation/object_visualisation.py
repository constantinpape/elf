from functools import partial

import numba
import numpy as np

from ..evaluation.dice import _best_dice_nifty
from ..evaluation.matching import _compute_scores
from ..evaluation.variation_of_information import object_vi


@numba.jit()
def map_dict_2d(array, replace_dict):
    """@private
    """
    out = np.zeros_like(array, dtype="float32")
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            out[i, j] = replace_dict[array[i, j]]
    return out


@numba.jit()
def map_dict_3d(array, replace_dict):
    """@private
    """
    out = np.zeros_like(array, dtype="float32")
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                out[i, j, k] = replace_dict[array[i, j, k]]
    return out


def visualise_object_scores(segmentation, groundtruth, scoring_function, ignore_background):
    """@private
    """
    assert segmentation.shape == groundtruth.shape
    assert segmentation.ndim in (2, 3)
    object_ids = np.unique(segmentation)
    scores = scoring_function(segmentation, groundtruth)
    assert len(scores) == len(object_ids), f"{len(scores)}, {len(object_ids)}"
    if ignore_background:
        scores[0] = 0.0
    object_layer = map_dict_2d(segmentation, scores) if segmentation.ndim == 2 else map_dict_3d(segmentation, scores)
    return object_layer


def iou_scoring(segmentation, groundtruth):
    """@private
    """
    object_ids = np.unique(segmentation)
    scores = _compute_scores(segmentation, groundtruth, "iou")[-1]
    scores = np.max(scores, axis=1)
    scores = np.concatenate([np.array([0.0]), scores], axis=0)
    assert len(scores) == len(object_ids), f"{len(scores)}, {len(object_ids)}"
    scores = {k: v for k, v in zip(object_ids, scores)}
    return scores


def visualise_iou_scores(segmentation: np.ndarray, groundtruth: np.ndarray) -> np.ndarray:
    """Visualize the IOU scores of a candidate segmentation compared to a groundtruth segmentation.

    Args:
        segmentation: The candidate segmentation.
        groundtruth: The groundtruth segmentation.

    Returns:
        The score visualization.
    """
    object_layer = visualise_object_scores(segmentation, groundtruth, iou_scoring, ignore_background=True)
    return object_layer


def voi_scoring(segmentation, groundtruth, voi):
    """@private
    """
    scores = object_vi(groundtruth, segmentation)
    if voi == "split":
        scores = {k: v[0] for k, v in scores.items()}
    elif voi == "merge":
        scores = {k: v[1] for k, v in scores.items()}
    elif voi == "full":
        scores = {k: sum(v) for k, v in scores.items()}
    return scores


def visualise_voi_scores(
    segmentation: np.ndarray, groundtruth: np.ndarray, voi: str = "full"
) -> np.ndarray:
    """Visualize the variation of information scores of a candidate segmentation compared to a groundtruth segmentation.

    Args:
        segmentation: The candidate segmentation.
        groundtruth: The groundtruth segmentation.
        voi: The variation of information type to visualize. One of 'split', 'merge', 'full'.

    Returns:
        The score visualization.
    """
    assert voi in ("split", "merge", "full")
    scoring_function = partial(voi_scoring, voi=voi)
    object_layer = visualise_object_scores(segmentation, groundtruth, scoring_function, ignore_background=False)
    return object_layer


def dice_scoring(segmentation, groundtruth):
    """@private
    """
    object_ids = np.unique(segmentation)
    scores = _best_dice_nifty(segmentation, groundtruth, average_scores=False)
    scores = np.concatenate([np.array([0.0]), scores], axis=0)
    assert len(scores) == len(object_ids), f"{len(scores)}, {len(object_ids)}"
    scores = {k: v for k, v in zip(object_ids, scores)}
    return scores


def visualise_dice_scores(segmentation: np.ndarray, groundtruth: np.ndarray) -> np.ndarray:
    """Visualize the dice scores of a candidate segmentation compared to a groundtruth segmentation.

    Args:
        segmentation: The candidate segmentation.
        groundtruth: The groundtruth segmentation.

    Returns:
        The score visualization.
    """
    scoring_function = dice_scoring
    object_layer = visualise_object_scores(segmentation, groundtruth, scoring_function, ignore_background=True)
    return object_layer
