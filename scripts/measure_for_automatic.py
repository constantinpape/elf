import os

import json
import numpy as np
import pandas as pd
import imageio.v3 as imageio

from traccuracy import run_metrics
from traccuracy.matchers import CTCMatcher
from traccuracy._tracking_graph import TrackingGraph
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy.loaders._ctc import _get_node_attributes, ctc_to_graph, _check_ctc


# ROOT = "/scratch/share/cidas/cca/experiments/results_for_automatic_tracking"
ROOT = "/home/pape/Work/my_projects/micro-sam/examples"


def _get_tracks_to_isbi():
    with open(os.path.join(ROOT, "tracking_result.json"), "r") as f:
        data = json.load(f)

    segmentation = imageio.imread(os.path.join(ROOT, "tracking_result.tif"))

    # we convert the parent ids to frames
    track_info = {}
    for parent_id, daughter_ids in data.items():
        parent_id = int(parent_id)
        frames = np.where(segmentation == parent_id)[0]

        if parent_id in track_info:
            parent_val = track_info[parent_id]['parent']
        else:
            parent_val = None

        # store the parent's track information
        track_info[parent_id] = {
            'frames': list(np.unique(frames)),
            'daughters': list(daughter_ids),
            'frame_div': frames.max() + 1,
            'parent': parent_val,
            'label': parent_id,
        }

        # next, store the daughter's track information
        for daughter_id in daughter_ids:
            frames = np.where(segmentation == daughter_id)[0]
            track_info[daughter_id] = {
                'frames': list(np.unique(frames)),
                'daughters': [],
                'frame_div': None,
                'parent': parent_id,
                'label': daughter_id
            }

    # now, the next step is to store track info for objects that did not split.
    for gt_id in np.unique(segmentation)[1:]:
        if gt_id in track_info:
            continue

        frames = np.where(segmentation == gt_id)[0]
        track_info[gt_id] = {
            'frames': list(np.unique(frames)),
            'daughters': [],
            'frame_div': None,
            'parent': None,
            'label': gt_id,
        }

    from deepcell_tracking import isbi_utils
    track_df = isbi_utils.trk_to_isbi(track_info)
    track_df.to_csv("automatic_tracks.csv")
    return track_df


def _get_tracks_df():
    tracks_path = "automatic_tracks.csv"
    segmentation = imageio.imread(os.path.join(ROOT, "tracking_result.tif"))
    if os.path.exists(tracks_path):
        track_df = pd.read_csv(tracks_path)
    else:
        track_df = _get_tracks_to_isbi()
    return segmentation, track_df


def _get_metrics_for_autotrack(segmentation, seg_df):
    # NOTE: for ground-truth
    gt = imageio.imread(os.path.join(ROOT, "tracking_gt_corrected.tif"))
    gt_nodes = _get_node_attributes(gt)
    gt_df = pd.read_csv("gt_tracks.csv")
    gt_G = ctc_to_graph(gt_df, gt_nodes)
    _check_ctc(gt_df, gt_nodes, gt)
    gt_T = TrackingGraph(gt_G, segmentation=gt, name="DynamicNuclearNet-GT")

    # NOTE: for segmentation results
    seg_nodes = _get_node_attributes(segmentation)
    seg_G = ctc_to_graph(seg_df, seg_nodes)
    _check_ctc(seg_df, seg_nodes, segmentation)
    seg_T = TrackingGraph(seg_G, segmentation=segmentation, name="DynamicNuclearNet-autotracking")

    ctc_results = run_metrics(
        gt_data=gt_T,
        pred_data=seg_T,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics(), DivisionMetrics(max_frame_buffer=0)]
    )
    print(ctc_results)


def main():
    segmentation, seg_df = _get_tracks_df()
    _get_metrics_for_autotrack(segmentation, seg_df)


main()
