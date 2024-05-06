import os
import numpy as np
import pandas as pd

from deepcell_tracking.isbi_utils import trk_to_isbi

from traccuracy import run_metrics
from traccuracy._tracking_graph import TrackingGraph
from traccuracy.matchers import CTCMatcher, IOUMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy.loaders._ctc import _get_node_attributes, ctc_to_graph, _check_ctc, load_ctc_data

from get_tracking_results import get_tracking_data, load_tracking_segmentation


def mark_potential_split(frames, last_frame, idx):
    if frames.max() == last_frame:  # object is tracked until the last frame
        split_frame = None  # they can't split in this case
        prev_parent_id = None
    else:  # object either goes out of frame or splits
        split_frame = frames.max()  # let's assume that it splits, we will know if it does or not
        prev_parent_id = idx
    return split_frame, prev_parent_id


def extract_df_from_segmentation(segmentation):
    track_ids = np.unique(segmentation)[1:]
    last_frame = segmentation.shape[0] - 1

    all_tracks = []
    prev_parent_id = None

    for idx in track_ids:
        frames = np.unique(np.where(segmentation == idx)[0])

        if frames.min() == 0:  # object starts at first frame
            pid = 0
            split_frame, prev_parent_id = mark_potential_split(frames, last_frame, idx)

        else:
            if split_frame is not None:  # takes the parent information from above
                # have fam is the end frame of the potential parent, so our frame has to be the next frame
                if split_frame + 1 == frames.min():
                    pid = prev_parent_id

                # otherwise we just have some track that starts so it's not the child
                else:
                    pid = 0
                    split_frame, prev_parent_id = mark_potential_split(frames, last_frame, idx)

            else:
                pid = 0  # assumes that it was an object that started at a random frame
                split_frame, prev_parent_id = mark_potential_split(frames, last_frame, idx)

        track_dict = {
            "Cell_ID": idx,
            "Start": frames.min(),
            "End": frames.max(),
            "Parent_ID": pid,
        }

        all_tracks.append(pd.DataFrame.from_dict([track_dict]))

    pred_tracks_df = pd.concat(all_tracks)
    return pred_tracks_df


def evaluate_tracking(labels, curr_lineages, segmentation_method):
    seg = load_tracking_segmentation(segmentation_method)

    if os.path.isdir(seg):  # for trackmate stardist
        seg_T = load_ctc_data(
            data_dir=seg, 
            track_path=os.path.join(seg, 'res_track.txt'),
            name=f'DynamicNuclearNet-{segmentation_method}'
        )

    else:  # for micro-sam
        seg_nodes = _get_node_attributes(seg)
        seg_df = extract_df_from_segmentation(seg)
        seg_G = ctc_to_graph(seg_df, seg_nodes)
        _check_ctc(seg_df, seg_nodes, seg)
        seg_T = TrackingGraph(seg_G, segmentation=seg, name=f"DynamicNuclearNet-{segmentation_method}")

    breakpoint()

    # calcuates node attributes for each detection
    gt_nodes = _get_node_attributes(labels)

    # converts inputs to isbi-tracking format - the version expected as inputs in traccuracy
    gt_df = trk_to_isbi(curr_lineages, path=None)

    # creates graphs from ctc-type info (isbi-type? probably means the same thing)
    gt_G = ctc_to_graph(gt_df, gt_nodes)

    # OPTIONAL: This tests if inputs (images, dfs and node attributes) to create tracking graphs are as expected
    _check_ctc(gt_df, gt_nodes, labels)

    gt_T = TrackingGraph(gt_G, segmentation=labels, name="DynamicNuclearNet-GT")

    ctc_results = run_metrics(
        gt_data=gt_T,
        pred_data=seg_T,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics(), DivisionMetrics(max_frame_buffer=0)],
    )
    print(ctc_results)

    breakpoint()

    iou_results = run_metrics(
        gt_data=gt_T,
        pred_data=seg_T,
        matcher=IOUMatcher(iou_threshold=0.1),
        metrics=[DivisionMetrics(max_frame_buffer=0)],
    )
    print(iou_results)


def main():
    raw, labels, curr_lineages, chosen_frames = get_tracking_data()

    segmentation_method = "vit_l_specialist"
    evaluate_tracking(labels, curr_lineages, segmentation_method)


if __name__ == "__main__":
    main()
