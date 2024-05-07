import os
import numpy as np
import pandas as pd

import h5py

from traccuracy import run_metrics
from traccuracy.matchers import CTCMatcher
from traccuracy._tracking_graph import TrackingGraph
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy.loaders._ctc import _get_node_attributes, ctc_to_graph, _check_ctc, load_ctc_data

from scipy.ndimage import binary_closing


# ROOT = "/scratch/usr/nimanwai/micro-sam/for_tracking/for_traccuracy/"  # hlrn
ROOT = "/media/anwai/ANWAI/results/micro-sam/for_traccuracy/"  # local


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


def evaluate_tracking(raw, labels, seg, segmentation_method, filter_label_ids):
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

    ids, sizes = np.unique(seg_df.Parent_ID.values, return_counts=True)
    print(ids, sizes)
    # calcuates node attributes for each detectionc
    gt_nodes = _get_node_attributes(labels)

    # converts inputs to isbi-tracking format - the version expected as inputs in traccuracy
    # it's preconverted using "from deepcell_tracking.isbi_utils import trk_to_isbi"
    gt_df = pd.read_csv(os.path.join(ROOT, "gt_tracks.csv"))
    mask = np.ones(len(gt_df), dtype="bool")
    # breakpoint()
    mask[np.isin(gt_df.Cell_ID, filter_label_ids)] = False
    gt_df = gt_df[mask]
    # breakpoint()

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


def size_filter(segmentation, min_size=100):
    ids, sizes = np.unique(segmentation, return_counts=True)
    filter_ids = ids[sizes < min_size]
    segmentation[np.isin(segmentation, filter_ids)] = 0
    return segmentation


def get_tracking_data(segmentation_method, visualize=False):
    _path = os.path.join(ROOT, "tracking_micro_sam.h5")

    with h5py.File(_path, "r") as f:
        raw = f["raw"][:]
        labels = f["labels"][:]

        if segmentation_method.startswith("vit"):
            segmentation = f[f"segmentations/{segmentation_method}"][:]
            segmentation = size_filter(segmentation)
        else:
            segmentation = os.path.join(ROOT, "trackmate_stardist", "01_RES")

    # test case
    def check_consecutive(instances):
        instance_ids = np.unique(instances)[1:]

        id_list = []
        for idx in instance_ids:
            frames = np.unique(np.where(instances == idx)[0])
            consistent_instance = (sorted(frames) == list(range(min(frames), max(frames) + 1)))
            if not consistent_instance:
                id_list.append(idx)

        return id_list

    def rectify_labels(instances):
        id_list = check_consecutive(instances)
        print("Closing instances", id_list)
        for idx in id_list:
            object_mask = (instances == idx)

            structuring_element = np.zeros((3, 1, 1))
            structuring_element[:, 0, 0] = 1
       
            closed_mask = binary_closing(object_mask.copy(), iterations=1, structure=structuring_element)
            # breakpoint()
            closed_mask = np.logical_or(object_mask, closed_mask)
            # breakpoint()

            instances[closed_mask] = idx

            # import napari
            # v = napari.Viewer()
            # v.add_image(closed_mask.astype("uint8") * 255, name="After Closing", blending="additive", colormap="blue")
            # v.add_image(object_mask.astype("uint8") * 255, name="Original")
            # v.add_labels(instances, visible=False)
            # napari.run()

        return instances

    filter_ids = check_consecutive(labels)
    labels[np.isin(labels, filter_ids)] = 0

    if not os.path.isdir(segmentation):
        segmentation = rectify_labels(segmentation)

    if visualize:
        import napari

        v = napari.Viewer()
        v.add_image(raw)
        if not os.path.isdir(segmentation):
            v.add_labels(segmentation, visible=False)

        napari.run()

    return raw, labels, segmentation, filter_ids


def main():
    segmentation_method = "vit_l_lm"

    raw, labels, segmentation, filter_ids = get_tracking_data(segmentation_method, visualize=False)
    evaluate_tracking(raw, labels, segmentation, segmentation_method, filter_ids)


if __name__ == "__main__":
    main()
