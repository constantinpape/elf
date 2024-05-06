import numpy as np

from deepcell_tracking.isbi_utils import trk_to_isbi

from traccuracy.loaders._ctc import _get_node_attributes

from get_tracking_results import get_tracking_data, load_tracking_segmentation


def extract_df_from_segmentation(segmentation):
    track_ids = np.unique(segmentation)[1:]
    last_frame = segmentation.shape[0] - 1

    all_tracks = []
    splits = 0
    for idx in track_ids:

        frames = np.unique(np.where(segmentation == idx)[0])

        if frames.min() == 0:  # object starts at first frame
            if frames.max() == last_frame:  # object is tracked until the last frame
                pid = 0
                have_fam = None  # they can't split in this case
            else:  # object either goes out of frame or splits
                pid = 0
                have_fam = frames.max()  # let's assume that it splits, we will know if it does or not

        else:
            if have_fam is not None:  # takes the parent information from above
                pid = have_fam
                splits += 1

                if splits > 2:  # assumes every mother cell splits into two daughter cells
                    print("The mother cell has made enough daughter splits, hence this is a new object.")
                    splits = 0
                    # pid = 0  # this is the case where an objects appears at nth frame and has no parent id
            else:
                pid = 0  # assumes that it was an object that started at a random frame

        track_dict = {
            "Cell_ID": idx,
            "Start": frames.min(),
            "End": frames.max(),
            "Parent_ID": pid,
        }

        print(track_dict)
        all_tracks.append(track_dict)

        breakpoint()


def evaluate_tracking(raw, labels, curr_lineages, chosen_frames, segmentation_method):
    seg = load_tracking_segmentation(segmentation_method)

    # calcuates node attributes for each detection
    gt_df = _get_node_attributes(labels)
    seg_df = _get_node_attributes(seg)

    # converts inputs to isbi-track format - the version expected as inputs in traccuracy
    output = trk_to_isbi(curr_lineages, path=None)

    df = extract_df_from_segmentation(seg)


def main():
    raw, labels, curr_lineages, chosen_frames = get_tracking_data()

    segmentation_method = "vit_l_specialist"
    evaluate_tracking(raw, labels, curr_lineages, chosen_frames, segmentation_method)


if __name__ == "__main__":
    main()
