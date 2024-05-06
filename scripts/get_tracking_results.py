import os
from pathlib import Path

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from deepcell_tracking.utils import load_trks


ROOT = "/scratch/projects/nim00007/sam/for_tracking"


def load_tracking_segmentation(experiment):
    result_dir = os.path.join(ROOT, "results")

    if experiment.startswith("vit"):
        if experiment == "vit_l":
            seg_path = os.path.join(result_dir, "vit_l.tif")
            seg = imageio.imread(seg_path)
            # HACK
            ignore_labels = [8, 44, 57, 102, 50]

        elif experiment == "vit_l_lm":
            seg_path = os.path.join(result_dir, "vit_l_lm.tif")
            seg = imageio.imread(seg_path)
            # HACK
            ignore_labels = []

        elif experiment == "vit_l_specialist":
            seg_path = os.path.join(result_dir, "vit_l_lm_specialist.tif")
            seg = imageio.imread(seg_path)
            # HACK
            ignore_labels = [88, 45, 30, 46]

        # elif experiment == "trackmate_stardist":
        #     seg_path = os.path.join(result_dir, "trackmate_stardist", "every_3rd_fr_result.tif")
        #     seg = imageio.imread(seg_path)

        else:
            raise ValueError(experiment)

        # HACK:
        # we remove some labels as they have a weird lineage, is creating issues for creating the graph
        # (e.g. frames where the object exists: 1, 2, 4, 5, 6)
        seg[np.isin(seg, ignore_labels)] = 0

        return seg

    else:  # return the result directory for stardist
        return os.path.join(result_dir, "trackmate_stardist", "01_RES")


def check_tracking_results(raw, labels, curr_lineages, chosen_frames):
    seg_default = load_tracking_segmentation("vit_l")
    seg_generalist = load_tracking_segmentation("vit_l_lm")
    seg_specialist = load_tracking_segmentation("vit_l_specialist")

    # let's get the tracks only for the objects present per frame
    for idx in np.unique(labels)[1:]:
        lineage = curr_lineages[idx]
        lineage["frames"] = [frame for frame in lineage["frames"] if frame in chosen_frames]

    import napari
    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(labels)

    v.add_labels(seg_default, visible=False)
    v.add_labels(seg_generalist, visible=False)
    v.add_labels(seg_specialist, visible=False)

    napari.run()


def get_tracking_data():
    data_dir = os.path.join(ROOT, "data", "DynamicNuclearNet-tracking-v1_0")
    data_source = np.load(os.path.join(data_dir, "data-source.npz"), allow_pickle=True)

    fname = "test.trks"
    track_file = os.path.join(data_dir, fname)
    split_name = Path(track_file).stem

    data = load_trks(track_file)

    X = data["X"]
    y = data["y"]
    lineages = data["lineages"]

    meta = pd.DataFrame(
        data_source[split_name],
        columns=["filename", "experiment", "pixel_size", "screening_passed", "time_step", "specimen"]
    )
    # print(meta)

    # let's convert the data to expected shape
    X = X.squeeze(-1)
    y = y.squeeze(-1)

    # NOTE: chosen slice for the tracking user study.
    _slice = 7
    raw, labels = X[_slice, ...], y[_slice, ...]
    curr_lineages = lineages[_slice]

    # NOTE: let's get every third frame of data and see how it looks
    chosen_frames = list(range(0, raw.shape[0], 3))
    raw = np.stack([raw[frame] for frame in chosen_frames])
    labels = np.stack([labels[frame] for frame in chosen_frames])

    # let's create a value map
    frmaps = {}
    for i, frval in enumerate(chosen_frames):
        frmaps[frval] = i

    # let's remove frames which are not a part of our chosen frames
    for k, v in curr_lineages.items():
        curr_frames = v["frames"]
        v["frames"] = [frmaps[frval] for frval in curr_frames if frval in chosen_frames]

    # HACK:
    # we remove label with id 62 as it has a weird lineage, is creating issues for creating the graph
    ignore_labels = [62, 87, 92, 99, 58]
    labels[np.isin(labels, ignore_labels)] = 0
    for _label in ignore_labels:
        curr_lineages.pop(_label)

    return raw, labels, curr_lineages, chosen_frames
