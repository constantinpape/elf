import os
from pathlib import Path

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from deepcell_tracking.utils import load_trks


ROOT = "/scratch/projects/nim00007/sam/for_tracking"


def load_tracking_segmentation(experiment):
    result_dir = os.path.join(ROOT, "results")
    if experiment == "vit_l":
        seg_path = os.path.join(result_dir, "vit_l.tif")
    elif experiment == "vit_l_lm":
        seg_path = os.path.join(result_dir, "vit_l_lm.tif")
    elif experiment == "vit_l_specialist":
        seg_path = os.path.join(result_dir, "vit_l_lm_specialist.tif")
    elif experiment == "trackmate_stardist":
        seg_path = os.path.join(result_dir, "trackmate_stardist", "every_3rd_fr_result.tif")
    else:
        raise ValueError(experiment)

    return imageio.imread(seg_path)


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

    return raw, labels, curr_lineages, chosen_frames
