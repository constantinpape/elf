import os
from glob import glob

import imageio.v3 as imageio
import napari
import numpy as np

from elf.tracking.motile_tracking import track_with_motile, get_representation_for_napari


def get_ctc_hela_data():
    # load the data. you can download it from TODO
    image_folder = "/home/pape/Work/my_projects/micro-sam/examples/data/DIC-C2DH-HeLa.zip.unzip/DIC-C2DH-HeLa/01"
    images = np.stack([imageio.imread(path) for path in sorted(glob(os.path.join(image_folder, "*.tif")))])

    seg_folder = "/home/pape/Work/my_projects/micro-sam/examples/finetuning/data/hela-ctc-01-gt.zip.unzip/masks"
    segmentation = np.stack([imageio.imread(path) for path in sorted(glob(os.path.join(seg_folder, "*.tif")))])
    assert images.shape == segmentation.shape

    return images, segmentation


def default_tracking():
    images, segmentation = get_ctc_hela_data()

    # run the tracking and get visualization data for napari
    segmentation, lineage_graph, lineages, track_graph, tracks = track_with_motile(segmentation)
    tracking_result, track_data, parent_graph = get_representation_for_napari(
        segmentation, lineage_graph, lineages, tracks
    )

    # visualize with napari
    v = napari.Viewer()
    v.add_image(images)
    v.add_labels(tracking_result)
    v.add_tracks(track_data, name="tracks", graph=parent_graph)
    napari.run()


def tracking_with_custom_edge_function():
    from functools import partial
    from elf.tracking.tracking_utils import compute_edges_from_centroid_distance

    images, segmentation = get_ctc_hela_data()

    # run the tracking and get visualization data for napari
    edge_cost_function = partial(compute_edges_from_centroid_distance, max_distance=50)
    segmentation, lineage_graph, lineages, track_graph, tracks = track_with_motile(
        segmentation, edge_cost_function=edge_cost_function
    )
    tracking_result, track_data, parent_graph = get_representation_for_napari(
        segmentation, lineage_graph, lineages, tracks
    )

    # visualize with napari
    v = napari.Viewer()
    v.add_image(images)
    v.add_labels(tracking_result)
    v.add_tracks(track_data, name="tracks", graph=parent_graph)
    napari.run()


def main():
    default_tracking()
    # tracking_with_custom_edge_function()


if __name__ == "__main__":
    main()
