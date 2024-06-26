"""Utility functions for setting up tracking problems in microscopy data.

Can be used with the functionality from `motile_tracking` to solve tracking problems with
motile or with other python tracking libraries.
"""

import nifty.ground_truth as ngt
import numpy as np

from scipy.spatial.distance import cdist
from skimage.measure import regionprops, label
from scipy.ndimage import binary_closing
from skimage.segmentation import relabel_sequential
from tqdm import trange


def compute_edges_from_overlap(segmentation, verbose=True):
    def compute_overlap_between_frames(frame_a, frame_b):
        overlap_function = ngt.overlap(frame_a, frame_b)

        node_ids = np.unique(frame_a)[1:]
        overlaps = [overlap_function.overlapArraysNormalized(node_id) for node_id in node_ids]

        source_ids = [src for node_id, ovlp in zip(node_ids, overlaps) for src in [node_id] * len(ovlp[0])]
        target_ids = [ov for ovlp in overlaps for ov in ovlp[0]]
        overlap_values = [ov for ovlp in overlaps for ov in ovlp[1]]
        assert len(source_ids) == len(target_ids) == len(overlap_values), \
            f"{len(source_ids)}, {len(target_ids)}, {len(overlap_values)}"

        edges = [
            {"source": source_id, "target": target_id, "score": ovlp}
            for source_id, target_id, ovlp in zip(source_ids, target_ids, overlap_values)
        ]

        # filter out zeros
        edges = [edge for edge in edges if edge["target"] != 0]
        return edges

    edges = []
    for t in trange(segmentation.shape[0] - 1, disable=not verbose, desc="Compute edges via overlap"):
        this_frame = segmentation[t]
        next_frame = segmentation[t + 1]
        frame_edges = compute_overlap_between_frames(this_frame, next_frame)
        edges.extend(frame_edges)
    return edges


def compute_edges_from_centroid_distance(segmentation, max_distance, normalize_distances=True, verbose=True):
    nt = segmentation.shape[0]
    props = regionprops(segmentation)
    centroids_and_labels = [[prop.centroid[0], prop.centroid[1:], prop.label] for prop in props]

    centroids, labels = {}, {}
    for t, centroid, label_id in centroids_and_labels:
        centroids[t] = centroids.get(t, []) + [centroid]
        labels[t] = labels.get(t, []) + [label_id]
    centroids = {t: np.stack(np.array(val)) for t, val in centroids.items()}
    labels = {t: np.array(val) for t, val in labels.items()}

    def compute_dist_between_frames(t):
        centers_a, centers_b = centroids[t], centroids[t + 1]
        labels_a, labels_b = labels[t], labels[t + 1]
        assert len(centers_a) == len(labels_a)
        assert len(centers_b) == len(labels_b)

        distances = cdist(centers_a, centers_b)
        edge_mask = distances <= max_distance
        distance_values = distances[edge_mask]

        idx_a, idx_b = np.where(edge_mask)
        source_ids, target_ids = labels_a[idx_a], labels_b[idx_b]
        assert len(distance_values) == len(source_ids) == len(target_ids)

        return source_ids, target_ids, distance_values
        # return edges

    source_ids, target_ids, distances = [], [], []
    for t in trange(nt - 1, disable=not verbose, desc="Compute edges via centroid distance"):
        this_src, this_tgt, this_dist = compute_dist_between_frames(t)
        source_ids.extend(this_src), target_ids.extend(this_tgt), distances.extend(this_dist)

    if normalize_distances:
        distances = np.array(distances)
        max_dist = distances.max()
        distances = 1.0 - distances / max_dist

    edges = [
        {"source": source_id, "target": target_id, "score": distance}
        for source_id, target_id, distance in zip(source_ids, target_ids, distances)
    ]
    return edges


# TODO does this work for 4d data (time + 3d)? if no we need to iterate over the time axis
def compute_node_costs_from_foreground_probabilities(segmentation, probabilities, cost_attribute="mean_intensity"):
    props = regionprops(segmentation, probabilities)
    costs = [getattr(prop, cost_attribute) for prop in props]
    return costs


def relabel_segmentation_across_time(segmentation):
    offset = 0
    relabeled = []
    for frame in segmentation:
        frame, _, _ = relabel_sequential(frame)
        frame[frame != 0] += offset
        offset = frame.max()
        relabeled.append(frame)
    return np.stack(relabeled)


def preprocess_closing(slice_segmentation, gap_closing, verbose=True):
    binarized = slice_segmentation > 0
    structuring_element = np.zeros((3, 1, 1))
    structuring_element[:, 0, 0] = 1
    closed_segmentation = binary_closing(binarized, iterations=gap_closing, structure=structuring_element)

    new_segmentation = np.zeros_like(slice_segmentation)
    n_slices = new_segmentation.shape[0]

    def process_slice(z, offset):
        seg_z = slice_segmentation[z]

        # Closing does not work for the first and last gap slices
        if z < gap_closing or z >= (n_slices - gap_closing):
            seg_z, _, _ = relabel_sequential(seg_z, offset=offset)
            offset = int(seg_z.max()) + 1
            return seg_z, offset

        # Apply connected components to the closed segmentation.
        closed_z = label(closed_segmentation[z])

        # Map objects in the closed and initial segmentation.
        # We take objects from the closed segmentation unless they
        # have overlap with more than one object from the initial segmentation.
        # This indicates wrong merging of closeby objects that we want to prevent.
        matches = ngt.overlap(closed_z, seg_z)
        matches = {seg_id: matches.overlapArrays(seg_id, sorted=False)[0]
                   for seg_id in range(1, int(closed_z.max() + 1))}
        matches = {k: v[v != 0] for k, v in matches.items()}

        ids_initial, ids_closed = [], []
        for seg_id, matched in matches.items():
            if len(matched) > 1:
                ids_initial.extend(matched.tolist())
            else:
                ids_closed.append(seg_id)

        seg_new = np.zeros_like(seg_z)
        closed_mask = np.isin(closed_z, ids_closed)
        seg_new[closed_mask] = closed_z[closed_mask]

        if ids_initial:
            initial_mask = np.isin(seg_z, ids_initial)
            seg_new[initial_mask] = relabel_sequential(seg_z[initial_mask], offset=seg_new.max() + 1)[0]

        seg_new, _, _ = relabel_sequential(seg_new, offset=offset)
        max_z = seg_new.max()
        if max_z > 0:
            offset = int(max_z) + 1

        return seg_new, offset

    # Further optimization: parallelize
    offset = 1
    for z in trange(n_slices, disable=not verbose):
        new_segmentation[z], offset = process_slice(z, offset)

    return new_segmentation
