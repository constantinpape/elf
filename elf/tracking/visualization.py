import numpy as np
import vigra

from nifty.tools import takeDict
from skimage.measure import regionprops


def compute_centers(labels, use_eccentricity=False):
    # this is more accurate, but extremely expensive
    if use_eccentricity:
        centers = vigra.filters.eccentricityCenters(labels.astype("uint32"))
        # TODO need to process this further
    else:
        props = regionprops(labels)
        centers = {prop.label: prop.centroid for prop in props}
    return centers


def color_by_tracking(segmentation, track_assignments, size_filter=0):
    tracking = np.zeros_like(segmentation)
    track_ids = np.unique([val for assignments in track_assignments.values() for val in assignments.values()])
    tracks_to_times = {track_id: [] for track_id in track_ids}
    for t in range(tracking.shape[0]):
        assignments = track_assignments[t]
        assignments[0] = 0
        track_t = takeDict(assignments, segmentation[t])
        if size_filter > 0:
            ids, sizes = np.unique(track_t, return_counts=True)
            too_small = ids[sizes < size_filter]
            track_t[np.isin(track_t, too_small)] = 0
        tracking[t] = track_t
        ids_t = np.unique(track_t)[1:]
        for track_id in ids_t:
            tracks_to_times[track_id].append(t)
    assert tracking.shape == segmentation.shape
    return tracking, tracks_to_times


def visualize_tracks(viewer, segmentation, track_assignments,
                     edge_width=4, size_filter=0, show_full_tracks=False,
                     selected_tracks=None):
    tracking, tracks_to_times = color_by_tracking(segmentation, track_assignments)
    track_ids = np.unique(tracking)[1:]

    color_map = {
        track_id: np.array(np.random.rand(3).tolist() + [1]) for track_id in track_ids
    }
    color_map[0] = np.array([0, 0, 0, 0])

    track_start = {track_id: np.min(tracks_to_times[track_id]) for track_id in track_ids}
    track_stop = {track_id: np.max(tracks_to_times[track_id]) + 1 for track_id in track_ids}

    current_centers = compute_centers(tracking[0])
    lines, line_colors = [], []
    for t in range(1, len(tracking)):
        next_centers = compute_centers(tracking[t])
        line_tracks = [track_id for track_id in current_centers if track_id in next_centers]
        if selected_tracks:
            line_tracks = set(line_tracks).intersection(set(selected_tracks))

        if show_full_tracks:
            lines_t, line_colors_t = [], []
            for track_id in line_tracks:
                t0, t1 = track_start[track_id], track_stop[track_id]
                lines_t.extend([
                    np.array([(t_track,) + current_centers[track_id], (t_track,) + next_centers[track_id]])
                    for t_track in range(t0, t1)
                ])
                line_colors_t.extend([color_map[track_id]] * (t1 - t0))

        else:
            lines_t = [
                np.array([(t - 1,) + current_centers[track_id], (t - 1,) + next_centers[track_id]])
                for track_id in line_tracks
            ]
            line_colors_t = [color_map[track_id] for track_id in line_tracks]

        assert len(lines_t) == len(line_colors_t)
        lines.extend(lines_t)
        line_colors.extend(line_colors_t)
        current_centers = next_centers

    viewer.add_labels(tracking, color=color_map)
    viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=edge_width,
        edge_color=line_colors
    )
