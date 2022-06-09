import multiprocessing as mp
from concurrent import futures

import numpy as np
import nifty.ground_truth as ngt
import tqdm
from scipy.ndimage.morphology import distance_transform_edt

#
# TODO naive tracking with divisions
#


def _compute_distance_matches(
    next_t, current_t, unmatched, max_assignment_distance, n_threads, t
):
    mask = np.isin(next_t, unmatched)
    distances, indices = distance_transform_edt(np.logical_not(mask), return_indices=True)
    current_ids = np.unique(current_t)

    def _find_distance_match(current_id):
        mask = current_t == current_id
        masked_distances = distances[mask]
        min_dist_point = np.argmin(masked_distances)
        min_dist = masked_distances[min_dist_point]
        if min_dist > max_assignment_distance:
            return None, None
        index = indices[:, mask]
        index = tuple(ind[min_dist_point] for ind in index)
        next_id = next_t[index]
        return next_id, min_dist

    with futures.ThreadPoolExecutor(n_threads) as tp:
        matched_next = list(tp.map(_find_distance_match, current_ids[1:]))
        matched_next = {curr_id: match for curr_id, match in zip(current_ids[1:], matched_next)}

    distance_matches = {}
    match_distances = {}
    for curr_id, (next_id, dist) in matched_next.items():
        if next_id is None:
            continue
        if next_id in distance_matches:
            prev_dist = match_distances[next_id]
            if dist < prev_dist:
                distance_matches[next_id] = curr_id
                match_distances[next_id] = dist
        else:
            distance_matches[next_id] = curr_id
            match_distances[next_id] = dist
    return distance_matches


def naive_tracking(time_series, max_assignment_distance, n_threads=-1, verbose=False):
    """Naive tracking without divisions.

    Arguments:
        time_series [np.ndarray] -
        max_assignment_distance [float] -
        allow_divisions [bool] -
        n_threads [int] -
        verbose [bool] -
    """
    track_ids = {}
    nt = len(time_series)

    # initialize the track ids with the first time point
    current_t = time_series[0]
    current_ids = np.setdiff1d(np.unique(current_t), [0])
    track_ids = {0: {curr_id: track_id for track_id, curr_id in enumerate(current_ids, 1)}}
    next_track_id = len(track_ids[0]) + 1

    if n_threads == -1:
        n_threads = mp.cpu_count()

    range_ = tqdm.trange(1, nt, desc="Naive tracking") if verbose else range(1, nt)
    for t in range_:
        next_t = time_series[t]
        next_ids = np.setdiff1d(np.unique(next_t), [0])

        # compute the area overlaps beween current and next time point
        ovlp_comp = ngt.overlap(next_t, current_t)
        ovlps = {next_id: ovlp_comp.overlapArrays(next_id, sorted=True) for next_id in next_ids}

        ovlp_ids, ovlp_counts = {}, {}
        for next_id, (labels, counts) in ovlps.items():
            if labels[0] == 0:
                labels, counts = labels[1:], counts[1:]
            if len(labels) == 0:
                continue
            ovlp_ids[next_id] = labels[0]
            ovlp_counts[next_id] = counts[0]

        # assign track ids based on maximum overlap
        prev_track_assignments = track_ids[t - 1]
        track_matches = {next_id: prev_track_assignments[matched] for next_id, matched in ovlp_ids.items()}
        unmatched = list(set(next_ids) - set(track_matches.keys()))
        if unmatched and max_assignment_distance > 0:
            distance_matches = _compute_distance_matches(
                next_t, current_t, unmatched, max_assignment_distance, n_threads, t
            )
            distance_matches = {
                next_id: prev_track_assignments[matched] for next_id, matched in distance_matches.items()
            }
            # don't distance match to previous overlap matches
            ovlp_tracks = set(track_matches.values())
            distance_matches = {k: v for k, v in distance_matches.items() if v not in ovlp_tracks}
            track_matches = {**track_matches, **distance_matches}

        unmatched = list(set(next_ids) - set(track_matches.keys()))
        new_tracks = {next_id: track_id for track_id, next_id in enumerate(unmatched, next_track_id)}
        next_track_id += len(new_tracks)

        track_ids_t = {**track_matches, **new_tracks}
        track_ids[t] = track_ids_t
        current_t = next_t
        current_ids = next_ids

    return track_ids
