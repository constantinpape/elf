import xml.etree.ElementTree as ET
import numpy as np

#
# adapted from code shared by @wolny
#


def _to_zyx_coordinates(mamut_coordinates, voxel_size):
    """Takes MaMUT coordinates and the size of the voxel and recovers the pixel coordinates
    """
    # convert string to double
    mamut_coordinates = np.array(list(map(float, mamut_coordinates)))
    # recover pixel coordinates
    return list(map(int, mamut_coordinates / voxel_size))


def _extract_tracks(root, flatten_spots=True):
    all_tracks = root.find('Model').find('AllTracks')
    tracks = {}
    for track in all_tracks:
        track_id = int(track.attrib['TRACK_ID'])
        spots = [[edge.attrib['SPOT_SOURCE_ID'], edge.attrib['SPOT_TARGET_ID']] for edge in track.findall('Edge')]
        if flatten_spots:
            spots = [spot for edge_spots in spots for spot in edge_spots]
        tracks[track_id] = spots

    return tracks


def extract_tracks_as_volume(path, timepoint, shape, voxel_size, binary=False):
    """Extract tracks as volume from MaMuT xml.

    Arguments:
        path [str] - path to the xml file with tracks
        timepoint [int] - timepoint for which to extract the tracks
        shape [tuple[int]] - shape of the output volume
        voxel_size [tuple[float]] - voxel size
    Returns:
        np.ndarray -
    """
    # get root XML element
    root = ET.parse(path).getroot()
    # retrieve all of the spots
    all_spots = root.find('Model').find('AllSpots')

    # get all spots for a given time frame
    spots = next((s for s in all_spots if s.attrib['frame'] == timepoint), None)

    if spots is None:
        raise RuntimeError('Could not find spots for time frame: ', timepoint)

    # get pixel coordinates
    pixel_coordinates = np.array([_to_zyx_coordinates(
        [spot.attrib['POSITION_Z'], spot.attrib['POSITION_Y'], spot.attrib['POSITION_X']],
        np.array([vsize for vsize in voxel_size])
    ) for spot in spots])

    # TODO how do we save the track id ???
    # save spots to nuclei file
    z = pixel_coordinates[:, 0]
    y = pixel_coordinates[:, 1]
    x = pixel_coordinates[:, 2]

    # extract the volume as binary
    if binary:
        spot_mask = np.zeros(shape, dtype='bool')
        spot_mask[z, y, x] = 1
        return spot_mask

    # extract volume with track ids
    spot_ids = [spot.attrib['Spot ID'] for spot in spots]
    track_volume = np.zeros(shape, dtype='uint32')

    tracks = _extract_tracks(root)
    spots_to_tracks = {spot: track for track, spots in tracks.items() for spot in spots}
    track_ids = np.array([spots_to_tracks[spot_id] for spot_id in spot_ids])

    track_volume[z, y, x] = track_ids

    return track_volume
