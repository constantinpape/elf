import os
import unittest
from shutil import rmtree

import numpy as np


# Minimal MaMuT xml with two spots in frame 0 connected by a single track, and
# one spot in frame 1 that must be ignored when extracting frame 0.
MAMUT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TrackMate version="6.0.1">
  <Model>
    <AllSpots nspots="3">
      <SpotsInFrame frame="0">
        <Spot ID="10" POSITION_Z="1.0" POSITION_Y="2.0" POSITION_X="3.0" />
        <Spot ID="11" POSITION_Z="4.0" POSITION_Y="4.0" POSITION_X="4.0" />
      </SpotsInFrame>
      <SpotsInFrame frame="1">
        <Spot ID="20" POSITION_Z="0.0" POSITION_Y="0.0" POSITION_X="0.0" />
      </SpotsInFrame>
    </AllSpots>
    <AllTracks>
      <Track TRACK_ID="7">
        <Edge SPOT_SOURCE_ID="10" SPOT_TARGET_ID="11" />
      </Track>
    </AllTracks>
  </Model>
</TrackMate>
"""


class TestMamut(unittest.TestCase):
    tmp_folder = "./tmp_mamut"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        self.path = os.path.join(self.tmp_folder, "tracks.xml")
        with open(self.path, "w") as f:
            f.write(MAMUT_XML)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def test_extract_tracks_as_volume_ids(self):
        from elf.tracking.mamut import extract_tracks_as_volume
        shape = (5, 5, 5)
        volume = extract_tracks_as_volume(
            self.path, timepoint=0, shape=shape, voxel_size=(1.0, 1.0, 1.0), binary=False
        )
        self.assertEqual(volume.shape, shape)
        # Both spots in frame 0 belong to track 7 and are placed at their (z, y, x).
        self.assertEqual(volume[1, 2, 3], 7)
        self.assertEqual(volume[4, 4, 4], 7)
        # Only the two frame-0 spots are set.
        self.assertEqual(np.count_nonzero(volume), 2)

    def test_extract_tracks_as_volume_binary(self):
        from elf.tracking.mamut import extract_tracks_as_volume
        shape = (5, 5, 5)
        mask = extract_tracks_as_volume(
            self.path, timepoint=0, shape=shape, voxel_size=(1.0, 1.0, 1.0), binary=True
        )
        self.assertEqual(mask.dtype, np.dtype("bool"))
        coords = set(map(tuple, np.argwhere(mask)))
        self.assertEqual(coords, {(1, 2, 3), (4, 4, 4)})

    def test_extract_tracks_as_volume_voxel_size(self):
        from elf.tracking.mamut import extract_tracks_as_volume
        # Positions are divided by the voxel size to recover pixel coordinates.
        shape = (5, 5, 5)
        volume = extract_tracks_as_volume(
            self.path, timepoint=0, shape=shape, voxel_size=(1.0, 2.0, 1.0), binary=False
        )
        # POSITION_Y of spot 10 is 2.0 -> 2.0 / 2.0 = 1.
        self.assertEqual(volume[1, 1, 3], 7)

    def test_extract_tracks_as_volume_missing_timepoint(self):
        from elf.tracking.mamut import extract_tracks_as_volume
        with self.assertRaises(RuntimeError):
            extract_tracks_as_volume(
                self.path, timepoint=99, shape=(5, 5, 5), voxel_size=(1.0, 1.0, 1.0)
            )


if __name__ == "__main__":
    unittest.main()
