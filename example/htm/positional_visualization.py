#
# use elf.htm for positional visualization of data from a multi-well plate using napari
# the example data is available at https://owncloud.gwdg.de/index.php/s/eu8JMlUFZ82ccHT
#


import argparse
import os
from glob import glob

import elf.htm as htm
import h5py
import numpy as np
from tqdm import tqdm


def load_channel(folder, channel_name):
    files = glob(os.path.join(folder, "*.h5"))
    files.sort()
    data = {}
    for ff in tqdm(files, desc=f"Load {channel_name}"):
        sample = "_".join(os.path.splitext(os.path.basename(ff))[0].split("_")[:2])
        with h5py.File(ff, "r") as f:
            im = f[channel_name][:]
        data[sample] = im
    return data


def create_positions(samples):
    im_w = 930
    im_h = 1024

    im_spacing = 8
    well_spacing = 64

    well_names = ["C01", "C02", "C03"]
    well_ids = {name: ii for ii, name in enumerate(well_names)}

    positions = {}
    scores = {}
    for sample in samples:
        well, pos = sample.split("_")
        well_id = well_ids[well]
        pos = int(pos)

        well_y = 0
        pos_y = pos // 3
        y = well_y + pos_y * im_h + pos_y * im_spacing

        well_x = well_id * 3 * im_w + well_spacing * well_id
        pos_x = pos % 3
        x = well_x + pos_x * im_w + pos_x * im_spacing

        positions[sample] = (y, x)
        scores[sample] = np.random.rand()

    return positions, {"score": scores}


def main():
    url = "https://owncloud.gwdg.de/index.php/s/eu8JMlUFZ82ccHT"
    parser = argparse.ArgumentParser(f"Example data is available at {url}")
    parser.add_argument("folder", help="Path to the folder with multi-well plate data.")
    args = parser.parse_args()

    image_channels = ["serum", "nuclei"]
    label_channels = ["segmentation/cells"]
    image_settings = {"serum": {"colormap": "green"}, "nuclei": {"colormap": "blue"}}

    folder = args.folder
    image_data = {name: load_channel(folder, name) for name in image_channels}
    label_data = {name: load_channel(folder, name) for name in label_channels}

    positions, scores = create_positions(list(image_data["serum"].keys()))
    htm.view_positional_images(image_data, positions, label_data, image_settings, sample_measurements=scores)


if __name__ == "__main__":
    main()
