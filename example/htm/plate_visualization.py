#
# use elf.htm to visualize a multi-well plate using napari
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
        well = os.path.basename(ff).split("_")[0]
        well_images = data.get(well, [])
        with h5py.File(ff, "r") as f:
            im = f[channel_name][:]
        well_images.append(im)
        data[well] = well_images
    return data


# TODO check that passing h5 dataset works and if it makes loading faster
# TODO check if passing pyramidial images works (probably need to change something to get image shapes)
def main():
    url = "https://owncloud.gwdg.de/index.php/s/eu8JMlUFZ82ccHT"
    parser = argparse.ArgumentParser(f"Uses a custom hdf5 multi-well layout, example data is available at {url}")
    parser.add_argument("folder", help="Path to the folder with multi-well plate data.")
    args = parser.parse_args()

    image_channels = ["serum", "nuclei", "marker"]
    label_channels = ["segmentation/cells", "segmentation/nuclei"]
    image_settings = {
        "serum": {"colormap": "green", "blending": "additive"},
        "nuclei": {"colormap": "blue", "blending": "additive"},
        "marker": {"colormap": "red", "blending": "additive"}
    }

    folder = args.folder
    image_data = {name: load_channel(folder, name) for name in image_channels}
    label_data = {name: load_channel(folder, name) for name in label_channels}

    # create toy data for scores to show as additional measurement with the wells
    well_names = list(image_data["serum"].keys())
    scores = {"score": {well_name: np.random.rand() for well_name in well_names}}
    htm.view_plate(image_data, label_data, image_settings, zero_based=False, well_measurements=scores)


if __name__ == "__main__":
    main()
