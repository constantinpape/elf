#
# use elf.htm to visualize a multi-well plate using napari
# the example data is available at https://owncloud.gwdg.de/index.php/s/eu8JMlUFZ82ccHT
#

import argparse
import os
from glob import glob

import elf.htm as htm
import h5py
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


def main():
    url = "https://owncloud.gwdg.de/index.php/s/eu8JMlUFZ82ccHT"
    parser = argparse.ArgumentParser(f"Uses a custom hdf5 multi-well layout, example data is available at {url}")
    parser.add_argument("folder", help="Path to the folder with multi-well plate data.")
    args = parser.parse_args()

    image_channels = ["serum", "nuclei", "marker"]
    label_channels = ["segmentation/cells", "segmentation/nuclei"]
    image_settings = {"serum": {"colormap": "green"}, "nuclei": {"colormap": "blue"}, "marker": {"colormap": "red"}}

    folder = args.folder
    image_data = {name: load_channel(folder, name) for name in image_channels}
    label_data = {name: load_channel(folder, name) for name in label_channels}

    htm.view_plate(image_data, label_data, image_settings, zero_based=False)


if __name__ == "__main__":
    main()
