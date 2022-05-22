import os
from glob import glob

import elf.io as io
from tqdm import tqdm

#
# parser functions for htm layout standards
# outputs can be parsed directly to htm.visualization.view_plastr
#


def _load_channel_simple(files, channel_name):
    data = {}
    for ff in tqdm(files, desc=f"Load {channel_name}"):
        well = os.path.basename(ff).split("_")[0]
        well_images = data.get(well, [])
        with io.open_file(ff, "r") as f:
            if channel_name not in f:
                continue
            im = f[channel_name][:]
        well_images.append(im)
        data[well] = well_images
    return data


# TODO enable lazy loading (return datasets instead of loaded data)
# TODO add support for image pyramid data
def parse_simple_htm(folder, pattern="*.h5", exclude_names=None):
    """Parse simple htm layout, see e.g. example data at
    https://owncloud.gwdg.de/index.php/s/eu8JMlUFZ82ccHT
    """
    files = glob(os.path.join(folder, pattern))
    files.sort()

    # get the channel and label names
    channel_names = []
    label_names = []
    with io.open_file(files[0], "r") as f:
        for name, obj in f.items():
            if exclude_names is not None and name in exclude_names:
                continue
            if io.is_dataset(obj):
                channel_names.append(name)
            elif io.is_group(obj) and name == "segmentation":
                for label_name, label in obj.items():
                    if exclude_names is not None and label_name in exclude_names:
                        continue
                    if io.is_dataset(label):
                        label_names.append(f"segmentation/{label_name}")

    assert channel_names
    image_data = {name: _load_channel_simple(files, name) for name in channel_names}
    label_data = None if label_names is None else {name: _load_channel_simple(files, name) for name in label_names}
    return image_data, label_data


# TODO
def parse_batchlib():
    pass


# TODO
def parse_mobie():
    pass


# TODO
def parse_incucyte():
    pass
