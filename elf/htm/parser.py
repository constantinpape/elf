import os
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import elf.io as io
from tqdm import tqdm

#
# Parser functions for htm layout standards.
# Outputs can be passed directly to htm.visualization.view_plate.
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
def parse_simple_htm(
    folder: str,
    pattern: str = "*.h5",
    exclude_names: Optional[Sequence[str]] = None,
) -> Tuple[
    Dict[str, Dict[str, List[np.ndarray]]],
    Dict[str, Dict[str, List[np.ndarray]]],
]:
    """Parse simple high throughput / high content microscopy layout.

    You can obtained sample data from this layout at:
    https://owncloud.gwdg.de/index.php/s/eu8JMlUFZ82ccHT

    Args:
        folder: The root folder with the data.
        pattern: The pattern for selecting files.
        exclude_names: Filenames to exclude from loading.

    Returns:
        A map of channel and well names to the corresponding image data.
        A map of label names and well names to the corresponding segmentation data.
    """
    files = sorted(glob(os.path.join(folder, pattern)))

    # Get the channel and label names.
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


# More formats we could support.
# def parse_batchlib():
#     pass
#
#
# def parse_mobie():
#     pass
#
#
# def parse_incucyte():
#     pass
