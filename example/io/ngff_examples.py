import argparse
import os

import imageio
import numpy as np
from elf.io import open_file
from elf.io.ngff import write_ome_zarr


def _kwargs_2d():
    return {"scale": (0.5, 0.5), "order": 0, "preserve_range": True}


def _kwargs_3d():
    return {"scale": (0.5, 0.5, 0.5), "order": 0, "preserve_range": True}


def _load_data(path, key, bb):
    if key is None:
        data = imageio.imread(path)
        data = data[bb]
    else:
        with open_file(path, 'r') as f:
            data = f[key][bb]
    return data


def _create_example(in_path, folder, axes, key=None, bb=np.s_[:]):
    data = _load_data(in_path, key, bb)
    assert data.ndim == len(axes)
    ax_name = ''.join(axes)
    out_path = os.path.join(folder, f"{ax_name}.ome.zr")
    kwargs = _kwargs_3d() if axes[-3:] == ('z', 'y', 'x') else _kwargs_2d()
    write_ome_zarr(data, out_path, axes, ax_name,
                   n_scales=3, kwargs=kwargs)

#
# create ngff ome.zarr example data
#
# all the filepath are hard-coded to the EMBL kreshuk group share


# axes: yx
def create_2d_example(folder):
    in_path = os.path.join("/g/kreshuk/data/covid/covid-data-vibor/20200405_test_images",
                           "WellC01_PointC01_0000_ChannelDAPI,WF_GFP,TRITC,WF_Cy5_Seq0216.tiff")
    _create_example(in_path, folder, axes=("y", "x"), bb=np.s_[0, :, :])


# add linked labels for the zyx example
# axes: zyx, cyx, tyx
def create_3d_examples(folder):
    pass


# axes: tcyx, tzyx, czyx
def create_4d_examples(folder):
    pass


# axes: tczyx
def create_5d_example(folder):
    pass


# using '.' dimension separator
def create_flat_example(folder):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_root', required=True)
    parser.add_argument('--version', default="v0.3")
    args = parser.parse_args()

    output_folder = os.path.join(args.output_root, args.version)
    os.makedirs(output_folder, exist_ok=True)

    create_2d_example(output_folder)
    create_3d_examples(output_folder)
    create_4d_examples(output_folder)
    create_5d_example(output_folder)
    create_flat_example(output_folder)
    # TODO copy README
