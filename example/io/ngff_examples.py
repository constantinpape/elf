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
        data = imageio.volread(path)
        data = data[bb]
    else:
        with open_file(path, 'r') as f:
            data = f[key][bb]
    return data


def _create_example(in_path, folder, axes, key=None, bb=np.s_[:], dimension_separator="/"):
    ax_name = ''.join(axes)
    if dimension_separator == "/":
        out_path = os.path.join(folder, f"{ax_name}.ome.zarr")
    else:
        out_path = os.path.join(folder, f"flat_{ax_name}.ome.zarr")
    if os.path.exists(out_path):
        print("Example data at", out_path, "is already present")
        return

    data = _load_data(in_path, key, bb)
    assert data.ndim == len(axes)
    kwargs = _kwargs_3d() if axes[-3:] == ('z', 'y', 'x') else _kwargs_2d()
    write_ome_zarr(data, out_path, axes, ax_name,
                   n_scales=3, kwargs=kwargs, dimension_separator=dimension_separator)

#
# create ngff ome.zarr example data
#
# all the filepath are hard-coded to the EMBL kreshuk group share


# axes: yx
def create_2d_example(folder):
    # yx: covid htm data with only nucleus channel
    in_path = os.path.join("/g/kreshuk/data/covid/covid-data-vibor/20200405_test_images",
                           "WellC01_PointC01_0000_ChannelDAPI,WF_GFP,TRITC,WF_Cy5_Seq0216.tiff")
    _create_example(in_path, folder, axes=("y", "x"), bb=np.s_[0, :, :])


def _make_t_volume(path, timepoints, out_path, scale=2):
    if os.path.exists(out_path):
        return
    data = []
    for tp in timepoints:
        key = f'setup0/timepoint{tp}/s{scale}'
        with open_file(path, 'r') as f:
            d = f[key][:]
        data.append(d[None])
    data = np.concatenate(data, axis=0)
    with open_file(out_path, 'w') as f:
        f.create_dataset('data', data=data, chunks=(1, 64, 64, 64))


# TODO add linked labels for the zyx example
# axes: zyx, cyx, tyx
def create_3d_examples(folder):

    # zyx: covid em data + labels
    raw_path = os.path.join('/g/kreshuk/pape/Work/mobie/covid-em-datasets/data',
                            'Covid19-S4-Area2/images/local/sbem-6dpf-1-whole-raw.n5')
    raw_key = 'setup0/timepoint0/s3'
    _create_example(raw_path, folder, axes=("z", "y", "x"), key=raw_key)
    # for linked labels
    # seg_path = os.path.join('/g/kreshuk/pape/Work/mobie/covid-em-datasets/data',
    #                         'Covid19-S4-Area2/images/local/s4_area2_segmentation.n5')
    # seg_key = 'setup0/timepoint0/s3'

    # cyx: covid htm data with more channels
    in_path = os.path.join("/g/kreshuk/data/covid/covid-data-vibor/20200405_test_images",
                           "WellC01_PointC01_0000_ChannelDAPI,WF_GFP,TRITC,WF_Cy5_Seq0216.tiff")
    _create_example(in_path, folder, axes=("c", "y", "x"), bb=np.s_[:3, :, :])

    # tyx: middle slice from arabidopsis dataset
    timepoints = [32, 33, 34]
    scale = 2
    raw_path = os.path.join('/g/kreshuk/pape/Work/mobie/arabidopsis-root-lm-datasets/data',
                            'arabidopsis-root/images/local/lm-membranes.n5')
    tmp_path = './tp_data.h5'
    _make_t_volume(raw_path, timepoints, tmp_path, scale=scale)
    _create_example(tmp_path, folder, axes=("t", "y", "x"), bb=np.s_[:, 200, :, :],
                    key='data')


def _make_tc_volume(path1, path2, timepoints, out_path, scale=2):
    if os.path.exists(out_path):
        return
    data = []
    for tp in timepoints:
        key = f'setup0/timepoint{tp}/s{scale}'
        with open_file(path1, 'r') as f:
            d1 = f[key][:]
        with open_file(path2, 'r') as f:
            d2 = f[key][:]
        d = np.concatenate([d1[None], d2[None]], axis=0)
        data.append(d[None])
    data = np.concatenate(data, axis=0)
    with open_file(out_path, 'w') as f:
        f.create_dataset('data', data=data, chunks=(1, 1, 64, 64, 64))


# axes: tzyx, tcyx, czyx
def create_4d_examples(folder):

    # tzyx: arabidopsis dataset (boundaries)
    tmp_path = './tp_data.h5'
    _create_example(tmp_path, folder, key="data", axes=("t", "z", "y", "x"), bb=np.s_[:])

    # tcyx: arabidopsis boundaries and nuclei, middle slice
    timepoints = [32, 33, 34]
    scale = 2
    path1 = '/g/kreshuk/pape/Work/mobie/arabidopsis-root-lm-datasets/data/arabidopsis-root/images/local/lm-membranes.n5'
    path2 = '/g/kreshuk/pape/Work/mobie/arabidopsis-root-lm-datasets/data/arabidopsis-root/images/local/lm-nuclei.n5'
    tmp_path = './tp_channel_data.h5'
    _make_tc_volume(path1, path2, timepoints, tmp_path, scale=scale)
    _create_example(tmp_path, folder, key="data", axes=("t", "c", "y", "x"), bb=np.s_[:, :, 200, :, :])

    # czyx: arabidopsis dataset (boundaries + nuclei), single timepoint
    _create_example(tmp_path, folder, key="data", axes=("c", "z", "y", "x"), bb=np.s_[0, :, :, :, :])


# axes: tczyx
def create_5d_example(folder):
    # tczyx: full arabidopsis dataset
    tmp_path = './tp_channel_data.h5'
    _create_example(tmp_path, folder, key="data", axes=("t", "c", "z", "y", "x"), bb=np.s_[:, :, :, :, :])


# using '.' dimension separator
def create_flat_example(folder):
    # yx: covid htm data with only nucleus channel
    in_path = os.path.join("/g/kreshuk/data/covid/covid-data-vibor/20200405_test_images",
                           "WellC01_PointC01_0000_ChannelDAPI,WF_GFP,TRITC,WF_Cy5_Seq0216.tiff")
    _create_example(in_path, folder, axes=("y", "x"), bb=np.s_[0, :, :], dimension_separator=".")


def copy_readme(output_folder, version):
    readme = f"""
# Example data for OME-ZARR NGFF v{version}

This folder contains the following example ome.zarr files
- yx: 2d image, data is the nucleus channel of an image from [1]
- zyx: 3d volume, data is an EM volume from [2]
- cyx: 2d image with channels, image with 3 channels from [1]
- tyx: timeseries of 2d images, timeseries of central slice of membrane channel from [3]
- tzyx: timeseries of 3d images, timeseries of membrane channel volume from [3]
- tcyx: timeseries of images with channel, timeseries of central slice of membrane + nucleus channel from [3]
- czyx: 3d volume with channel, single timepoint of membrane and nucleus channel from [3]
- tczyx: timeseries of 3d volumes with channel, full data from [3]
- flat_yx: same as yx, but using flat chunk storage (dimension_separator=".") instead of nested storage

Publications:
[1] https://onlinelibrary.wiley.com/doi/full/10.1002/bies.202000257
[2] https://www.sciencedirect.com/science/article/pii/S193131282030620X
[3] https://elifesciences.org/articles/57613
    """
    out_path = os.path.join(output_folder, 'Readme.md')
    with open(out_path, 'w') as f:
        f.write(readme)


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
    copy_readme(output_folder, args.version)
