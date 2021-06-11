import argparse
import os
from glob import glob

import napari
import zarr


def check_example(ff):
    fname = os.path.split(ff)[1]
    print("Checking", fname)
    if fname.startswith('flat'):
        store = zarr.storage.DirectoryStore(ff)
    else:
        store = zarr.storage.NestedDirectoryStore(ff)

    with zarr.open(store, mode='r') as f:
        data = f['s0'][:]
        print(data.shape)

    with napari.gui_qt():
        v = napari.Viewer()
        v.title = fname
        v.add_image(data)


def check_examples(folder):
    files = glob(os.path.join(folder, '*.ome.zarr'))
    for ff in files:
        check_example(ff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    args = parser.parse_args()
    check_examples(args.folder)
