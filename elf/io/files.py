import os
from .extensions import FILE_CONSTRUCTORS


def open_file(path, mode='a'):
    ext = os.path.splitext(path)[1]
    try:
        constructor = FILE_CONSTRUCTORS[ext.lower()]
    except KeyError:
        raise ValueError("""Could not infer file type from extension %s.
                          You may need to install additional dependencies (h5py, z5py, zarr)""" % ext)
    return constructor(path, mode=mode)
