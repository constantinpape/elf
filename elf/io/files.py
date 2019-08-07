import os
from .extensions import FILE_CONSTRUCTORS
from .extensions import h5py, z5py


def supported_extensions():
    """ Returns a list with the supported file extensions.
    """
    return list(FILE_CONSTRUCTORS.keys())


# TODO support pathlib path
def open_file(path, mode='a'):
    """ Open a hdf5, zarr, n5 or knossos file on the filesystem.

    The formats and extensions supported depend on the available libraries.
    Check for the supported extensions by calling `elf.io.supported_extensions`.

    Arguments:
        path [str] - path to the file to be opened
        mode [str] - mode in which to open the file (default: 'a')
    """
    ext = os.path.splitext(path)[1]
    try:
        constructor = FILE_CONSTRUCTORS[ext.lower()]
    except KeyError:
        raise ValueError("""Could not infer file type from extension %s.
                          You may need to install additional dependencies (h5py, z5py, zarr)""" % ext)
    return constructor(path, mode=mode)


# TODO group and dataset checks for zarr-python
def is_group(node):
    """ Check if argument is an h5py or z5py group
    """
    if h5py and isinstance(node, h5py.Group):
        return True
    if z5py and isinstance(node, z5py.Group):
        return True
    return False


def is_dataset(node):
    """ Check if argument is an h5py or z5py dataset
    """
    if h5py and isinstance(node, h5py.Dataset):
        return True
    if z5py and isinstance(node, z5py.Dataset):
        return True
    return False
