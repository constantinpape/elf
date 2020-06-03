import os
from .extensions import (
    FILE_CONSTRUCTORS, GROUP_LIKE, DATASET_LIKE,
    h5py, z5py, pyn5, zarr,
)
from .knossos_wrapper import KnossosFile, KnossosDataset


def supported_extensions():
    """ Returns a list with the supported file extensions.
    """
    return list(FILE_CONSTRUCTORS.keys())


# TODO support pathlib path
def open_file(path, mode='a', ext=None):
    """ Open a hdf5, zarr, n5 or knossos file on the filesystem.

    The formats and extensions supported depend on the available libraries.
    Check for the supported extensions by calling `elf.io.supported_extensions`.

    Arguments:
        path [str] - path to the file to be opened
        mode [str] - mode in which to open the file (default: 'a')
        ext [str] - file extension. This can be used to force an extension
            if it cannot be inferred from the filename. (default: None)
    """
    ext = os.path.splitext(path)[1] if ext is None else ext
    try:
        constructor = FILE_CONSTRUCTORS[ext.lower()]
    except KeyError:
        raise ValueError(
            f"Could not infer file type from extension {ext}, "
            f"because it is not in the supported extensions: "
            f"{' '.join(supported_extensions())}. "
            f"You may need to install additional dependencies (h5py, z5py, zarr)."
        )
    return constructor(path, mode=mode)


def is_group(node):
    """ Check if argument is an h5py or z5py group
    """
    return isinstance(node, tuple(GROUP_LIKE))


def is_dataset(node):
    """ Check if argument is an h5py or z5py dataset
    """
    return isinstance(node, tuple(DATASET_LIKE))


def is_z5py(node):
    """ Check if this is a z5py object
    """
    return z5py and isinstance(node, (z5py.Dataset, z5py.Group))


def is_h5py(node):
    """ Check if this is a h5py object
    """
    return h5py and isinstance(node, (h5py.Dataset, h5py.Group))


def is_zarr(node):
    return zarr and isinstance(node, (zarr.core.Array, zarr.hierarchy.Group))


def is_pyn5(node):
    return pyn5 and isinstance(node, (pyn5.Dataset, pyn5.Group))


def is_knossos(node):
    """ Check if this is a KnossosWrapper object
    """
    return isinstance(node, (KnossosFile, KnossosDataset))
