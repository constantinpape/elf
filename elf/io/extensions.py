import functools

import numpy as np

from .image_stack_wrapper import ImageStackFile, ImageStackDataset
from .knossos_wrapper import KnossosFile, KnossosDataset
from .mrc_wrapper import MRCFile, MRCDataset
from .nifti_wrapper import NiftiFile, NiftiDataset
from .intern_wrapper import InternFile, InternDataset


__all__ = [
    "FILE_CONSTRUCTORS", "GROUP_LIKE", "DATASET_LIKE",
    "h5py", "z5py", "pyn5", "zarr", "zarr_open",
]

FILE_CONSTRUCTORS = {}
"""@private
"""
ZARR_EXTS = [".zarr", ".zr"]
"""@private
"""
N5_EXTS = [".n5"]
"""@private
"""

GROUP_LIKE = []
"""@private
"""
DATASET_LIKE = [np.ndarray]
"""@private
"""


def _ensure_iterable(item):
    """Ensure item is a non-string iterable (wrap in a list if not)."""
    try:
        len(item)
        has_len = True
    except TypeError:
        has_len = False

    if isinstance(item, str) or not has_len:
        return [item]
    return item


def register_filetype(constructor, extensions=(), groups=(), datasets=(), overwrite=False):
    """@private
    """
    extensions = _ensure_iterable(extensions)
    FILE_CONSTRUCTORS.update({
        ext.lower(): constructor
        for ext in _ensure_iterable(extensions)
        if ext not in FILE_CONSTRUCTORS or overwrite
    })
    GROUP_LIKE.extend(_ensure_iterable(groups))
    DATASET_LIKE.extend(_ensure_iterable(datasets))


# add hdf5 extensions if we have h5py
try:
    import h5py
    register_filetype(h5py.File, [".h5", ".hdf", ".hdf5"], h5py.Group, h5py.Dataset)
except ImportError:
    h5py = None

# add n5 and zarr extensions if we have z5py
try:
    import z5py
    register_filetype(z5py.File, N5_EXTS + ZARR_EXTS, z5py.Group, z5py.Dataset)
except ImportError:
    z5py = None

try:
    # will not override z5py
    import pyn5
    register_filetype(pyn5.File, N5_EXTS, pyn5.Group, pyn5.Dataset)
except ImportError:
    pyn5 = None

# add mrc extensions if we have mrcfile
try:
    import mrcfile
    register_filetype(MRCFile, [".mrc", ".rec"], MRCFile, MRCDataset)
except ImportError:
    mrcfile = None

# add bossdb extensions if we have intern
try:
    import intern
    register_filetype(InternFile, ["bossdb://"], InternFile, InternDataset)
except ImportError:
    intern = None

# add nifti extensions if we have nibabel
try:
    import nibabel
    register_filetype(NiftiFile, [".nii.gz", ".nii"], NiftiFile, NiftiDataset)
except ImportError:
    nibabel = None


def identity(arg):
    """@private
    """
    return arg


def noop(*args, **kwargs):
    """@private
    """
    pass


try:
    # will not override z5py
    import zarr

    # zarr stores cannot be used as context managers,
    # which breaks compatibility with similar libraries.
    # This wrapper patches in those methods.
    @functools.wraps(zarr.open)
    def zarr_open(*args, **kwargs):
        """@private
        """
        z = zarr.open(*args, **kwargs)
        ztype = type(z)
        if not hasattr(ztype, "__enter__"):
            ztype.__enter__ = identity
        if not hasattr(ztype, "__exit__"):
            ztype.__exit__ = noop
        return z

    register_filetype(
        zarr_open, N5_EXTS + ZARR_EXTS, zarr.hierarchy.Group, zarr.core.Array, True
    )
except ImportError:
    zarr = None
    zarr_open = None


def folder_based(path, mode="a"):
    """@private
    """
    try:
        return KnossosFile(path, mode)
    except RuntimeError:
        return ImageStackFile(path, mode)


# Are there any typical knossos extensions?
# add folder based wrappers (no extension)
register_filetype(folder_based, ["", ".tif", ".tiff"],
                  (ImageStackFile, KnossosFile),
                  (ImageStackDataset, KnossosDataset))
