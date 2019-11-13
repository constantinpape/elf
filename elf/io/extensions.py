import numpy as np

from .knossos_wrapper import KnossosFile, KnossosDataset


__all__ = [
    "FILE_CONSTRUCTORS", "GROUP_LIKE", "DATASET_LIKE", "h5py", "z5py",
]

FILE_CONSTRUCTORS = {}
ZARR_EXTS = [".zarr", ".zr"]
N5_EXTS = [".n5"]

GROUP_LIKE = []
DATASET_LIKE = [np.ndarray]


def _ensure_iterable(item):
    """Ensure item is a non-string iterable (wrap in a list if not)"""
    try:
        len(item)
        has_len = True
    except TypeError:
        has_len = False

    if isinstance(item, str) or not has_len:
        return [item]
    return item


def register_filetype(constructor, extensions=(), groups=(), datasets=()):
    extensions = _ensure_iterable(extensions)
    FILE_CONSTRUCTORS.update({
        ext.lower(): constructor
        for ext in _ensure_iterable(extensions)
        if ext not in FILE_CONSTRUCTORS
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

# Are there any typical knossos extensions?
# add knossos (no extension)
register_filetype(KnossosFile, '', KnossosFile, KnossosDataset)
