import os
from pathlib import Path
from typing import List, Optional, Union

from .extensions import (
    FILE_CONSTRUCTORS, GROUP_LIKE, DATASET_LIKE,
    h5py, z5py, pyn5, zarr,
)
from .knossos_wrapper import KnossosFile, KnossosDataset
from .mrc_wrapper import MRCFile, MRCDataset
from .intern_wrapper import InternFile, InternDataset


def supported_extensions() -> List[str]:
    """Returns a list with the supported file extensions.

    Returns:
        List iwth supported file extensions.
    """
    return list(FILE_CONSTRUCTORS.keys())


def open_file(
    path: Union[str, os.PathLike],
    mode: str = "a",
    ext: Optional[str] = None,
    **kwargs,
):
    """Open a hdf5, zarr, n5, knossos or other filetype supported by elf.

    The formats and file extensions that are supported depend on the available libraries.
    Check for the supported extensions by calling `elf.io.supported_extensions`.

    Args:
        path: Path to the file to be opened.
        mode: Mode in which to open the file.
        ext: File extension. This can be used to force an extension if it cannot be inferred from the filename.

    Returns:
        The handle for the opened datsaet.
    """

    # Before checking the extension suffix, check for "protocol-style" cloud provider prefixes.
    if isinstance(path, str) and "://" in path:
        ext = path.split("://")[0] + "://"

    elif ext is None:
        path_ = Path(path.rstrip("/")) if isinstance(path, str) else path
        suffixes = path_.suffixes
        # We need to treat .nii.gz differently.
        if len(suffixes) == 2 and "".join(suffixes) == ".nii.gz":
            ext = ".nii.gz"
        elif len(suffixes) == 0:
            ext = ""
        else:
            ext = suffixes[-1]

    try:
        constructor = FILE_CONSTRUCTORS[ext.lower()]
    except KeyError:
        raise ValueError(
            f"Could not infer file type from extension {ext}, "
            f"because it is not in the supported extensions: "
            f"{' '.join(supported_extensions())}. "
            f"You may need to install additional dependencies (h5py, z5py, zarr, intern)."
        )

    return constructor(path, mode=mode, **kwargs)


def is_group(node) -> bool:
    """Check if the argument is an hdf5/n5/zarr group.
    """
    return isinstance(node, tuple(GROUP_LIKE))


def is_dataset(node) -> bool:
    """Check if the argument is an hdf5/n5/zarr dataset.
    """
    return isinstance(node, tuple(DATASET_LIKE))


def is_z5py(node) -> bool:
    """Check if the argument is a z5py object.
    """
    return z5py and isinstance(node, (z5py.Dataset, z5py.Group))


def is_h5py(node) -> bool:
    """Check if the argument is a h5py object.
    """
    return h5py and isinstance(node, (h5py.Dataset, h5py.Group))


def is_zarr(node) -> bool:
    """Check if the argument is a zarr object.
    """
    return zarr and isinstance(node, (zarr.core.Array, zarr.hierarchy.Group))


def is_pyn5(node) -> bool:
    """Check if the argument is a pyn5 object.
    """
    return pyn5 and isinstance(node, (pyn5.Dataset, pyn5.Group))


def is_knossos(node) -> bool:
    """Check if the argument is a KnossosWrapper object.
    """
    return isinstance(node, (KnossosFile, KnossosDataset))


def is_mrc(node) -> bool:
    """Check if the argument is a MRCWrapper object.
    """
    return isinstance(node, (MRCFile, MRCDataset))


def is_intern(node) -> bool:
    """Check if the argument is a Intern wrapper object.
    """
    return isinstance(node, (InternFile, InternDataset))
