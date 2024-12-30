import os
import warnings
from collections.abc import Mapping
from typing import Union

import numpy as np

try:
    import mrcfile
except ImportError:
    mrcfile = None


class MRCDataset:
    """Dataset object for a file handle representing a mrc file.
    """
    def __init__(self, data_object):
        # Need to flip the data's axis to meet axes conventions.
        self._data = np.flip(data_object, axis=1) if data_object.ndim == 3 else np.flip(data_object, axis=0)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    # TODO chunks are arbitrary, how do we handle this?
    @property
    def chunks(self):
        return None

    @property
    def shape(self):
        return self._data.shape

    def __getitem__(self, key):
        return self._data[key].copy()

    @property
    def size(self):
        return self._data.size

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        return {}


class MRCFile(Mapping):
    """Root object for a file handle representing a mrc file.

    Args:
        path: The filepath of the mrc file.
        mode: The mode for opening the folder, only supports 'r' (read mode).
    """
    def __init__(self, path: Union[os.PathLike, str], mode: str = "r"):
        self.path = path
        self.mode = mode
        if mrcfile is None:
            raise AttributeError("mrcfile is required to read mrc or rec files, but is not installed")
        try:
            self._f = mrcfile.mmap(self.path, self.mode)
        except ValueError as e:

            # check if error comes from old version of SerialEM used for acquisition
            if (
                "Unrecognised machine stamp: 0x44 0x00 0x00 0x00" in str(e) or
                "Unrecognised machine stamp: 0x00 0x00 0x00 0x00" in str(e)
            ):
                try:
                    self._f = mrcfile.mmap(self.path, self.mode, permissive="True")
                except ValueError:
                    self._f = mrcfile.open(self.path, self.mode, permissive="True")

            else:  # Other kind of error -> try to open without mmap.
                try:
                    self._f = mrcfile.open(self.path, self.mode)
                except ValueError as e:
                    self._f = mrcfile.open(self.path, self.mode, permissive="True")
                    warnings.warn(
                        f"Opening mrcfile {self.path} failed with unknown error {e} without permissive opening."
                        "The file will still be opened but the contents may be incorrect."
                    )

    def __getitem__(self, key):
        if key != "data":
            raise KeyError(f"Could not find key {key}")
        return MRCDataset(self._f.data)

    def __iter__(self):
        yield "data"

    def __len__(self):
        return 1

    def __contains__(self, name):
        return name == "data"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._f.close()

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        return {}
