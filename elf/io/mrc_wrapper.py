import warnings
from collections.abc import Mapping

import numpy as np

try:
    import mrcfile
except ImportError:
    mrcfile = None


class MRCDataset:
    def __init__(self, data_object):
        im = data_object
        # need to swap and flip to meet axes conventions
        data0 = np.swapaxes(im, 0, -1)
        data1 = np.fliplr(data0)
        self._data = np.swapaxes(data1, 0, -1)

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
    """ Wrapper for an mrc file
    """

    def __init__(self, path, mode="r"):
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
