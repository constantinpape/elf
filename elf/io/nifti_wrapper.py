from collections.abc import Mapping
from ..util import normalize_index, squeeze_singletons

import numpy as np
try:
    import nibabel
except ImportError:
    nibabel = None


class NiftiFile(Mapping):
    def __init__(self, path, mode="r"):
        if nibabel is None:
            raise AttributeError("nibabel is required for nifti images, but is not installed.")
        self.path = path
        self.mode = mode
        self.nifti = nibabel.load(self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # dummy attrs to be compatible with h5py/z5py/zarr API
    # alternatively we could also map the header to attributes
    @property
    def attrs(self):
        return {}

    def __getitem__(self, key):
        if key != "data":
            raise KeyError(f"Could not find key {key}")
        return NiftiDataset(self.nifti)

    def __iter__(self):
        yield "data"

    def __len__(self):
        return 1

    def __contains__(self, name):
        return name == "data"


class NiftiDataset:
    def __init__(self, data):
        self._data = data

    @property
    def dtype(self):
        return self.data.get_data_dtype()

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def chunks(self):
        return None

    @property
    def shape(self):
        return self._data.shape[::-1]

    def __getitem__(self, key):
        key, to_squeeze = normalize_index(key, self.shape)
        transposed_key = key[::-1]
        data = self._data.dataobj[transposed_key].T
        return squeeze_singletons(data, to_squeeze)

    @property
    def size(self):
        return np.prod(self._data.shape)

    # dummy attrs to be compatible with h5py/z5py/zarr API
    # alternatively we could also map the header to attributes
    @property
    def attrs(self):
        return {}
