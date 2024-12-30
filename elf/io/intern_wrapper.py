from collections.abc import Mapping
import numpy as np

try:
    from intern import array

    intern_imported = True
except ImportError:
    intern_imported = False


def _check_intern_importable():
    if not intern_imported:
        raise ImportError(
            "Could not import the `intern` library. This means you cannot "
            "download or upload cloud datasets. To fix this, you can install "
            "intern with: \n\n\t"
            "pip install intern"
        )
    return True


class InternDataset:
    """Dataset object for a handle representing an intern resource.
    """
    def __init__(self, cloud_path):
        _check_intern_importable()
        self._data = array(cloud_path)

    @property
    def dtype(self):
        return np.dtype(self._data.dtype)

    @property
    def ndim(self):
        return 3  # todo: this COULD be 4 etc...

    # TODO chunks are arbitrary, how do we handle this?
    @property
    def chunks(self):
        return None

    @property
    def shape(self):
        return self._data.shape

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    @property
    def size(self):
        shape = self._data.shape
        return shape[0] * shape[1] * shape[2]

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        return {}


class InternFile(Mapping):
    """Root object for a handle representing an intern resource.

    Args:
        path: The URL of the intern resource to open.
        mode: The mode for opening the resource. Only 'r' (read mode) is supported.
    """

    def __init__(self, path: str, mode: str = "r"):
        _check_intern_importable()
        self.path = path
        self.mode = mode

    def __getitem__(self, key):
        return InternDataset(self.path)

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
