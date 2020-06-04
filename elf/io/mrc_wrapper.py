from collections.abc import Mapping
try:
    import mrcfile
except ImportError:
    mrcfile = None


class MRCDataset:
    def __init__(self, data_object):
        self._data = data_object

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
        return self._data[key]

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        return {}


class MRCFile(Mapping):
    """ Wrapper for an mrc file
    """
    def __init__(self, path, mode='r'):
        self.path = path
        self.mode = mode
        if mrcfile is None:
            raise AttributeError("mrcfile is not available")
        try:
            self._f = mrcfile.memmap(self.path, self.mode)
        except ValueError:
            self._f = mrcfile.open(self.path, self.mode)

    def __getitem__(self, key):
        if key != 'data':
            raise KeyError(f"Could not find key {key}")
        return MRCDataset(self._f.data)

    def __iter__(self):
        yield 'data'

    def __len__(self):
        return 1

    def __contains__(self, name):
        return name == 'data'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._f.close()

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        return {}
