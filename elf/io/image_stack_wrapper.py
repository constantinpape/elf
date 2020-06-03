import os
from collections.abc import Mapping
from concurrent import futures
from glob import glob

import numpy as np
import imageio

try:
    import tifffile
except ImportError:
    tifffile = None

from ..util import normalize_index, squeeze_singletons


class ImageStackFile(Mapping):
    def __init__(self, path, mode='r'):
        self.path = path
        self.file_name = os.path.split(self.path)[1]

    def __getitem__(self, key):
        # key must be a valid pattern
        files = glob(os.path.join(self.path, key))
        if len(files) == 0:
            raise ValueError(f"Invalid file pattern {key}")
        if TifStackDataset.is_tif_dataset(files):
            return TifStackDataset(files, sort_files=True)
        else:
            return ImageStackDataset(files, sort_files=True)

    # this could be done more sophisticated to find more complex patterns
    def get_all_patterns(self):
        all_files = glob(os.path.join(self.path, '*'))
        extensions = list(set(os.path.splitext(ff)[1] for ff in all_files))
        patterns = ['*' + ext for ext in extensions]
        return patterns

    def __iter__(self):
        patterns = self.get_all_patterns()
        for pattern in patterns:
            yield patterns

    def __len__(self):
        counter = 0
        for _ in self:
            counter += 1
        return counter

    def __contains__(self, name):
        files = glob(os.path.join(self.path, name))
        return len(files) > 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        return {}


class ImageStackDataset:

    def get_im_shape_and_dtype(self, files):
        im0 = imageio.imread(files[0])
        assert im0.ndim == 2
        return im0.shape, im0.dtype

    def initialize(self, files, sort_files=True):
        if sort_files:
            files.sort()
        self.files = files

        # get the shapes and dtype
        n_slices = len(files)
        self.im_shape, dtype = self.get_im_shape_and_dtype(files)

        self._shape = (n_slices,) + self.im_shape
        self._chunks = (1,) + self.im_shape
        self._dtype = dtype
        self._ndim = 3
        self._size = np.prod(list(self._shape))

    @classmethod
    def from_pattern(cls, folder, pattern, n_threads=1):
        files = glob(os.path.join(folder, pattern))
        return cls(files, n_threads=n_threads, sort_files=True)

    def __init__(self, files, n_threads=1, sort_files=True):
        self.initialize(files, sort_files=sort_files)
        self.n_threads = n_threads

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return self._ndim

    @property
    def chunks(self):
        return self._chunks

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    def _load_roi(self, roi):
        # init data
        roi_shape = tuple(rr.stop - rr.start for rr in roi)
        data = np.zeros(roi_shape, dtype=self.dtype)

        z0 = roi[0].start
        im_roi = roi[1:]

        def _load_and_write_image(z):
            z_abs = z + z0
            im = imageio.imread(self.files[z_abs])
            assert im.shape == self.im_shape
            data[z] = im[im_roi]

        # load the slices and write them into the output data
        with futures.ThreadPoolExecutor(self.n_threads) as tp:
            tasks = [tp.submit(_load_and_write_image, z) for z in range(roi_shape[0])]
            [t.result() for t in tasks]

        return data

    def __getitem__(self, key):
        roi, to_squeeze = normalize_index(key, self.shape)
        return squeeze_singletons(self._load_roi(roi), to_squeeze)

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        return {}


class TifStackDataset(ImageStackDataset):
    tif_exts = ('.tif', '.tiff')

    @staticmethod
    def is_tif_dataset(files):
        f0 = files[0]
        ext = os.path.splitext(f0)[1]
        if tifffile is None:
            return False
        if ext.lower() not in TifStackDataset.tif_exts:
            return False
        try:
            tifffile.memmap(f0)
        except ValueError:
            return False
        return True

    def get_im_shape_and_dtype(self, files):
        im0 = tifffile.memmap(files[0], mode='r')
        im_shape = im0.shape
        im_shapes = [tifffile.memmap(ff, mode='r').shape for ff in files[1:]]
        if any(sh != im_shape for sh in im_shapes):
            raise ValueError("Incompatible shapes for Image Stack")
        return im_shape, im0.dtype

    def _load_roi(self, roi):
        # init data
        roi_shape = tuple(rr.stop - rr.start for rr in roi)
        data = np.zeros(roi_shape, dtype=self.dtype)

        z0 = roi[0].start
        im_roi = roi[1:]

        def _load_and_write_image(z):
            z_abs = z + z0
            im = tifffile.memmap(self.files[z_abs], mode='r')
            assert im.shape == self.im_shape
            data[z] = im[im_roi]

        # load the slices and write them into the output data
        with futures.ThreadPoolExecutor(self.n_threads) as tp:
            tasks = [tp.submit(_load_and_write_image, z) for z in range(roi_shape[0])]
            [t.result() for t in tasks]

        return data
