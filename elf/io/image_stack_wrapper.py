import os
from collections.abc import Mapping
from concurrent import futures
from glob import glob

import numpy as np
import imageio

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
        return ImageStackDataset(files, sort_files=True)

    # this could be done more sophisticated to find mor complex patterns
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


class ImageStackDataset:

    def initialize(self, files, sort_files=True):
        if sort_files:
            files.sort()
        self.files = files

        # get the shapes and dtype
        n_slices = len(files)
        im0 = imageio.imread(files[0])
        assert im0.ndim == 2

        self.im_shape = im0.shape
        self._shape = (n_slices,) + self.im_shape
        self._chunks = (1,) + self.im_shape
        self._dtype = im0.dtype
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