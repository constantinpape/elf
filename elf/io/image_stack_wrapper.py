import os
from collections.abc import Mapping
from concurrent import futures
from glob import glob
from typing import Union

import numpy as np
import imageio.v3 as imageio

try:
    import tifffile
except ImportError:
    tifffile = None

from ..util import normalize_index, squeeze_singletons


class ImageStackFile(Mapping):
    """Root object for a file handle representing multiple image files in a folder.

    Args:
        path: The filepath to the folder.
        mode: The mode for opening the folder, only supports 'r' (read mode).
    """
    def __init__(self, path: Union[os.PathLike, str], mode: str = "r"):
        self.path = path
        self.file_name = os.path.split(self.path)[1]

    def __getitem__(self, key):
        # if the key is empty, we assume to have an image stack (=3d image volume in one file)
        if key == "":
            if not os.path.isfile(self.path):
                raise ValueError(f"{self.path} needs to be a file to be loaded as image stack")

            if TifStackDataset.is_tif_stack(self.path):
                return TifStackDataset.from_stack(self.path)
            else:
                return ImageStackDataset.from_stack(self.path)

        # key must be a valid pattern
        pattern = os.path.join(self.path, key)
        files = glob(pattern)
        if len(files) == 0:
            raise ValueError(f"Invalid file pattern {pattern}")
        if TifStackDataset.is_tif_slices(files):
            return TifStackDataset(files, sort_files=True)
        else:
            return ImageStackDataset(files, sort_files=True)

    # this could be done more sophisticated to find more complex patterns
    def get_all_patterns(self):
        """@private
        """
        all_files = glob(os.path.join(self.path, "*"))
        extensions = list(set(os.path.splitext(ff)[1] for ff in all_files))
        patterns = ["*" + ext for ext in extensions]
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
        # if the key is empty, we assume to have an image stack (=3d image volume in one file)
        if name == "":
            return os.path.isfile(self.path)
        files = glob(os.path.join(self.path, name))
        return len(files) > 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        """@private
        """
        return {}


class ImageStackDataset:
    """Dataset object for a file handle representing multiple image files in a folder.
    """

    def get_im_shape_and_dtype(self, files):
        """@private
        """
        im0 = imageio.imread(files[0])
        assert im0.ndim == 2
        return im0.shape, im0.dtype

    def initialize_from_slices(self, files, sort_files=True):
        """@private
        """
        if sort_files:
            files.sort()
        self.files = files

        # get the shapes and dtype
        n_slices = len(files)
        self.im_shape, dtype = self.get_im_shape_and_dtype(files)

        self._shape = (n_slices,) + self.im_shape
        self._chunks = (1,) + self.im_shape
        self._dtype = dtype
        self._size = np.prod(list(self._shape))

    def initialize_from_stack(self, files):
        """@private
        """
        self.files = files
        self._volume = self._read_volume()

        self._shape = self._volume.shape
        # chunks are arbitrary
        self._chunks = None
        self._dtype = self._volume.dtype
        self._size = np.prod(list(self._shape))

    @classmethod
    def from_pattern(cls, folder, pattern, n_threads=1):
        """@private
        """
        files = glob(os.path.join(folder, pattern))
        return cls(files, n_threads=n_threads, sort_files=True)

    @classmethod
    def from_stack(cls, stack_path, n_threads=1):
        """@private
        """
        return cls(stack_path, n_threads=n_threads, is_stack=True)

    def __init__(self, files, n_threads=1, sort_files=True, is_stack=False):
        if is_stack:
            self.initialize_from_stack(files)
        else:
            self.initialize_from_slices(files, sort_files=sort_files)
        self.is_stack = is_stack
        self.n_threads = n_threads

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def chunks(self):
        return self._chunks

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    def _read_image(self, index):
        return imageio.imread(self.files[index])

    def _read_volume(self):
        return imageio.imread(self.files)

    def _load_roi_from_stack(self, roi):
        return self._volume[roi]

    def _load_roi_from_slices(self, roi):
        # init data
        roi_shape = tuple(rr.stop - rr.start for rr in roi)
        data = np.zeros(roi_shape, dtype=self.dtype)

        z0 = roi[0].start
        im_roi = roi[1:]

        def _load_and_write_image(z):
            z_abs = z + z0
            im = self._read_image(z_abs)
            assert im.shape == self.im_shape, f"{im.shape}, {self.im_shape}"
            data[z] = im[im_roi]

        # load the slices and write them into the output data
        with futures.ThreadPoolExecutor(self.n_threads) as tp:
            tasks = [tp.submit(_load_and_write_image, z) for z in range(roi_shape[0])]
            [t.result() for t in tasks]

        return data

    def __getitem__(self, key):
        roi, to_squeeze = normalize_index(key, self.shape)
        if self.is_stack:
            data = self._load_roi_from_stack(roi)
        else:
            data = self._load_roi_from_slices(roi)
        return squeeze_singletons(data, to_squeeze)

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        return {}

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]


class TifStackDataset(ImageStackDataset):
    """Dataset object for a file handle representing multiple tif files in a folder.
    """
    tif_exts = (".tif", ".tiff")

    @staticmethod
    def is_tif_slices(files):
        if tifffile is None:
            return False

        f0 = files[0]
        ext = os.path.splitext(f0)[1]
        if ext.lower() not in TifStackDataset.tif_exts:
            return False
        try:
            for ff in files:
                tifffile.memmap(ff, mode="r")
        except ValueError:
            return False
        return True

    @staticmethod
    def is_tif_stack(path):
        if tifffile is None:
            return False
        ext = os.path.splitext(path)[1]
        if ext.lower() not in TifStackDataset.tif_exts:
            return False
        try:
            tifffile.memmap(path, mode="r")
        except ValueError:
            return False
        return True

    def _read_image(self, index):
        return tifffile.memmap(self.files[index], mode="r")

    def _read_volume(self):
        return tifffile.memmap(self.files, mode="r")

    def get_im_shape_and_dtype(self, files):
        im0 = tifffile.memmap(files[0], mode="r")
        im_shape = im0.shape
        im_shapes = [tifffile.memmap(ff, mode="r").shape for ff in files[1:]]
        if any(sh != im_shape for sh in im_shapes):
            raise ValueError("Incompatible shapes for Image Stack")
        return im_shape, im0.dtype
