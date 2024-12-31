import os
from collections.abc import Mapping
from concurrent import futures
from typing import Union

import numpy as np
import imageio.v3 as imageio
from ..util import normalize_index, squeeze_singletons, map_chunk_to_roi, chunks_overlapping_roi


class KnossosDataset:
    """Dataset object for a file handle representing a knossos dataset.
    """
    block_size = 128

    @staticmethod
    def _chunks_dim(dim_root):
        files = os.listdir(dim_root)
        files = [f for f in files if os.path.isdir(os.path.join(dim_root, f))]
        return len(files)

    def get_shape_and_grid(self):
        """@private
        """
        cx = self._chunks_dim(self.path)
        y_root = os.path.join(self.path, "x0000")
        cy = self._chunks_dim(y_root)
        z_root = os.path.join(y_root, "y0000")
        cz = self._chunks_dim(z_root)

        grid = (cz, cy, cx)
        shape = tuple(sh * self.block_size for sh in grid)
        return shape, grid

    def __init__(self, path, file_prefix, load_png):
        self.path = path
        self.ext = "png" if load_png else "jpg"
        self.file_prefix = file_prefix

        self._ndim = 3
        self._chunks = self._ndim * (self.block_size,)
        self._shape, self._grid = self.get_shape_and_grid()
        self.n_threads = 1

    @property
    def dtype(self):
        return np.dtype("uint8")

    @property
    def ndim(self):
        return self._ndim

    @property
    def chunks(self):
        return self._chunks

    @property
    def shape(self):
        return self._shape

    def load_block(self, grid_id):
        # NOTE need to reverse grid id, because knossos folders are stored in x, y, z order
        block_path = ["%s%04i" % (dim, gid) for dim, gid in zip(("x", "y", "z"), grid_id[::-1])]
        dim_str = "_".join(block_path)
        fname = "%s_%s.%s" % (self.file_prefix, dim_str, self.ext)
        block_path.append(fname)
        path = os.path.join(self.path, *block_path)
        data = np.array(imageio.imread(path)).reshape(self._chunks)
        return data

    def _load_roi(self, roi):
        # snap roi to grid
        grid_points = chunks_overlapping_roi(roi, self.chunks)

        # init data (dtype is hard-coded to uint8)
        roi_shape = tuple(rr.stop - rr.start for rr in roi)
        data = np.zeros(roi_shape, dtype="uint8")

        def load_tile(grid_id):
            tile_data = self.load_block(grid_id)
            tile_bb, out_bb = map_chunk_to_roi(grid_id, roi, self.chunks)
            data[out_bb] = tile_data[tile_bb]

        if self.n_threads > 1:
            with futures.ThreadPoolExecutor(self.n_threads) as tp:
                tasks = [tp.submit(load_tile, sp) for sp in grid_points]
                [t.result() for t in tasks]
        else:
            [load_tile(sp) for sp in grid_points]
        return data

    def __getitem__(self, key):
        roi, to_squeeze = normalize_index(key, self.shape)
        return squeeze_singletons(self._load_roi(roi), to_squeeze)

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        return {}

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]


class KnossosFile(Mapping):
    """Root object for a file handle representing a knossos dataset.

    Args:
        path: Filepath to the knossos dataset.
        mode: The mode for opening the folder, only supports 'r' (read mode).
    """
    def __init__(self, path: Union[os.PathLike, str], mode: str = "r", load_png: bool = True):
        if not os.path.exists(os.path.join(path, "mag1")):
            raise RuntimeError("Not a knossos file structure")
        self.path = path
        self.load_png = load_png
        self.file_name = os.path.split(self.path)[1]

    def __getitem__(self, key):
        sub_path = os.path.join(self.path, key)
        if not os.path.exists(sub_path):
            raise ValueError("Key %s does not exist" % key)
        if not os.path.isdir(sub_path) and key.startswith("mag"):
            raise ValueError("Key %s is not a valid knossos dataset" % key)
        file_prefix = "%s_%s" % (self.file_name, key)
        return KnossosDataset(sub_path, file_prefix, self.load_png)

    def __iter__(self):
        for name in os.listdir(self.path):
            if os.path.isdir(os.path.join(self.path, name)) and name.startswith("mag"):
                yield name

    def __len__(self):
        counter = 0
        for _ in self:
            counter += 1
        return counter

    def __contains__(self, name):
        return super().__contains__(name.lstrip("/"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        return {}
