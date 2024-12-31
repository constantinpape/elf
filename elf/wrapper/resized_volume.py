from typing import Tuple

import numpy as np
import vigra
from numpy.typing import ArrayLike
from skimage.transform import resize

from .base import WrapperBase
from ..util import normalize_index, squeeze_singletons


# TODO
# - check if we can use skimage.transform.resize instead of vigra
# - smooth after resize (for order > 0) to avoid aliasing (skimage has this built-in)
# - implement loading with halo for sub-slices to avoid boundary artifacts
# - support more dimensions and multichannel
class ResizedVolume(WrapperBase):
    """Wrapper to resize a volume on the fly.

    Args:
        volume: The data to wrap.
        shape: The target shape for resizing.
        order: The interpolation order to use.
    """
    def __init__(self, volume: ArrayLike, shape: Tuple[int, ...], order: int = 0):
        if len(shape) != volume.ndim:
            raise ValueError(f"Expect volume and shape to have same dimensionality, got {len(shape)}, {volume.ndim}")
        if volume.ndim not in (2, 3):
            raise ValueError(f"Expect 2d or 3d input data, got {volume.ndim}")
        super().__init__(volume)
        self._shape = shape

        self._scale = [sh / float(fsh) for sh, fsh in zip(self.volume.shape, self.shape)]
        self.order = order

        if np.dtype(self.dtype) == np.dtype(bool):
            self.min, self.max = 0, 1
        else:
            try:
                self.min = np.iinfo(np.dtype(self.dtype)).min
                self.max = np.iinfo(np.dtype(self.dtype)).max
            except ValueError:
                self.min = np.finfo(np.dtype(self.dtype)).min
                self.max = np.finfo(np.dtype(self.dtype)).max

    @property
    def shape(self):
        return self._shape

    @property
    def scale(self):
        return self._scale

    def _interpolate_vigra(self, data, shape):
        data = vigra.sampling.resize(data.astype("float32"), shape=shape, order=self.order)
        np.clip(data, self.min, self.max, out=data)
        return data.astype(self.dtype)

    def _interpolate_skimage(self, data, shape):
        if self.order > 0:
            data = resize(data, shape, order=self.order, preserve_range=True)
        else:
            data = resize(data, shape, order=self.order, anti_aliasing=False, preserve_range=True)
        return data.astype(self.dtype)

    def _interpolate(self, data, shape):
        # vigra can't deal with singletons, so we use skimage in that case, but stil use
        # vigra otherwise due to better performance
        singletons = tuple(sh == 1 for sh in data.shape)
        if any(singletons):
            data = self._interpolate_skimage(data, shape)
        else:
            data = self._interpolate_vigra(data, shape)
        return data

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)

        # get the return shape and find singleton axes
        ret_shape = tuple(ind.stop - ind.start for ind in index)
        singletons = tuple(sh == 1 for sh in ret_shape)

        # get the sampled index, respecting singletons
        starts = tuple(int(round(ind.start * sc, 0)) for ind, sc in zip(index, self.scale))
        stops = tuple(max(int(round(ind.stop * sc, 0)), sta + 1)
                      for ind, sc, sta in zip(index, self.scale, starts))
        index = tuple(slice(sta, sto) for sta, sto in zip(starts, stops))

        # check if we have a singleton in the return shape
        data_shape = tuple(idx.stop - idx.start for idx in index)
        # remove singletons from data iff axis is not singleton in return data
        index = tuple(slice(idx.start, idx.stop) if sh > 1 or is_single else
                      slice(idx.start, idx.stop + 1)
                      for idx, sh, is_single in zip(index, data_shape, singletons))
        data = self.volume[index]

        # Speed ups for empty blocks and masks.
        if data.sum() == 0:
            out = np.zeros(ret_shape, dtype=self.dtype)
        elif (data == 1).sum() == data.size:
            out = np.ones(ret_shape, dtype=self.dtype)
        else:
            out = self._interpolate(data, ret_shape)
        return squeeze_singletons(out, to_squeeze)
