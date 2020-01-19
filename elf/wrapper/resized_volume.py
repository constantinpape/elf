from functools import partial
from math import floor, ceil

import numpy as np
import vigra

from .base import WrapperBase
from ..util import normalize_index, squeeze_singletons


# TODO
# - check if we can use skimage.transform.resize instead of vigra
# - smooth after resize (for order > 0) to avoid aliasing (skimage has this built-in)
# - implement loading with halo for sub-slices to avoid boundary artifacts
# - support more dimensions and multichannel
class ResizedVolume(WrapperBase):
    """ Resized volume to a different shape.

    Arguments:
        volume [np.ndarray]: input volume
        shape [tuple]: target shape for interpolation
        order [int]: order used for interpolation
    """
    def __init__(self, volume, shape, order=0):
        assert len(shape) == volume.ndim == 3, "Only 3d supported"
        super().__init__(volume)
        self._shape = shape

        self._scale = [sh / float(fsh) for sh, fsh in zip(self.volume.shape, self.shape)]

        if np.dtype(self.dtype) == np.bool:
            self.min, self.max = 0, 1
        else:
            try:
                self.min = np.iinfo(np.dtype(self.dtype)).min
                self.max = np.iinfo(np.dtype(self.dtype)).max
            except ValueError:
                self.min = np.finfo(np.dtype(self.dtype)).min
                self.max = np.finfo(np.dtype(self.dtype)).max

        self.interpol_function = partial(vigra.sampling.resize, order=order)

    @property
    def shape(self):
        return self._shape

    @property
    def scale(self):
        return self._scale

    def _interpolate(self, data, shape):
        # vigra can't deal with singleton dimensions, so we need to handle this seperately
        have_squeezed = False
        # check for singleton axes
        singletons = tuple(sh == 1 for sh in data.shape)
        if any(singletons):
            assert all(sh == 1 for is_single, sh in zip(singletons, shape) if is_single)
            inflate = tuple(slice(None) if sh > 1 else None for sh in data.shape)
            data = data.squeeze()
            shape = tuple(sh for is_single, sh in zip(singletons, shape) if not is_single)
            have_squeezed = True

        data = self.interpol_function(data.astype('float32'), shape=shape)
        np.clip(data, self.min, self.max, out=data)

        if have_squeezed:
            data = data[inflate]
        return data.astype(self.dtype)

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
        # get the return shape and singletons
        ret_shape = tuple(ind.stop - ind.start for ind in index)
        singletons = tuple(sh == 1 for sh in ret_shape)

        # get the donwsampled index; respecting singletons
        starts = tuple(int(floor(ind.start * sc)) for ind, sc in zip(index, self.scale))
        stops = tuple(sta + 1 if is_single else int(ceil(ind.stop * sc))
                      for ind, sc, sta, is_single in zip(index, self.scale,
                                                         starts, singletons))
        index_ = tuple(slice(sta, sto) for sta, sto in zip(starts, stops))

        # check if we have a singleton in the return shape
        data_shape = tuple(idx.stop - idx.start for idx in index_)
        # remove singletons from data iff axis is not singleton in return data
        index_ = tuple(slice(idx.start, idx.stop) if sh > 1 or is_single else
                       slice(idx.start, idx.stop + 1)
                       for idx, sh, is_single in zip(index_, data_shape, singletons))
        data = self.volume[index_]

        # speed ups for empty blocks and masks
        dsum = data.sum()
        if dsum == 0:
            out = np.zeros(ret_shape, dtype=self.dtype)
        elif dsum == data.size:
            out = np.ones(ret_shape, dtype=self.dtype)
        else:
            out = self._interpolate(data, ret_shape)
        return squeeze_singletons(out, to_squeeze)
