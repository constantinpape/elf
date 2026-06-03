from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from .base import SimpleTransformationWrapper, WrapperBase
from ..util import normalize_index, squeeze_singletons


class NormalizeWrapper(SimpleTransformationWrapper):
    """Wrapper to normalize data to [0, 1] on the fly.

    Args:
        volume: The data to wrap
        dtype: The data type.
    """
    eps = 1.e-6

    def __init__(self, volume: ArrayLike, dtype: str = "float32"):
        super().__init__(volume, self._normalize, dtype=np.dtype(dtype))

    def _normalize(self, input_):
        input_ = input_.astype(self.dtype)
        input_ -= input_.min()
        input_ /= (input_.max() + self.eps)
        return input_


class ThresholdWrapper(SimpleTransformationWrapper):
    """Wrapper to apply a threshold to data on the fly.

    Args:
        volume: The data to wrap.
        threshold: The threshold.
        operator: The operator for thresholding.
    """
    def __init__(self, volume: ArrayLike, threshold: float, operator: callable = np.greater):
        super().__init__(volume, lambda x: operator(x, threshold), dtype=np.dtype("bool"))
        self._threshold = threshold

    @property
    def threshold(self):
        return self._threshold


class RoiWrapper(WrapperBase):
    """Wrapper to restrict data to a region of interest.

    Args:
        volume: The data to wrap.
        roi: The region of interest
    """
    def __init__(self, volume: ArrayLike, roi: Tuple[slice, ...]):
        super().__init__(volume)
        self._roi, _ = normalize_index(roi, volume.shape)

    @property
    def shape(self):
        return tuple(b.stop - b.start for b in self._roi)

    def map_index_to_volume(self, index):
        index = tuple(slice(ind.start + roi.start, ind.stop + roi.start)
                      for ind, roi in zip(index, self._roi))
        return index

    def __getitem__(self, index):
        index, to_squeeze = normalize_index(index, self.shape)
        index = self.map_index_to_volume(index)
        out = self._volume[index]
        return squeeze_singletons(out, to_squeeze)

    def __setitem__(self, index, item):
        index, _ = normalize_index(index, self.shape)
        index = self.map_index_to_volume(index)
        self._volume[index] = item


class PadWrapper(WrapperBase):
    """Wrapper to pad the input.

    This only supports right-padding.

    Args:
        volume: The data to pad.
        pad_shape: The shape for padding the data.
    """
    def __init__(self, volume: ArrayLike, pad_width: Tuple[int, ...], mode: str = "constant"):
        assert volume.ndim == len(pad_width)
        super().__init__(volume)
        self._pad_width = pad_width
        self._shape = volume.shape
        self._mode = mode

    @property
    def shape(self):
        return tuple(sh + pw for sh, pw in zip(self._shape, self._pad_width))

    def __getitem__(self, index):
        index, to_squeeze = normalize_index(index, self.shape)

        local_pad, local_index = [], []
        for idx, sh in zip(index, self._shape):
            overhang_start = max(0, idx.start - sh)
            overhang_stop = max(0, idx.stop - sh)
            if overhang_start > 0:
                raise NotImplementedError
            elif overhang_stop > 0:
                local_pad.append(overhang_stop)
                local_index.append(slice(idx.start, sh))
            else:
                local_pad.append(0)
                local_index.append(idx)

        local_index = tuple(local_index)
        out = self._volume[local_index]
        if any(lpad > 0 for lpad in local_pad):
            pad_width = tuple((0, lpad) for lpad in local_pad)
            out = np.pad(out, pad_width, mode=self._mode)

        return squeeze_singletons(out, to_squeeze)
