import numpy as np
from .base import SimpleTransformationWrapper, WrapperBase
from ..util import normalize_index, squeeze_singletons


# TODO allow arbitrary range for normalization
class NormalizeWrapper(SimpleTransformationWrapper):
    """ Wrapper to normalize tensor to 0, 1.
    """
    eps = 1.e-6

    def __init__(self, volume, dtype="float32"):
        super().__init__(volume, self._normalize, dtype=np.dtype(dtype))

    def _normalize(self, input_):
        input_ = input_.astype(self.dtype)
        input_ -= input_.min()
        input_ /= (input_.max() + self.eps)
        return input_


class ThresholdWrapper(SimpleTransformationWrapper):
    """ Wrapper to threshold tensor on the fly.
    """

    def __init__(self, volume, threshold, operator=np.greater):
        super().__init__(volume, lambda x: operator(x, threshold), dtype=np.dtype("bool"))
        self._threshold = threshold

    @property
    def threshold(self):
        return self._threshold


class RoiWrapper(WrapperBase):
    """ Wrapper to only expose region of interest of the volume.
    """
    def __init__(self, volume, roi):
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
