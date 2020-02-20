import numpy as np
from .base import SimpleTransformationWrapper


# TODO allow arbitrary range for normalization
class NormalizeWrapper(SimpleTransformationWrapper):
    """ Wrapper to normalize tensor to 0, 1.
    """
    eps = 1.e-6

    def __init__(self, volume, dtype='float32'):
        self._dtype = np.dtype(dtype)
        super().__init__(volume, self._normalize)

    @property
    def dtype(self):
        return self._dtype

    def _normalize(self, input_):
        input_ = input_.astype(self._dtype)
        input_ -= input_.min()
        input_ /= (input_.max() + self.eps)
        return input_


class ThresholdWrapper(SimpleTransformationWrapper):
    """ Wrapper to threshold tensor on the fly.
    """

    def __init__(self, volume, threshold, operator=np.greater):
        super().__init__(volume, lambda x: operator(x, threshold))
        self._threshold = threshold

    @property
    def threshold(self):
        return self._threshold

    @property
    def dtype(self):
        return np.bool
