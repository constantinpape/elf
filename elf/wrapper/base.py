from abc import ABC
from ..util import normalize_index, squeeze_singletons


class WrapperBase(ABC):
    """ Base class for a volume / tensor wrapper.
    """
    def __init__(self, volume):
        self._volume = volume

    @property
    def volume(self):
        return self._volume

    @property
    def shape(self):
        return self._volume.shape

    @property
    def ndim(self):
        return self._volume.ndim

    @property
    def dtype(self):
        return self._volume.dtype

    @property
    def chunks(self):
        try:
            return self._volume.chunks
        except AttributeError:
            return None


class SimpleTransformationWrapper(WrapperBase):
    """ Volume wrapper to apply transformation to the data on the fly.

    The transformation needs to be simple, i.e. it cannot change the shape
    of the data and cannot depend on the coordinate. I.e. the function signature is
    `def transformation(volume)`
    """

    def __init__(self, volume, transformation):
        if not callable(transformation):
            raise ValueError("Expect the transformation to be callable.")
        self.transformation = transformation
        super().__init__(volume)

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
        out = self._volume[index]
        out = self.transformation(out)
        return squeeze_singletons(out, to_squeeze)


class TransformationWrapper(WrapperBase):
    """ Volume wrapper to apply transformation to the data on the fly.

    The transformation cannot change the shape of the data,
    but it's assumed to depend on the coordinate. I.e. the function signature is
    `def transformation(volume, index)`
    """

    def __init__(self, volume, transformation):
        if not callable(transformation):
            raise ValueError("Expect the transformation to be callable.")
        self.transformation = transformation
        super().__init__(volume)

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
        out = self._volume[index]
        out = self.transformation(out, index)
        return squeeze_singletons(out, to_squeeze)
