from abc import ABC
from ..util import normalize_index, squeeze_singletons


class WrapperBase(ABC):
    """ Base class for a volume / tensor wrapper.
    """
    def __init__(self, volume, shape=None, dtype=None, chunks=None):
        self._volume = volume
        self._shape = shape
        self._ndim = None if shape is None else len(shape)
        self._dtype = dtype
        self._chunks = chunks

    @property
    def volume(self):
        return self._volume

    @property
    def shape(self):
        return self._volume.shape if self._shape is None else self._shape

    @property
    def ndim(self):
        return self._volume.ndim if self._ndim is None else self._ndim

    @property
    def dtype(self):
        return self._volume.dtype if self._dtype is None else self._dtype

    @property
    def chunks(self):
        if self._chunks is not None:
            return self._chunks
        try:
            return self._volume.chunks
        except AttributeError:
            return None

    # most wrappers will not implement setitem
    def __setitem__(self, index, item):
        raise NotImplementedError("Setitem not implemented")


class SimpleTransformationWrapper(WrapperBase):
    """ Volume wrapper to apply transformation to the data on the fly.

    The transformation needs to be simple: it cannot depend on the coordinate.
    I.e. the function signature is `def transformation(volume)`
    """

    def __init__(self, volume, transformation, with_channels=False, **kwargs):
        if not callable(transformation):
            raise ValueError("Expect the transformation to be callable.")
        self.transformation = transformation
        self.with_channels = with_channels
        super().__init__(volume, **kwargs)

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
        if self.with_channels:
            index = (slice(None),) + index
        out = self._volume[index]
        out = self.transformation(out)
        # TODO take care of with channels
        return squeeze_singletons(out, to_squeeze)


class TransformationWrapper(WrapperBase):
    """ Volume wrapper to apply transformation to the data on the fly.

    The transformation is assumed to depend on the coordinate.
    I.e. the function signature is
    `def transformation(volume, index)`
    """

    def __init__(self, volume, transformation, **kwargs):
        if not callable(transformation):
            raise ValueError("Expect the transformation to be callable.")
        self.transformation = transformation
        super().__init__(volume, **kwargs)

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
        out = self._volume[index]
        out = self.transformation(out, index)
        return squeeze_singletons(out, to_squeeze)


# class TransformationMultiWrapper(WrapperBase):
#     """ Volume wrapper to apply transformation to multiple data blocks on the fly.
#
#     The transformation cannot change the shape of the data,
#     but it's assumed to depend on the coordinate. I.e. the function signature is
#     `def transformation(volume, index)`
#     """
#
#     def __init__(self, transformation, *volumes):
#         if not callable(transformation):
#             raise ValueError("Expect the transformation to be callable.")
#         self._volumes = volumes
#         assert all(vol.shape == self._volumes[0].shape for vol in self._volumes[1:])
#         self.transformation = transformation
#         super().__init__(self._volumes[0])
#
#     def __getitem__(self, key):
#         index, to_squeeze = normalize_index(key, self.shape)
#         inputs = [vol[index] for vol in self._volumes]
#         out = self.transformation(*inputs)
#         return squeeze_singletons(out, to_squeeze)
