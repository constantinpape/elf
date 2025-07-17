from abc import ABC
from typing import Tuple

from numpy.typing import ArrayLike
from ..util import normalize_index, squeeze_singletons


class WrapperBase(ABC):
    """@private
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
    """Wrapper to apply simple transformation to data on the fly.

    The transformation must depend only on the data values, not on coordinates.
    Hence, the expected function signature is `def transformation(volume)`.

    Args:
        volume: The data to wrap.
        transformation: The transformation to apply.
        with_channels: Whether the data has channels.
        kwargs: Additional keyword arguments for the base class.
    """
    def __init__(self, volume: ArrayLike, transformation: callable, with_channels: bool = False, **kwargs) -> None:
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
        if self.with_channels and to_squeeze:
            to_squeeze = tuple(sq + 1 for sq in to_squeeze)
        out = squeeze_singletons(out, to_squeeze)
        return out


class SimpleTransformationWrapperWithHalo(WrapperBase):
    """Wrapper to apply simple transformation to data on the fly with a halo.

    The transformation must depend only on the data values, not on coordinates.
    Hence, the expected function signature is `def transformation(volume)`.

    Args:
        volume: The data to wrap.
        transformation: The transformation to apply.
        halo: The halo for the transformation.
        with_channels: Whether the data has channels.
        kwargs: Additional keyword arguments for the base class.
    """
    def __init__(
        self,
        volume: ArrayLike,
        transformation: callable,
        halo: Tuple[int, ...],
        with_channels: bool = False,
        **kwargs
    ) -> None:
        super().__init__(volume, **kwargs)
        if not callable(transformation):
            raise ValueError("Expect the transformation to be callable.")
        if self.ndim != len(halo):
            raise ValueError(f"Invalid halo: {halo}")
        self.transformation = transformation
        self.halo = halo
        self.with_channels = with_channels

    def _extend_halo(self, index):
        assert len(index) == len(self.halo)
        extended_index, local_crop = [], []
        for idx, ha, sh in zip(index, self.halo, self.shape):
            start = max(idx.start - ha, 0)
            stop = min(idx.stop + ha, sh)
            extended_index.append(slice(start, stop))
            crop_len = stop - start
            left_halo = idx.start - start
            right_halo = stop - idx.stop
            local_crop.append(slice(left_halo, crop_len - right_halo))
        return tuple(extended_index), tuple(local_crop)

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
        index, local_crop = self._extend_halo(index)
        if self.with_channels:
            index = (slice(None),) + index
        out = self._volume[index]
        out = self.transformation(out)
        if self.with_channels and to_squeeze:
            to_squeeze = tuple(sq + 1 for sq in to_squeeze)
        out = out[local_crop]
        out = squeeze_singletons(out, to_squeeze)
        return out


class TransformationWrapper(WrapperBase):
    """Wrapper to apply transformation to data on the fly.

    The transformation may depend on the data values and on the coordinates.
    Hence, the expected function signature is `def transformation(volume, index)`.

    Args:
        volume: The data to wrap.
        transformation: The transformation to apply.
        with_channels: Whether the data has channels.
        kwargs: Additional keyword arguments for the base class.
    """
    def __init__(self, volume: ArrayLike, transformation: callable, **kwargs) -> None:
        if not callable(transformation):
            raise ValueError("Expect the transformation to be callable.")
        self.transformation = transformation
        super().__init__(volume, **kwargs)

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
        out = self._volume[index]
        out = self.transformation(out, index)
        out = squeeze_singletons(out, to_squeeze)
        return out


class MultiTransformationWrapper(WrapperBase):
    """Volume wrapper to apply a transformation to blocks from multiple data volumes on the fly.

    The transformation must depend only on the data values, not on coordinates.
    Hence, the expected function signature is `def transformation(*volumes)`.

    Args:
        transformation: The transformation to apply.
        volumes: The data volumes to wrap.
        apply_to_list: Whether the transformation is applied to a list of inputs instead of
            passing each input as an individual argument via *.
        kwargs: Additional keyword arguments for the base class.
    """

    def __init__(
        self,
        transformation: callable,
        *volumes: ArrayLike,
        apply_to_list: bool = False,
        **kwargs
    ) -> None:
        if not callable(transformation):
            raise ValueError("Expect the transformation to be callable.")
        self._volumes = volumes
        assert all(vol.shape == self._volumes[0].shape for vol in self._volumes[1:])
        self.transformation = transformation
        self.apply_to_list = apply_to_list
        super().__init__(self._volumes[0], **kwargs)

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
        inputs = [vol[index] for vol in self._volumes]
        out = self.transformation(inputs) if self.apply_to_list else self.transformation(*inputs)
        return squeeze_singletons(out, to_squeeze)
