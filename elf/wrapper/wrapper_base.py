from abc import ABC


class WrapperBase(ABC):
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
