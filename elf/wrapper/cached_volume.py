from .wrapper_base import WrapperBase
from ..util import normalize_index, squeeze_singletons


# TODO implement
class CachedVolume(WrapperBase):
    """
    """
    def __init__(self, volume):
        super().__init__(volume)

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
