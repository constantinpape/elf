from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .base import WrapperBase
from ..transformation import transform_subvolume_affine
from ..util import normalize_index, squeeze_singletons


# TODO
# - implement loading with halo for sub-slices to avoid boundary artifacts
# - support more dimensions and multichannel
class ResizedVolume(WrapperBase):
    """Wrapper to resize a volume on the fly.

    A resize corresponds to an affine transformation with a diagonal scale matrix that maps
    output coordinates to input coordinates. The actual interpolation is delegated to
    `elf.transformation.transform_subvolume_affine`, which is backed by `bioimage_cpp` and
    supports both in-memory (numpy) and chunked (e.g. zarr / hdf5) input data.

    Args:
        volume: The data to wrap.
        shape: The target shape for resizing.
        order: The interpolation order to use, supports orders 0 to 5.
        sigma_anti_aliasing: The sigma value used for pre-smoothing the input in order to
            avoid aliasing effects when downsampling. By default no pre-smoothing is applied.
    """
    def __init__(
        self,
        volume: ArrayLike,
        shape: Tuple[int, ...],
        order: int = 0,
        sigma_anti_aliasing: Optional[float] = None,
    ):
        if len(shape) != volume.ndim:
            raise ValueError(f"Expect volume and shape to have same dimensionality, got {len(shape)}, {volume.ndim}")
        if volume.ndim not in (2, 3):
            raise ValueError(f"Expect 2d or 3d input data, got {volume.ndim}")
        super().__init__(volume)
        self._shape = shape

        self._scale = [sh / float(fsh) for sh, fsh in zip(self.volume.shape, self.shape)]
        self.order = order
        self.sigma_anti_aliasing = sigma_anti_aliasing

        # bioimage_cpp's affine transform does not support boolean data, so for boolean volumes
        # we transform via uint8 and cast the result back to bool.
        self._is_bool = np.dtype(self.dtype) == np.dtype(bool)

        # Affine matrix mapping output coordinates to input coordinates (diagonal scale, no shift).
        self._matrix = np.diag(self._scale + [1.0])

    @property
    def shape(self):
        return self._shape

    @property
    def scale(self):
        return self._scale

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
        ret_shape = tuple(ind.stop - ind.start for ind in index)

        volume = self._volume
        if self._is_bool:
            volume = volume.astype("uint8") if isinstance(volume, np.ndarray) else volume

        out = transform_subvolume_affine(
            volume, self._matrix, index, order=self.order,
            fill_value=0, sigma=self.sigma_anti_aliasing,
        )

        if self._is_bool:
            out = out.astype(bool)
        assert out.shape == ret_shape, f"{out.shape}, {ret_shape}"
        return squeeze_singletons(out, to_squeeze)
