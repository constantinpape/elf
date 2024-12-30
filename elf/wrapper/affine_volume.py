from numbers import Number
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .base import WrapperBase
from ..transformation import compute_affine_matrix, transform_subvolume_affine
from ..util import normalize_index, squeeze_singletons


# we need to support origin shifts,
# but it's probably better to do this with the affine matrix already
class AffineVolume(WrapperBase):
    """Wrapper to apply affine transformation to data on the fly.

    The transformation can either be defined by an affine matrix,
    or by individual paramters for scale, rotation, shear and translation.

    Args:
        volume: The data to wrap.
        shape: The output shape, deduced from the data by default.
        affine_matrix: The matrix defining the affine transformation.
        scale: The scale factors.
        rotation: The rotation angles in degrees.
        shear: The shear angles in degrees.
        translation: The translation vector.
        order: The order of interpolation, supports 0 to 5.
        fill_value: The fill value for invalid regions.
        sigma_anti_aliasing: The sigma value used for smoothing.
    """
    def __init__(
        self,
        volume: ArrayLike,
        shape: Optional[Tuple[int, ...]] = None,
        affine_matrix: Optional[np.ndarray] = None,
        scale: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        shear: Optional[List[float]] = None,
        translation: Optional[List[float]] = None,
        order: int = 0,
        fill_value: Number = 0,
        sigma_anti_aliasing: Optional[float] = None,
    ):
        # TODO support 2d + channels and 3d + channels
        assert volume.ndim in (2, 3), "Only 2d or 3d supported"
        super().__init__(volume)

        # scipy transformation options
        self.order = order
        self.fill_value = fill_value
        self.sigma_anti_aliasing = sigma_anti_aliasing

        # validate the affine parameter
        have_matrix = affine_matrix is not None
        have_parameter = translation is not None or scale is not None or\
            rotation is not None or shear is not None

        if not (have_matrix != have_parameter):
            raise RuntimeError("Exactly one of affine_matrix or affine parameter needs to be passed")

        # get the affine matrix
        if have_matrix:
            self.matrix = affine_matrix
        else:
            assert shear is None, "Shear is not properly implemented yet"
            self.matrix = compute_affine_matrix(scale, rotation, shear, translation)
        assert self.matrix.shape == (self.ndim + 1, self.ndim + 1), "Invalid affine matrix"

        # build the inverse matrix
        self.inverse_matrix = np.linalg.inv(self.matrix)

        # set shape
        self._shape = volume.shape if shape is None else shape

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, index):
        # normalize the index
        index, to_squeeze = normalize_index(index, self.shape)

        # apply affine for the subvolume
        out = transform_subvolume_affine(self._volume, self.matrix, index,
                                         self.order, self.fill_value,
                                         self.sigma_anti_aliasing)

        return squeeze_singletons(out, to_squeeze)
