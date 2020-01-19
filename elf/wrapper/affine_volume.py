import numpy as np
from .base import WrapperBase
from ..transformation import compute_affine_matrix, affine_transform_for_subvolume
from ..util import normalize_index, squeeze_singletons


# we need to support origin shifts,
# but it's probably better to do this with the affine matrix already
class AffineVolume(WrapperBase):
    """ Apply affine transformation to the volume.

    Arguments:
        volume: volume to which to apply the affine.
        output_shape: output shape, deduced from data by default (default: None)
        affine_matrix: matrix defining the affine transformation
        scale: scale factor
        rotation: rotation in degrees
        shear: shear in degrees
        translation: translation in pixel
        order: order of interpolation (supports 0 up to 5)
        fill_value: fill value for invalid regions (default: 0)
        sigma_anti_aliasing: sigma used for smoothing
    """
    def __init__(self, volume, shape=None, affine_matrix=None,
                 scale=None, rotation=None, shear=None, translation=None,
                 order=0, fill_value=0, sigma_anti_aliasing=None):

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

        # TODO do we need the matrix or the inverse matrix ?
        # apply affine for the subvolume
        out = affine_transform_for_subvolume(self._volume, self.matrix, index,
                                             self.order, self.fill_value,
                                             self.sigma_anti_aliasing)

        return squeeze_singletons(out, to_squeeze)
