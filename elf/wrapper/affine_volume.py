from math import ceil
import numpy as np
from scipy.ndimage import affine_transform

try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff

from .wrapper_base import WrapperBase
from ..transformation import compute_affine_matrix, transform_roi
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
    def __init__(self, volume, output_shape=None,
                 affine_matrix=None,
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

        # get the inverse matrix
        self.inverse_matrix = np.linalg.inv(self.matrix)

        # comptue the extent and where the origin is mapped to in the target space
        extent, origin = self.compute_extent_and_origin()
        self.origin = origin

        # compute the shape after interpolation
        if output_shape is None:
            self._shape = extent
        else:
            assert isinstance(output_shape, tuple)
            assert len(output_shape) == self.ndim
            self._shape = output_shape

    @property
    def shape(self):
        return self._shape

    def compute_extent_and_origin(self):
        roi_start, roi_stop = transform_roi([0] * self.ndim, self.volume.shape, self.matrix)
        extent = tuple(int(ceil(sto - sta)) for sta, sto in zip(roi_start, roi_stop))
        return extent, roi_start

    def crop_to_input_space(self, roi_start, roi_stop):
        return ([max(rs, 0) for rs in roi_start],
                [min(rs, sh) for rs, sh in zip(roi_stop, self.volume.shape)])

    def __getitem__(self, index):
        # 1.) normalize the index to have a proper bounding box
        index, to_squeeze = normalize_index(index, self.shape)
        roi_start, roi_stop = [ind.start for ind in index], [ind.stop for ind in index]
        out_shape = tuple(sto - sta for sta, sto in zip(roi_start, roi_stop))

        # 2.) transform the bounding box back to the original space
        # (= coordinate system of self.volume)
        # first, add origin as offset
        tr_start = [rs + orig for rs, orig in zip(roi_start, self.origin)]
        tr_stop = [rs + orig for rs, orig in zip(roi_stop, self.origin)]
        # then, pull coordinates back to the initial coordinate space
        tr_start, tr_stop = transform_roi(tr_start, tr_stop, self.inverse_matrix)

        # 3.) crop the transformed bounding box to the valid region
        # and load the input data
        tr_start, tr_stop = self.crop_to_input_space(tr_start, tr_stop)
        transformed_index = tuple(slice(int(sta), int(ceil(sto)))
                                  for sta, sto in zip(tr_start, tr_stop))
        input_ = self.volume[transformed_index]

        if self.sigma_anti_aliasing is not None:
            input_ = ff.gaussianSmoothing(input_.astype('float32'), self.sigma_anti_aliasing)

        # NOTE: this still doesn't pass all the tests fro cutouts
        # 4.) adapt the matrix for the local cutout
        offset, _ = transform_roi(self.ndim * [0], input_.shape, self.matrix)
        mat = self.matrix.copy()
        mat[:self.ndim, self.ndim] = [-off for off in offset]
        mat = np.linalg.inv(mat)

        # 5.) apply the affine transformation
        # out_shape = [2 * sh for sh in out_shape]
        out = affine_transform(input_, mat, output_shape=out_shape, order=self.order,
                               mode='constant', cval=self.fill_value).astype(self.dtype)

        return squeeze_singletons(out, to_squeeze)

    def __setitem__(self, index, item):
        raise NotImplementedError("Setitem not implemented")
