import warnings
from itertools import product
from functools import partial
from numbers import Number
from typing import List, Optional, Tuple

import numpy as np
import bioimage_cpp as bic
from numpy.typing import ArrayLike

from .transform_impl import transform_subvolume
from ..util import sigma_to_halo


def update_parameters(scale, rotation, shear, translation, dim):
    """@private
    """
    if scale is None:
        scale = [1.] * dim
    if rotation is None:
        rotation = 0. if dim == 2 else [0.] * 3
    if shear is None:
        # TODO how many shear angles do we have in 3d ?
        shear = 0. if dim == 2 else [0.] * 3
    if translation is None:
        translation = [0.] * dim
    return scale, rotation, shear, translation


def affine_matrix_2d(scale=None, rotation=None, shear=None, translation=None, angles_in_degree=True):
    """@private
    """
    matrix = np.zeros((3, 3))
    scale, rotation, shear, translation = update_parameters(
        scale, rotation, shear, translation, dim=2
    )

    # Wrapper for new numpy behavour that returns array with single val instead of scalar.
    def np_wrap(x, func):
        ret = func(x)
        if hasattr(ret, "item"):
            ret = ret.item()
        return ret

    # Make life easier.
    cos, sin = partial(np_wrap, func=np.cos), partial(np_wrap, func=np.sin)
    sx, sy = scale

    if angles_in_degree:
        phi = np.deg2rad(rotation)
        shear_angle = np.deg2rad(shear)
    else:
        phi = rotation
        shear_angle = shear

    # TODO this formular is taken from skimage, however I am very skeptical about the shear, see
    # https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py#L786-L817
    # set the main transformation matrix
    matrix[0, 0] = sx * cos(phi)
    matrix[0, 1] = - sy * sin(phi + shear_angle)

    matrix[1, 0] = sx * sin(phi)
    matrix[1, 1] = sy * cos(phi + shear_angle)

    # set the translation
    matrix[:2, 2] = translation

    # set extra element
    matrix[2, 2] = 1
    return matrix


def affine_matrix_3d(scale=None, rotation=None, shear=None, translation=None, angles_in_degree=True):
    """@private
    """
    matrix = np.zeros((4, 4))
    scale, rotation, shear, translation = update_parameters(
        scale, rotation, shear, translation, dim=3
    )

    # make life easier
    cos, sin = np.cos, np.sin
    sx, sy, sz = scale
    if angles_in_degree:
        phi, theta, psi = np.deg2rad(rotation)
    else:
        phi, theta, psi = rotation

    # TODO this is missing shear !
    matrix[0, 0] = sx * cos(theta) * cos(psi)
    matrix[0, 1] = sy * (-cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi))
    matrix[0, 2] = sz * (sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi))

    matrix[1, 0] = sx * cos(theta) * sin(psi)
    matrix[1, 1] = sy * (cos(phi) * cos(psi) + sin(phi) * sin(theta) * sin(psi))
    matrix[1, 2] = sz * (- sin(phi) * cos(theta) + cos(phi) * sin(theta) * sin(psi))

    matrix[2, 0] = -sx * sin(theta)
    matrix[2, 1] = sy * sin(phi) * sin(theta)
    matrix[2, 2] = sz * cos(phi) * cos(theta)

    # set the translation
    matrix[:3, 3] = translation

    # set extra elemnt
    matrix[3, 3] = 1
    return matrix


# shear is not working properly yet
def compute_affine_matrix(
    scale: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    shear: Optional[List[float]] = None,
    translation: Optional[List[float]] = None,
) -> np.ndarray:
    """Compute 2d or 3d affine matrix.

    Args:
        scale: Scaling factors for the dimensions, must have length 2 for 2d / 3 for 3d.
        rotation: Rotation, single angle in 2d, three euler angles (phi, theta, psi) in 3d, in degrees.
        shear: Shear angle, NOT WORKING CORRECTLY YET.
        translation: Translation along the dimensions, must have length 2 for 2d / 3 for 3d.

    Returns:
        The affine matrix.
    """

    # validate the input parameters
    parameters = [scale, rotation, shear, translation]
    assert not all(param is None for param in parameters)

    # determine and validate the dimension
    lens = [None if param is None else len(param) for param in parameters]
    # for scale and translation len = dimension
    # for rotation and shear len = 3 if dim == 3 else len = 1
    dims = [ll if ii in (0, 3) else 2 if ll == 1 else 3 for ii, ll in enumerate(lens) if ll is not None]
    assert len(set(dims)) == 1
    dim = dims[0]

    matrix = affine_matrix_2d(scale, rotation, shear, translation) if dim == 2 else\
        affine_matrix_3d(scale, rotation, shear, translation)
    return matrix


def transform_coordinate(coord, matrix):
    """@private
    """
    # x = matrix[0, 0] * coord[0] + matrix[0, 1] * coord[1] + matrix[0, 2] * coord[2] + matrix[0, 3]
    # y = matrix[1, 0] * coord[0] + matrix[1, 1] * coord[1] + matrix[1, 2] * coord[2] + matrix[1, 3]
    # z = matrix[2, 0] * coord[0] + matrix[2, 1] * coord[1] + matrix[2, 2] * coord[2] + matrix[2, 3]
    ndim = len(coord)
    return tuple(sum(coord[jj] * matrix[ii, jj] for jj in range(ndim)) + matrix[ii, -1] for ii in range(ndim))


def transform_roi_with_affine(
    roi_start: List[float], roi_stop: List[float], matrix: np.ndarray
) -> Tuple[List[float], List[float]]:
    """Transform a region of interest with an affine transformation.

    Args:
        roi_start: The start of the region of interest, corresponding to the lower left corner.
        roi_stop: The stop of the region of intereset, corresponding to the upper right corner.
        matrix: The affine matrix.

    Returns:
        The transformed start coordinates of the ROI.
        The transformed stop coordinates of the ROI.
    """
    dim = len(roi_start)

    corners = [corner for corner in product(*zip(roi_start, roi_stop))]
    transformed_corners = [transform_coordinate(corner, matrix) for corner in corners]

    transformed_start = [min(corner[d] for corner in transformed_corners) for d in range(dim)]
    transformed_stop = [max(corner[d] for corner in transformed_corners) for d in range(dim)]
    return transformed_start, transformed_stop


# TODO we assume no shear here
# extract components from the affine matrix, cf.
# https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati/417813

def translation_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Extract the translation vector from an affine matrix.

    Args:
        matrix: The affine matrix.

    Returns:
        The translation vector.
    """
    ndim = matrix.shape[0] - 1
    return matrix[:ndim, ndim]


def scale_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Extract the scale from an affine matrix.

    Args:
        matrix: The affine matrix.

    Returns:
        The scale factors.
    """
    ndim = matrix.shape[0] - 1
    scale = [np.linalg.norm(matrix[:ndim, d]) for d in range(ndim)]
    return np.array(scale)


# TODO need to figure out how to go from affine elements to euler angles
# def rotation_from_matrix(matrix):
#     """Return the rotation from the affine matrix """
#     pass


def _transform_subvolume_affine_chunked(data, matrix, bb, order, fill_value, sigma):
    """@private

    Apply an affine transformation to a sub-volume of chunked on-disk data.

    bioimage-cpp's affine transform only operates on numpy arrays, so we reconstruct the chunked
    transformation by blocking over the output frame: for each output block we determine the input
    region it samples from (by transforming the block corners through the matrix), read only that
    region into memory, and apply the in-memory affine transform with a matrix shifted into the
    local coordinate frames.
    """
    ndim = data.ndim
    matrix = np.asarray(matrix, dtype="float64")
    linear = matrix[:ndim, :ndim]
    translation = matrix[:ndim, ndim]

    out_start = tuple(b.start for b in bb)
    out_stop = tuple(b.stop for b in bb)
    out_shape = tuple(sto - sta for sta, sto in zip(out_start, out_stop))
    out = np.full(out_shape, fill_value, dtype=data.dtype)

    # The required input region per output block is computed from the (linear) corner mapping.
    # We pad it by an interpolation halo (and a smoothing halo if pre-smoothing is requested) and
    # then clamp to the data bounds so that the in-memory transform finds all samples it needs.
    halo = order + 1
    if sigma is not None:
        sigma_halo = sigma_to_halo(sigma, order)
        if isinstance(sigma_halo, Number):
            sigma_halo = ndim * (sigma_halo,)
        halo = tuple(halo + sh for sh in sigma_halo)
    else:
        halo = ndim * (halo,)

    blocking = bic.utils.Blocking(list(out_start), list(out_stop), list(data.chunks))
    for block_id in range(blocking.number_of_blocks):
        block = blocking.get_block(block_id)
        block_begin = list(block.begin)
        block_end = list(block.end)

        # Determine the input region sampled by this output block and pad / clamp it.
        in_start_f, in_stop_f = transform_roi_with_affine(block_begin, block_end, matrix)
        in_start = [max(0, int(np.floor(s)) - h) for s, h in zip(in_start_f, halo)]
        in_stop = [min(sh, int(np.ceil(s)) + h) for s, h, sh in zip(in_stop_f, halo, data.shape)]
        if any(sto <= sta for sta, sto in zip(in_start, in_stop)):
            # The block maps entirely outside the input data, it stays at fill_value.
            continue

        in_bb = tuple(slice(sta, sto) for sta, sto in zip(in_start, in_stop))
        in_region = np.asarray(data[in_bb])
        if sigma is not None:
            in_region = bic.filters.gaussian_smoothing(in_region, sigma).astype(data.dtype, copy=False)

        # Shift the matrix into the local coordinate frames: it maps local output coordinates
        # (starting at 0) to local input coordinates (relative to in_start).
        local_matrix = matrix.copy()
        local_matrix[:ndim, ndim] = linear @ np.array(block_begin, dtype="float64") + translation - in_start

        block_shape = tuple(end - beg for beg, end in zip(block_begin, block_end))
        local_bb = tuple(slice(0, sh) for sh in block_shape)
        res = bic.transformation.affine_transform(
            in_region, local_matrix, bounding_box=local_bb, order=order, fill_value=fill_value
        )

        out_bb = tuple(slice(beg - osta, end - osta) for beg, end, osta in zip(block_begin, block_end, out_start))
        out[out_bb] = res

    return out


def transform_subvolume_affine(
    data: ArrayLike,
    matrix: np.ndarray,
    bb: Tuple[slice, ...],
    order: int = 0,
    fill_value: Number = 0,
    sigma: Optional[float] = None,
    use_python_fallback_impl: bool = False,
) -> np.ndarray:
    """Apply affine transformation to a subvolume.

    Args:
        data: The input data, can be a numpy array or a chunked array-like object (e.g. zarr / hdf5).
        matrix: The matrix defining the affine transformation, with shape (ndim + 1, ndim + 1).
            It maps output coordinates to input coordinates in numpy axis order.
        bb: The bounding box into the output data.
        order: The interpolation order, supports orders 0 to 5 (see bioimage_cpp.transformation.affine_transform).
        fill_value: Output value for invald coordinates.
        sigma: Sigma value used for pre-smoothing the input in order to avoid aliasing effects.
        use_python_fallback_impl: Whether to use the slow pure python implementation.

    Returns:
        The transformed subvolume.
    """
    if use_python_fallback_impl:
        warnings.warn("Using the slow pure python implementation for the affine transformation.")
        trafo = partial(transform_coordinate, matrix=matrix)
        return transform_subvolume(data, trafo, bb, order=order, fill_value=fill_value, sigma=sigma)

    if isinstance(data, np.ndarray):
        if sigma is None:
            return bic.transformation.affine_transform(
                data, matrix, bounding_box=bb, order=order, fill_value=fill_value
            )
        return bic.transformation.resample(
            data, matrix, bounding_box=bb, order=order, fill_value=fill_value, anti_aliasing_sigma=sigma
        )

    # Chunked on-disk data: bioimage-cpp's affine transform is numpy-only, so we apply it block-wise.
    return _transform_subvolume_affine_chunked(data, matrix, bb, order, fill_value, sigma)
