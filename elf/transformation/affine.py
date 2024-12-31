import warnings
from itertools import product
from functools import partial
from numbers import Number
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .transform_impl import transform_subvolume
from ..io import is_z5py, is_h5py

try:
    import nifty.transformation as ntrafo
except ImportError:
    ntrafo = None


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
    # make life easier
    cos, sin = np.cos, np.sin
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
        data: The input data, can be a numpy array or another array-like object.
        matrix: The 4x4 matrix defining the affine transformation.
        bb: The ounding box into the output data.
        order: The interpolation order.
        fill_value: Output value for invald coordinates.
        sigma: Sigma value used for pre-smoothing the input in order to avoid aliasing effects.
        use_python_fallback_impl: Whether to use the slow pure python implementation.

    Returns:
        The transformed subvolume.
    """
    # TODO implement pre-smoothing in nifty
    # TODO more orders in nifty
    has_nifty_trafo = (ntrafo is not None) and (isinstance(data, np.ndarray) or is_z5py(data) or is_h5py(data))
    has_nifty_trafo = has_nifty_trafo and (sigma is None) and (order < 2)

    if has_nifty_trafo:
        if isinstance(data, np.ndarray):
            return ntrafo.affineTransformation(data, matrix, order, bb, fill_value)
        elif is_z5py(data):
            return ntrafo.affineTransformationZ5(data, matrix, order, bb, fill_value, sigma)
        elif is_h5py(data):
            return ntrafo.affineTransformationH5(data, matrix, order, bb, fill_value, sigma)
    else:
        if not use_python_fallback_impl:
            msg = (
                "Could not find c++ implementation for affine transformation"
                "set 'use_python_fallback_impl' to True to compute the transformation via slow python fallback"
            )
            raise RuntimeError(msg)
        warnings.warn("Could not find c++ implementation for affine transformation, using slow python implementation.")
        trafo = partial(transform_coordinate, matrix=matrix)
        return transform_subvolume(
            data, trafo, bb, order=order, fill_value=fill_value, sigma=sigma
        )
