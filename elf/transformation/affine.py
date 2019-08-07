from itertools import product
import numpy as np


def update_parameters(scale, rotation, shear, translation, dim):
    if scale is None:
        scale = [1.] * dim
    if rotation is None:
        rotation = [0.] if dim == 2 else [0.] * 3
    if shear is None:
        # TODO how many shear angles do we have in 3d ?
        shear = [0.] * (dim - 1)
    if translation is None:
        translation = [0.] * dim
    return scale, rotation, shear, translation


def affine_matrix_2d(scale=None, rotation=None, shear=None, translation=None):
    matrix = np.zeros((3, 3))
    scale, rotation, shear, translation = update_parameters(scale,
                                                            rotation,
                                                            shear,
                                                            translation,
                                                            dim=2)
    # make life easier
    cos, sin = np.cos, np.sin
    sx, sy = scale
    phi = np.deg2rad(rotation)[0]
    shear_angle = np.deg2rad(shear)[0]

    # TODO this formular is taken from skimage, however I am very skeptical about
    # the shear, see
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


def affine_matrix_3d(scale=None, rotation=None, shear=None, translation=None):
    matrix = np.zeros((4, 4))
    scale, rotation, shear, translation = update_parameters(scale,
                                                            rotation,
                                                            shear,
                                                            translation,
                                                            dim=3)

    # make life easier
    cos, sin = np.cos, np.sin
    sx, sy, sz = scale
    phi, theta, psi = np.deg2rad(rotation)

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


# TODO shear is not working properly yet
def compute_affine_matrix(scale=None, rotation=None, shear=None, translation=None):
    """ Compute 2d or 3d affine matrix.

    Argeuments:
        scale [listlike]: scale-factor for axes, must have length 2 for 2d / 3 for 3d
        rotation [listlike]: rotation, single angle in 2d, three euler angles (phi, theta, psi) in 3d,
            expects degrees
        shear [listlike]: shear angle, NOT WORKING PROPERLY YET
        translation [listlike]: translation along axes, must have length 2 for 2d / 3 for 3d
    """

    # validate the input parameters
    parameters = [scale, rotation, shear, translation]
    assert not all(param is None
                   for param in parameters)

    # determine and validate the dimension
    lens = [None if param is None else len(param)
            for param in parameters]
    # for scale and translation len = dimension
    # for rotation and shear len = 3 if dim == 3 else len = 1
    dims = [ll if ii in (0, 3) else 2 if ll == 1 else 3
            for ii, ll in enumerate(lens) if ll is not None]
    assert len(set(dims)) == 1
    dim = dims[0]

    matrix = affine_matrix_2d(scale, rotation, shear, translation) if dim == 2 else\
        affine_matrix_3d(scale, rotation, shear, translation)
    return matrix


def transform_coordinate_2d(coord, matrix):
    x = matrix[0, 0] * coord[0] + matrix[0, 1] * coord[1] + matrix[0, 2]
    y = matrix[1, 0] * coord[0] + matrix[1, 1] * coord[1] + matrix[1, 2]
    return x, y


def transform_coordinate_3d(coord, matrix):
    x = matrix[0, 0] * coord[0] + matrix[0, 1] * coord[1] + matrix[0, 2] * coord[2] + matrix[0, 3]
    y = matrix[1, 0] * coord[0] + matrix[1, 1] * coord[1] + matrix[1, 2] * coord[2] + matrix[1, 3]
    z = matrix[2, 0] * coord[0] + matrix[2, 1] * coord[1] + matrix[2, 2] * coord[2] + matrix[2, 3]
    return x, y, z


def transform_roi(roi_start, roi_stop, matrix):
    """ Transform a roi under the affine transformation defined by
    the matrix.
    """
    dim = len(roi_start)
    trafo = transform_coordinate_2d if dim == 2 else transform_coordinate_3d

    corners = [corner for corner in product(*zip(roi_start, roi_stop))]
    transformed_corners = [trafo(corner, matrix) for corner in corners]

    transformed_start = [min(corner[d] for corner in transformed_corners) for d in range(dim)]
    transformed_stop = [max(corner[d] for corner in transformed_corners) for d in range(dim)]
    return transformed_start, transformed_stop


# TODO we assume no shear here
# extract components from the affine matrix, cf.
# https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati/417813

def translation_from_matrix(matrix):
    """ Return the translation vector from the affine matrix """
    ndim = matrix.shape[0] - 1
    translation = matrix[:ndim, ndim]
    return translation


def scale_from_matrix(matrix):
    """ Return the scales from the affine matrix """
    ndim = matrix.shape[0] - 1
    scale = [np.linalg.norm(matrix[:ndim, d]) for d in range(ndim)]
    return scale


# TODO need to figure out how to go from affine elements to euler angles
def rotation_from_matrix(matrix):
    """ Return the rotation from the affine matrix """
    pass


def bdv_trafo_to_affine_matrix(trafo):
    """ Translate bdv transformation (XYZ) to affine matrix (ZYX)
    """
    assert len(trafo) == 12

    sub_matrix = np.zeros((3, 3))
    sub_matrix[0, 0] = trafo[10]
    sub_matrix[0, 1] = trafo[9]
    sub_matrix[0, 2] = trafo[8]

    sub_matrix[1, 0] = trafo[6]
    sub_matrix[1, 1] = trafo[5]
    sub_matrix[1, 2] = trafo[4]

    sub_matrix[2, 0] = trafo[2]
    sub_matrix[2, 1] = trafo[1]
    sub_matrix[2, 2] = trafo[0]

    shift = [trafo[11], trafo[7], trafo[3]]

    matrix = np.zeros((4, 4))
    matrix[:3, :3] = sub_matrix
    matrix[:3, 3] = shift
    matrix[3, 3] = 1

    return matrix
