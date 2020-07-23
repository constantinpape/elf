import numpy as np
from . import elastix_parser
from .affine import affine_matrix_3d

# Converter functions to translate in between different representations of affine transformations.
# Currently supports the following representations:
# - native:  the representation expected by elf transformation functions (same as scipy);
#            transformation is represented by 4x4 affine matrix
#            transformation is always given in voxel space
#            transformation is given in backward direction
# - bdv:     the representation in big dataviewer
#            transformation is represented by parameter vector of length 12
#            transformation is given in (variable) physical unit
#            transformation is given in forward direction
# - elastix: the representation in elastix;
#            affine transformations and metadata are stored in a text file
#            physical unit is milimeter
#            transformation is given in backward direction

# TODO
# - implement converter for 2d as well
# - support converting transformations to elastix


#
# General
#

def pretty_print_trafo(trafo):
    if isinstance(trafo, np.ndarray) and trafo.ndim == 2:
        trafo = matrix_to_parameters(trafo)
    trafo = " ".join([f"{param:.4f}" for param in trafo])
    print(trafo)


def matrix_to_parameters(matrix):
    """ Affine matrix to parameter vector.

    Returns parameter vector layed out as
    [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23]
    """
    assert matrix.shape == (4, 4)
    trafo = matrix[0].tolist() + matrix[1].tolist() + matrix[2].tolist()
    return trafo


def parameters_to_matrix(trafo):
    """ Parameter vector to affine matrix.

    Assumes parameter vector layed out as
    [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23]
    """
    assert len(trafo) == 12

    sub_matrix = np.zeros((3, 3), dtype='float64')
    sub_matrix[0, 0] = trafo[0]
    sub_matrix[0, 1] = trafo[1]
    sub_matrix[0, 2] = trafo[2]

    sub_matrix[1, 0] = trafo[4]
    sub_matrix[1, 1] = trafo[5]
    sub_matrix[1, 2] = trafo[6]

    sub_matrix[2, 0] = trafo[8]
    sub_matrix[2, 1] = trafo[9]
    sub_matrix[2, 2] = trafo[10]

    shift = [trafo[3], trafo[7], trafo[11]]

    matrix = np.zeros((4, 4))
    matrix[:3, :3] = sub_matrix
    matrix[:3, 3] = shift
    matrix[3, 3] = 1

    return matrix


#
# Elastix
#

def _elastix_affine_to_bdv(trafo):
    assert len(trafo) == 12

    sub_matrix = np.zeros((3, 3), dtype='float64')
    sub_matrix[0, 0] = trafo[0]
    sub_matrix[0, 1] = trafo[1]
    sub_matrix[0, 2] = trafo[2]

    sub_matrix[1, 0] = trafo[3]
    sub_matrix[1, 1] = trafo[4]
    sub_matrix[1, 2] = trafo[5]

    sub_matrix[2, 0] = trafo[6]
    sub_matrix[2, 1] = trafo[7]
    sub_matrix[2, 2] = trafo[8]

    shift = [trafo[9], trafo[10], trafo[11]]

    matrix = np.zeros((4, 4))
    matrix[:3, :3] = sub_matrix
    matrix[:3, 3] = shift
    matrix[3, 3] = 1

    return matrix


def _elastix_euler_to_bdv(trafo):
    assert len(trafo) == 6
    matrix = affine_matrix_3d(rotation=trafo[:3],
                              translation=trafo[3:],
                              angles_in_degree=False)
    return matrix


def _elastix_similarity_to_bdv(trafo):
    assert len(trafo) == 7
    scale = 3 * [trafo[-1]]
    matrix = affine_matrix_3d(scale=scale,
                              rotation=trafo[:3],
                              translation=trafo[3:6],
                              angles_in_degree=False)
    return matrix


def _elastix_translation_to_bdv(trafo):
    assert len(trafo) == 3
    matrix = affine_matrix_3d(translation=trafo)
    return matrix


def elastix_parameter_to_bdv_matrix(trafo, trafo_type):
    """ Convert elastix parameters to affine matrix in bdv convention.

    Note that the elastix parameter use a different convention than
    what is used natively and by bdv.
    """

    if trafo_type == 'AffineTransform':
        matrix = _elastix_affine_to_bdv(trafo)
    elif trafo_type == 'EulerTransfrom':
        matrix = _elastix_euler_to_bdv(trafo)
    elif trafo_type == 'SimilarityTransform':
        matrix = _elastix_similarity_to_bdv(trafo)
    elif trafo_type == 'TranslationTransform':
        matrix = _elastix_translation_to_bdv(trafo)
    else:
        raise ValueError(f"Invalid transformation type {trafo_type}")

    return matrix


def _convert_elastix_trafo(trafo_file):
    """Based on:
    https://github.com/image-transform-converters/image-transform-converters/blob/c405a8c820a3a3e0e35a40183384da7372687d4a/src/main/java/itc/converters/ElastixAffine3DToAffineTransform3D.java
    """
    trafo = elastix_parser.get_transformation(trafo_file)
    trafo_type = elastix_parser.get_transformation_type(trafo_file)

    # we make all calculations in bdv convention and only change to the python convention in the end

    # go fron the transformation vector to affine matrix. we can just use the
    # bdv functionality, because both bdv and elastix have the same axis convention
    trafo = elastix_parameter_to_bdv_matrix(trafo, trafo_type)

    # initialize the resulting affine matrix with the identity
    matrix = affine_matrix_3d()

    # load the rotation center from the elastix transformation definition
    rot_center = elastix_parser.get_rotation_center(trafo_file)
    if rot_center is not None:
        rot_center_neg = [-ce for ce in rot_center]

        # translate to the rotation center
        translate_to_rot = affine_matrix_3d(translation=rot_center_neg)
        matrix = translate_to_rot @ matrix

    # apply rotation and scale
    rot_and_scale = trafo.copy()
    rot_and_scale[3, :3] = 0
    matrix = rot_and_scale @ matrix

    # translate back from the rotation center
    if rot_center is not None:
        translate_from_rot = affine_matrix_3d(translation=rot_center)
        matrix = translate_from_rot @ matrix

    return matrix


def _combine_elastix_trafos(trafos, resolution, scale_factor):

    # transformation to scale from voxel space to millimeter (which is the fixed physical unit in elastix)
    vox_to_mm = affine_matrix_3d(scale=[scale_factor] * 3)

    # transformation to scale from millimiter to the physicial unit we use
    # usually we use micrometer and then scale_factor = 10^3
    # for nanometer it would be 10^6 etc.
    mm_to_unit = affine_matrix_3d(scale=[res / scale_factor for res in resolution])

    # combine the scaling transfomraions and the actual elastix transformations
    matrix = vox_to_mm
    for trafo in trafos:
        # elastix uses the opposite transformation direction as bdv,
        # so we need to invert the elastix transformation here
        matrix = matrix @ np.linalg.inv(trafo)
    matrix = matrix @ mm_to_unit

    return matrix


def elastix_to_bdv(trafo_file, resolution, scale_factor=1e3, load_initial_trafos=True):
    """ Convert elastix transformation in text file to bdv transformation.

    Arguments:
        trafo_file [str] - the file defining the elastix transformation
        resolution [list[float]] - resolution of the data in physical units
        scale_factor [float] - scale factor of physical units compared to millimeter, which is
            the default unit for elastix tranformations. By default, assume that physical
            units is in micrometer, which corresponds to a scale of 10^-3 (default: 1e3)
        load_initial_trafos [bool] - whether to load the initial transformations (default: True)
    Returns:
        list - parameter vector for bdv transformation
    """

    trafo_type = elastix_parser.get_transformation_type(trafo_file)
    if trafo_type is None or trafo_type not in elastix_parser.AFFINE_COMPATIBLE:
        msg = f"Transormation type in {trafo_file}: {trafo_type} is not compatible with affine transformation"
        raise ValueError(msg)

    trafo_files = [trafo_file]

    if load_initial_trafos:
        initial_trafo_file = elastix_parser.get_initial_transform_file(trafo_file)
    else:
        initial_trafo_file = None

    # load all transformations that need to be concatenated from the elastix transformation file
    while initial_trafo_file is not None:
        trafo_files.append(initial_trafo_file)
        initial_trafo_file = elastix_parser.get_initial_transform_file(initial_trafo_file)

        if initial_trafo_file is not None:
            trafo_type = elastix_parser.get_transformation_type(trafo_file)
            if trafo_type is None or trafo_type not in elastix_parser.AFFINE_COMPATIBLE:
                msg = (f"Transormation type in {initial_trafo_file}: {trafo_type}"
                       "is not compatible with affine transformation")
                raise ValueError(msg)

    # reverse the order of transformations and load the transformations
    # in bdv matrix format
    trafo_files = trafo_files[::-1]
    trafos = [_convert_elastix_trafo(trafo) for trafo in trafo_files]

    # combine the transformations and apply the scaling transformations to
    # change from the elastix space (measured in millimeter)
    # to the bdv space (which maps the physical unit (usually micrometer) to voxels)
    trafo = _combine_elastix_trafos(trafos, resolution, scale_factor)

    # return the transformation as parameter vector instead of affine matrix
    trafo = matrix_to_parameters(trafo)
    return trafo


def elastix_to_native(trafo_file, load_initial_trafos=True):
    """ Convert elastix transformation in text file to native transformation.

    Arguments:
        trafo_file [str] - the file defining the elastix transformation
        load_initial_trafos [bool] - whether to load the initial transformations (default: True)
    Returns:
        np.ndarray - 4x4 affine matrix in native format
    """
    # the native transformation is always computed in voxel space
    resolution = [1., 1., 1.]

    # get the resolution from elastix (in millimeter) to compute the correct scale factor
    res_elastix = elastix_parser.get_resolution(trafo_file)
    scale_factor = resolution[0] / res_elastix[0]

    # TODO support anisotropy, but need to figure out axis conventions ...
    if any(re / res != scale_factor for re, res in zip(resolution, res_elastix)):
        raise NotImplementedError

    # compute the transformation in bdv space and then convert to native
    trafo = elastix_to_bdv(trafo_file, resolution, scale_factor=scale_factor,
                           load_initial_trafos=load_initial_trafos)
    return bdv_to_native(trafo, invert=True)


#
# BigDataViewer
#

def bdv_to_native(trafo, resolution=None, invert=True):
    """ Convert bdv transformation parameter vector to
    affine matrix in native format.

    Bdv and elf expect the transformation in the opposite direction.
    So to be directly applied the transformation also needs to be inverted.
    (In addition to changing between the axis conventions.)
    The bdv transformations often also include the transformation from
    voxel space to physical space.

    Arguments:
        trafo [listlike] - parameter vector of the bdv transformation
        resolution [listlike] - physical resolution of the data in bdv.
            If given, the transformation will be scaled to voxel sapec (default: None)
        invert [bool] - invert the resulting affine matrix.
            This is necessary to apply the affine matrix directly in elf (default: True)
    Returns:
        np.ndarray - 4x4 affine matrix
    """
    assert len(trafo) == 12

    sub_matrix = np.zeros((3, 3), dtype='float64')
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

    # TODO include scaling transformation from physical space to voxel space
    if resolution is not None:
        raise NotImplementedError

    if invert:
        matrix = np.linalg.inv(matrix)

    return matrix


def native_to_bdv(matrix, resolution=None, invert=True):
    """ Convert affine matrix in native format to
    bdv transformation parameter vector.

    Bdv and elf expect the transformation in the opposite direction.
    So to be directly applied the transformation also needs to be inverted.
    (In addition to changing between the axis conventions.)
    The bdv transformations often also include the transformation from
    voxel space to physical space.

    Arguments:
        trafo [listlike] - parameter vector of the bdv transformation
        resolution [listlike] - physical resolution of the data in bdv.
            If given, the transformation will be scaled to voxel sapec (default: None)
        invert [bool] - invert the resulting affine matrix.
            This is necessary to apply the affine matrix directly in elf (default: True)
    Returns:
        np.ndarray - 4x4 affine matrix in native format
    """
    # TODO include scaling transformation from physical space to voxel space
    if resolution is not None:
        raise NotImplementedError

    if invert:
        matrix = np.linalg.inv(matrix)

    trafo = [matrix[2, 2], matrix[2, 1], matrix[2, 0], matrix[2, 3],
             matrix[1, 2], matrix[1, 1], matrix[1, 0], matrix[1, 3],
             matrix[0, 2], matrix[0, 1], matrix[0, 0], matrix[0, 3]]
    return trafo
