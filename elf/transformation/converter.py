from typing import List, Optional
import numpy as np
from . import elastix_parser
from .affine import affine_matrix_2d, affine_matrix_3d

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
# - support converting transformations to elastix


#
# General
#

def pretty_print_trafo(trafo):
    """@private
    """
    if isinstance(trafo, np.ndarray) and trafo.ndim == 2:
        trafo = matrix_to_parameters(trafo)
    trafo = " ".join([f"{param:.4f}" for param in trafo])
    print(trafo)


def matrix_to_parameters(matrix: np.ndarray) -> List[float]:
    """Convert affine matrix to parameter vector.

    The parameter vector returned has the layout:
        [a00, a01, a02, a10, a11, a12] (for 2d) or
        [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23] (for 3d)

    Args:
        matrix: The affine matrix.

    Returns:
        The parameter vector.
    """
    if matrix.shape[0] == 4:
        assert matrix.shape == (4, 4)
        trafo = matrix[0].tolist() + matrix[1].tolist() + matrix[2].tolist()
    else:
        assert matrix.shape == (3, 3)
        trafo = matrix[0].tolist() + matrix[1].tolist()
    return trafo


def parameters_to_matrix(trafo: List[float]) -> np.ndarray:
    """Converts parameter vector to affine matrix.

    Assumes parameter vector that has the layout:
        [a00, a01, a02, a10, a11, a12] (for 2d) or
        [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23] (for 3d)

    Args:
        trafo: The transformation parameter vector.

    Returns:
        The affine transformation matrix.
    """
    if len(trafo) == 12:
        sub_matrix = np.zeros((3, 3), dtype="float64")
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

    elif len(trafo) == 6:
        sub_matrix = np.zeros((2, 2), dtype="float64")
        sub_matrix[0, 0] = trafo[0]
        sub_matrix[0, 1] = trafo[1]

        sub_matrix[1, 0] = trafo[3]
        sub_matrix[1, 1] = trafo[4]

        shift = [trafo[2], trafo[5]]

        matrix = np.zeros((3, 3))
        matrix[:2, :2] = sub_matrix
        matrix[:2, 2] = shift
        matrix[2, 2] = 1

    else:
        raise ValueError(f"Invalid number of parameters {len(trafo)}")

    return matrix


#
# Elastix
#

def _elastix_affine_to_bdv(trafo):
    if len(trafo) == 12:  # 3d transformation
        sub_matrix = np.zeros((3, 3), dtype="float64")
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

    elif len(trafo) == 6:  # 2d transformation
        sub_matrix = np.zeros((2, 2), dtype="float64")
        sub_matrix[0, 0] = trafo[0]
        sub_matrix[0, 1] = trafo[1]

        sub_matrix[1, 0] = trafo[2]
        sub_matrix[1, 1] = trafo[3]

        shift = [trafo[4], trafo[5]]

        matrix = np.zeros((3, 3))
        matrix[:2, :2] = sub_matrix
        matrix[:2, 2] = shift
        matrix[2, 2] = 1

    else:
        raise ValueError(f"Invalid number of parameters for affine transformation: {len(trafo)}")

    return matrix


def _elastix_euler_to_bdv(trafo):
    nparam = len(trafo)
    if nparam == 6:
        matrix = affine_matrix_3d(rotation=trafo[:3],
                                  translation=trafo[3:],
                                  angles_in_degree=False)
    elif nparam == 3:
        matrix = affine_matrix_2d(rotation=trafo[0],
                                  translation=trafo[1:],
                                  angles_in_degree=False)
    else:
        raise ValueError(f"Invalid number of parameters for euler transform: {nparam}")
    return matrix


def _elastix_similarity_to_bdv(trafo):
    nparam = len(trafo)
    if nparam == 7:
        scale = 3 * [trafo[-1]]
        matrix = affine_matrix_3d(scale=scale,
                                  rotation=trafo[:3],
                                  translation=trafo[3:6],
                                  angles_in_degree=False)
    elif nparam == 4:
        scale = 2 * [trafo[0]]
        matrix = affine_matrix_2d(scale=scale,
                                  rotation=trafo[1],
                                  translation=trafo[2:],
                                  angles_in_degree=False)
    else:
        raise ValueError(f"Invalid number of parameters for similarity transform: {nparam}")
    return matrix


def _elastix_translation_to_bdv(trafo):
    nparam = len(trafo)
    if nparam == 3:
        matrix = affine_matrix_3d(translation=trafo)
    elif nparam == 2:
        matrix = affine_matrix_2d(translation=trafo)
    else:
        raise ValueError(f"Invalid number of parameters for similarity transform: {nparam}")
    return matrix


def elastix_parameter_to_bdv_matrix(trafo: List[int], trafo_type: str) -> np.ndarray:
    """Convert elastix transformation parameters to affine matrix in bdv convention.

    Note that elastix uses a different convention than the native python or bdv order.

    Args:
        trafo: The transformation parameters.
        trafo_type: The transformation types.

    Returns:
        The affine matrix.
    """
    if trafo_type == "AffineTransform":
        matrix = _elastix_affine_to_bdv(trafo)
    elif trafo_type == "EulerTransform":
        matrix = _elastix_euler_to_bdv(trafo)
    elif trafo_type == "SimilarityTransform":
        matrix = _elastix_similarity_to_bdv(trafo)
    elif trafo_type == "TranslationTransform":
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

    def convert2d(trafo):
        # initialize the resulting affine matrix with the identity
        matrix = affine_matrix_2d()

        # load the rotation center from the elastix transformation definition
        rot_center = elastix_parser.get_rotation_center(trafo_file)
        if rot_center is not None:
            rot_center_neg = [-ce for ce in rot_center]

            # translate to the rotation center
            translate_to_rot = affine_matrix_2d(translation=rot_center_neg)
            matrix = translate_to_rot @ matrix

        # apply rotation and scale
        rot_and_scale = trafo.copy()
        rot_and_scale[2, :2] = 0
        matrix = rot_and_scale @ matrix

        # translate back from the rotation center
        if rot_center is not None:
            translate_from_rot = affine_matrix_2d(translation=rot_center)
            matrix = translate_from_rot @ matrix
        return matrix

    def convert3d(trafo):
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

    # go fron the transformation vector to affine matrix. we can use the
    # bdv functionality, because both bdv and elastix have the same axis convention
    trafo = elastix_parameter_to_bdv_matrix(trafo, trafo_type)

    # convert in 2d or 3d
    if trafo.shape[0] == 3:
        return convert2d(trafo)
    else:
        return convert3d(trafo)


def _combine_elastix_trafos_bdv(trafos, resolution, scale_factor):
    is_2d = trafos[0].shape[0] == 3

    # transformation to scale from voxel space to millimeter
    # (which is the fixed physical unit in elastix)
    if is_2d:  # 2d case
        vox_to_mm = affine_matrix_2d(scale=2 * [scale_factor])
    else:  # 3d case
        vox_to_mm = affine_matrix_3d(scale=3 * [scale_factor])

    # transformation to scale from millimiter to the physicial unit we use
    # usually we use micrometer and then scale_factor = 10^3
    # for nanometer it would be 10^6 etc.
    if is_2d:  # 2d case
        mm_to_unit = affine_matrix_2d(scale=[res / scale_factor for res in resolution])
    else:  # 3d case
        mm_to_unit = affine_matrix_3d(scale=[res / scale_factor for res in resolution])

    # combine the scaling transfomraions and the actual elastix transformations
    matrix = vox_to_mm
    for trafo in trafos:
        # elastix uses the opposite transformation direction as bdv,
        # so we need to invert the elastix transformation here
        matrix = matrix @ np.linalg.inv(trafo)
    matrix = matrix @ mm_to_unit

    return matrix


def _get_elastix_trafo_files(trafo_file, load_initial_trafos):
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

    # reverse the order of transformations
    return trafo_files[::-1]


def elastix_to_bdv(
    trafo_file: str,
    resolution: List[float],
    scale_factor: float = 1e3,
    load_initial_trafos: bool = True,
) -> List[float]:
    """Convert elastix transformation in text file to bdv transformation.

    Args:
        trafo_file: The file defining the elastix transformation.
        resolution: Resolution of the data in physical units.
        scale_factor: Scale factor of physical units compared to millimeter, which is
            the default unit for elastix tranformations. By default, assume that physical
            units is in micrometer, which corresponds to a scale of 10^3.
        load_initial_trafos: Whether to load the initial transformations.

    Returns:
        Parameter vector for bdv transformation.
    """

    # get elastix trafos in bdv matrix format
    trafo_files = _get_elastix_trafo_files(trafo_file, load_initial_trafos)
    trafos = [_convert_elastix_trafo(trafo) for trafo in trafo_files]

    # combine the transformations and apply the scaling transformations to
    # change from the elastix space (measured in millimeter)
    # to the bdv space (which maps the physical unit (usually micrometer) to voxels)
    # NOTE we need to reverse the resolution here to switch from ZYX to XYZ axis convention
    trafo = _combine_elastix_trafos_bdv(trafos, resolution[::-1], scale_factor)

    # return the transformation as parameter vector instead of affine matrix
    return matrix_to_parameters(trafo)


def elastix_to_native(
    trafo_file: str,
    resolution: List[float],
    scale_factor: float = 1e3,
    load_initial_trafos: bool = True
) -> np.ndarray:
    """Convert elastix transformation in text file to native transformation.

    Args:
        trafo_file: The file defining the elastix transformation.
        resolution: Resolution of the data in physical units.
        scale_factor: Scale factor of physical units compared to millimeter, which is
            the default unit for elastix tranformations. By default, assume that physical
            units is in micrometer, which corresponds to a scale of 10^3.
        load_initial_trafos: Whether to load the initial transformations.

    Returns:
        Affine transformation matrix in native format.
    """
    # get elastix trafo in bdv parameter format format
    trafo = elastix_to_bdv(trafo_file, resolution, scale_factor, load_initial_trafos)
    # convert to native format
    trafo = bdv_to_native(trafo, resolution, invert=True)
    return trafo


def _native_to_elastix_trafo(trafo, resolution=None):
    """Convert native transformation matrix to elastix transformation parameter.
    """
    params = 12 * [0]

    params[0] = trafo[2, 2]
    params[1] = trafo[2, 1]
    params[2] = trafo[2, 0]

    params[3] = trafo[1, 2]
    params[4] = trafo[1, 1]
    params[5] = trafo[1, 0]

    params[6] = trafo[0, 2]
    params[7] = trafo[0, 1]
    params[8] = trafo[0, 0]

    if resolution is None:
        params[9] = trafo[2, 3]
        params[10] = trafo[1, 3]
        params[11] = trafo[0, 3]
    else:
        params[9] = trafo[2, 3] * resolution[2]
        params[10] = trafo[1, 3] * resolution[1]
        params[11] = trafo[0, 3] * resolution[0]

    return params


#
# BigDataViewer
#

def bdv_to_native(
    trafo: List[float], resolution: Optional[List[float]] = None, invert: bool = True
) -> np.ndarray:
    """Convert bdv transformation parameters to affine matrix in native format.

    Bdv and elf expect the transformation in the opposite direction.
    So to be directly applied, the transformation also needs to be inverted,
    in addition to changing between the axis conventions.
    The bdv transformations often also include the transformation from voxel space to physical space.

    Args:
        trafo: Parameter vector of the bdv transformation.
        resolution: Physical resolution of the data. If given, the transformation will be scaled to voxel space.
        invert: Invert the resulting affine matrix. This is necessary to apply the affine matrix directly in elf.

    Returns:
        The affine transformation matrix.
    """

    if len(trafo) == 12:  # 3d case
        sub_matrix = np.zeros((3, 3), dtype="float64")
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

    elif len(trafo) == 6:  # 2d case
        sub_matrix = np.zeros((2, 2), dtype="float64")
        sub_matrix[0, 0] = trafo[4]
        sub_matrix[0, 1] = trafo[3]

        sub_matrix[1, 0] = trafo[1]
        sub_matrix[1, 1] = trafo[0]

        shift = [trafo[5], trafo[2]]

        matrix = np.zeros((3, 3))
        matrix[:2, :2] = sub_matrix
        matrix[:2, 2] = shift
        matrix[2, 2] = 1

    else:
        raise ValueError(f"Invalid number of parameters {len(trafo)}")

    # invert the matrix, because the transformation directions of bdv and
    # the native format are opposite. can be deactivated for debugging purposes.
    if invert:
        matrix = np.linalg.inv(matrix)

    # scale from physical resolution to voxels
    if resolution is not None:
        scale = affine_matrix_3d(scale=resolution) if len(trafo) == 12 else affine_matrix_2d(scale=resolution)
        matrix = matrix @ scale

    return matrix


def native_to_bdv(
    matrix: np.ndarray, resolution: Optional[List[float]] = None, invert: bool = True
) -> List[float]:
    """Convert affine matrix in native format to bdv transformation parameter vector.

    Bdv and elf expect the transformation in the opposite direction.
    So to be directly applied the transformation also needs to be inverted,
    in addition to changing between the axis conventions.
    The bdv transformations often also include the transformation from voxel space to physical space.

    Args:
        matrix: Native affine transformation matrix.
        resolution: Physical resolution of the data. If given, the transformation will be scaled to voxel space.
        invert: Invert the resulting affine matrix. This is necessary to apply the affine matrix directly in elf.

    Returns:
        Vector with transformation parameters.
    """
    # TODO include scaling transformation from physical space to voxel space
    if resolution is not None:
        raise NotImplementedError

    if invert:
        matrix = np.linalg.inv(matrix)

    if matrix.shape[0] == 4:
        trafo = [matrix[2, 2], matrix[2, 1], matrix[2, 0], matrix[2, 3],
                 matrix[1, 2], matrix[1, 1], matrix[1, 0], matrix[1, 3],
                 matrix[0, 2], matrix[0, 1], matrix[0, 0], matrix[0, 3]]
    else:
        trafo = [matrix[1, 1], matrix[1, 0], matrix[1, 2],
                 matrix[0, 1], matrix[0, 0], matrix[0, 2]]
    return trafo
