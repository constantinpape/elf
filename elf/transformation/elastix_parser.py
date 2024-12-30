import os
from typing import List, Optional

ELASTIX_TRAFO_TYPES = {
    "TranslationTransform": [3, 2],
    "EulerTransform": [6, 3],
    "SimilarityTransform": [7, 4],
    "AffineTransform": [12, 6],
    "BSplineTransform": None
}
"""@private
"""
AFFINE_COMPATIBLE = ["TranslationTransform",
                     "EulerTransform",
                     "SimilarityTransform",
                     "AffineTransform"]
"""@private
"""


def get_transformation_type(transform_file: str) -> Optional[str]:
    """Read the transformation type from elastix transformation file.

    Args:
        transformation_file: Filepath to the elastix transformation.

    Returns:
        The transformation type, None if the file is not valid.
    """

    trafo_type = None
    n_params = None
    with open(transform_file) as f:
        for line in f:
            if line.startswith("(Transform") and not line.startswith("(TransformParameters"):
                trafo_type = line.rstrip("\n")[1:-1].split()[1]
                trafo_type = trafo_type[1:-1]
            if line.startswith("(NumberOfParameters"):
                n_params = line.rstrip("\n")[1:-1].split()[1]
                n_params = int(n_params)

    if trafo_type is None:
        return None
    elif trafo_type not in ELASTIX_TRAFO_TYPES:
        return None
    else:
        n_params_exp = ELASTIX_TRAFO_TYPES[trafo_type]
        if n_params_exp is not None and n_params not in n_params_exp:
            return None
        else:
            return trafo_type


def get_transformation(transform_file: str) -> List[float]:
    """Read the transformation parameters from elastix transformation file.

    Args:
        transformation_file: Filepath to the elastix transformation.

    Returns:
        The transformation parameters.
    """
    trafo_type = get_transformation_type(transform_file)
    if trafo_type is None:
        raise ValueError(f"Invalid transformation file {transform_file}")

    exp_n_params = ELASTIX_TRAFO_TYPES[trafo_type]
    params = None
    with open(transform_file) as f:
        for line in f:
            if line.startswith("(TransformParameters"):
                params = line.rstrip("\n")[1:-1].split()
                params = [float(param) for param in params[1:]]
                n_params = len(params)
                if exp_n_params is not None and n_params not in exp_n_params:
                    raise ValueError(f"Invalid transformation file {transform_file}")

    if params is None:
        raise ValueError(f"Invalid transformation file {transform_file}")

    return params


def get_initial_transform_file(transform_file: str) -> str:
    """Get the name of the file with the initial transformation.

    Args:
        transformation_file: Filepath to the elastix transformation.

    Returns:
        The filepath to the initial transformation file, None if not present or if it cannot be found.
    """
    file_name = None
    with open(transform_file) as f:
        for line in f:
            if line.startswith("(InitialTransformParametersFileName"):
                file_name = line.rstrip("\n")[2:-2].split()[1]

    if file_name is None or file_name == "\"NoInitialTransform":
        return None

    alt_file_name = os.path.join(os.path.split(transform_file)[0],
                                 os.path.split(file_name)[1])
    if os.path.exists(file_name):
        return file_name
    elif os.path.exists(alt_file_name):
        return alt_file_name
    else:
        msg = f"Could not find the initial trafo file {file_name} specified in {transform_file}"
        raise RuntimeError(msg)


def get_shape(transform_file):
    """@private
    """
    with open(transform_file) as f:
        for line in f:
            if line.startswith("(Size"):
                shape = line.rstrip("\n")[1:-1].split()[1:]
    shape = [int(sh) for sh in shape]
    return shape


def get_rotation_center(transform_file):
    """@private
    """
    rot_center = None
    with open(transform_file) as f:
        for line in f:
            if line.startswith("(CenterOfRotationPoint"):
                rot_center = line.rstrip("\n")[1:-1].split()[1:]
                rot_center = [float(ce) for ce in rot_center]
    return rot_center


def get_resolution(transform_file, to_um=False):
    """@private
    """
    with open(transform_file) as f:
        for line in f:
            if line.startswith("(Spacing"):
                resolution = line.rstrip("\n")[1:-1].split()[1:]
                resolution = [float(res) for res in resolution]
    if to_um:
        resolution = [res * 1000. for res in resolution]
    return resolution
