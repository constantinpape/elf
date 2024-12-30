from typing import Optional, Tuple, Union

import numpy as np

from .thinning import thinning

# Add teasar implementation?
METHODS = {"thinning": thinning}
DEFAULT_PARAMS = {}


def get_method_names():
    """@private
    """
    return list(METHODS.keys())


def get_method_params(name):
    """@private
    """
    return DEFAULT_PARAMS.get(name, {})


def skeletonize(
    obj: np.ndarray,
    resolution: Optional[Union[Tuple[float, ...], float]] = None,
    boundary_distances: Optional[np.ndarray] = None,
    method: str = "thinning",
    **method_params
) -> Tuple[np.ndarray, np.ndarray]:
    """Skeletonize an object defined by a binary mask.

    Args:
        obj: Binary object mask.
        resolution: Size of the voxels in physical units, can be a tuple for anisotropic input.
        boundary_distances: Distance to object boundaries.
        method: Method used for skeletonization.
        method_params: Parameter for skeletonization method.

    Returns:
        The nodes of the skeleton.
        The edges between skeleton nodes.
    """
    impl = METHODS.get(method, None)
    if impl is None:
        raise ValueError(f"Inalid method {method}, expect one of {', '.join(get_method_names())}")
    params = DEFAULT_PARAMS.get(method, {})
    params.update(method_params)

    ndim = obj.ndim
    if resolution is None:
        resolution = ndim * (1,)
    if isinstance(resolution, int):
        resolution = ndim * (resolution,)

    nodes, edges = impl(obj, resolution, boundary_distances, **params)
    return nodes, edges
