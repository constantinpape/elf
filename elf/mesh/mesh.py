from typing import Optional, Tuple

import bioimage_cpp as bic
import numpy as np
from skimage.measure import marching_cubes as marching_cubes_impl


def marching_cubes(
    obj: np.ndarray,
    smoothing_iterations: int = 0,
    resolution: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mesh via marching cubes.

    This is a wrapper around the skimage marching cubes implementation that provides
    additional mesh smoothing.

    Args:
        obj: Volume containing the object to be meshed.
        smoothing_iterations: Number of mesh smoothing iterations.
        resolution: Resolution of the data.

    Returns:
        The vertices of the mesh.
        The faces of the mesh.
        The normals of the mesh.
    """
    resolution = (1.0, 1.0, 1.0) if resolution is None else resolution
    if len(resolution) != 3:
        raise ValueError(f"Invalid resolution argument: {resolution}")
    resolution = tuple(resolution)

    verts, faces, normals, _ = marching_cubes_impl(obj, spacing=resolution)
    if smoothing_iterations > 0:
        verts, normals = smooth_mesh(verts, normals, faces, smoothing_iterations)

    return verts, faces, normals


def smooth_mesh(
    verts: np.ndarray, normals: np.ndarray, faces: np.ndarray, iterations: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth mesh surface via laplacian smoothing.

    Args:
        verts: The mesh vertices.
        normals: The mesh normals.
        faces: The mesh faces.
        iterations: The number of smoothing iterations.

    Returns:
        The vertices after smoothing.
        The normals after smoothing.
    """
    return bic.utils.smooth_mesh(verts, normals, faces, iterations)
