import nifty
import numpy as np
from skimage.measure import marching_cubes as marching_cubes_lewiner


def marching_cubes(obj, smoothing_iterations=0, resolution=None):
    """Compute mesh via marching cubes.

    This is a wrapper around the skimage marching cubes implementation that provides
    additional mesh smoothing

    Arguments:
        obj [np.ndarray] - volume containing the object to be meshed
        smoothing_iterations [int] - number of mesh smoothing iterations (default: 0)
        resolution[listlike[int]] - resolution of the data (default: None)
    """
    resolution = (1., 1., 1.) if resolution is None else resolution
    if len(resolution) != 3:
        raise ValueError(f"Invalid resolution argument: {resolution}")
    resolution = tuple(resolution)

    verts, faces, normals, _ = marching_cubes_lewiner(obj, spacing=resolution)
    if smoothing_iterations > 0:
        verts, normals = smooth_mesh(verts, normals, faces, smoothing_iterations)

    return verts, faces, normals


def smooth_mesh(verts, normals, faces, iterations):
    """ Smooth mesh surfacee via laplacian smoothing.

    Arguments:
        verts [np.ndarray] - mesh vertices
        normals [np.ndarray] - mesh normals
        faces [np.ndarray] - mesh faces
        iterations [int] - number of smoothing rounds
    """
    n_verts = len(verts)
    g = nifty.graph.undirectedGraph(n_verts)

    edges = np.concatenate([faces[:, :2], faces[:, 1:], faces[:, ::2]], axis=0)
    g.insertEdges(edges)

    current_verts = verts
    current_normals = normals
    new_verts = np.zeros_like(verts, dtype=verts.dtype)
    new_normals = np.zeros_like(normals, dtype=normals.dtype)

    # TODO implement this directly in nifty for speed up
    for it in range(iterations):
        for vert in range(n_verts):
            nbrs = np.array([vert] + [nbr[0] for nbr in g.nodeAdjacency(vert)], dtype="int")
            new_verts[vert] = np.mean(current_verts[nbrs], axis=0)
            new_normals[vert] = np.mean(current_normals[nbrs], axis=0)
        current_verts = new_verts
        current_normals = new_normals

    return new_verts, new_normals
