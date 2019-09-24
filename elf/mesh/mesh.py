import numpy as np
from skimage.measure import marching_cubes_lewiner

# nifty import for mesh smoothing support
try:
    import nifty
except ImportError:
    nifty = None

# TODO benchmark this compared to the skimage function and get
# rid of it if it does not provide significant speed-up
# import ilastik_marching_cubes for alternate smoothing implementation
try:
    from marching_cubes import march
except ImportError:
    march = None


def marching_cubes(obj, smoothing_iterations=0,
                   resolution=None, use_ilastik=False):
    """ Compute mesh via marching cubes.

    Arguments:
        obj [np.ndarray] - volume containing the object to be meshed
        smoothing_iterations [int] - number of mesh smoothing iterations (default: 0)
        resolution[listlike[int]] - resolution of the data (default: None)
        use_ilastik [bool] - whether to use the ilastk marching cubes implementation.
            Default is skimage (default: False)
    """
    resolution = (1., 1., 1.) if resolution is None else resolution
    if len(resolution) != 3:
        raise ValueError("Invalid resolution argument")
    resolution = tuple(resolution)

    if use_ilastik:
        if march is None:
            raise RuntimeError("Ilastik marching cubes implementation not found")
        if resolution is not None:
            raise RuntimeError("Ilastik marching cubes does not support resolution.")
        verts, normals, faces = march(obj.T, smoothing_iterations)
        verts = verts[:, ::-1]
    else:
        verts, faces, normals, _ = marching_cubes_lewiner(obj, spacing=resolution)
        if smoothing_iterations > 0:
            if nifty is None:
                raise RuntimeError("Need nifty to perform mesh smoothing")
            verts, normals = smooth_mesh(verts, normals, faces, smoothing_iterations)

    return verts, faces, normals


# couuld implement in nifty directly in order to speed this up
def smooth_mesh(verts, normals, faces, iterations):
    """ Smooth mesh surfacee via laplacian smoothing.

    Argumennts:
        verts [np.ndarray] - mesh vertices
        normals [np.ndarray] - mesh normals
        faces [np.ndarray] - mesh faces
        iterations [int] - number of smoothing rounds
    """
    n_verts = len(verts)
    g = nifty.graph.undirectedGraph(n_verts)

    edges = np.concatenate([faces[:, :2],
                            faces[:, 1:],
                            faces[:, ::2]], axis=0)
    g.insertEdges(edges)

    current_verts = verts
    current_normals = normals
    new_verts = np.zeros_like(verts, dtype=verts.dtype)
    new_normals = np.zeros_like(normals, dtype=normals.dtype)

    for it in range(iterations):
        for vert in range(n_verts):
            nbrs = np.array([vert] + [nbr[0] for nbr in g.nodeAdjacency(vert)],
                            dtype='int')
            new_verts[vert] = np.mean(current_verts[nbrs], axis=0)
            new_normals[vert] = np.mean(current_normals[nbrs], axis=0)
        current_verts = new_verts
        current_normals = new_normals

    return new_verts, new_normals
