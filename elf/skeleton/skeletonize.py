from .methods import thinning

# TODO get a teasar impl
METHODS = {'thinning': thinning}
DEFAULT_PARAMS = {}


def get_method_names():
    return list(METHODS.keys())


def get_method_params(name):
    return DEFAULT_PARAMS.get(name, {})


def skeletonize(obj, resolution=None, boundary_distances=None,
                method='thinning', **method_params):
    """ Skeletonize object defined by binary mask.

    Arguments:
        obj [np.ndarray] - binary object mask
        resolution [int, float or list] - size of the voxels in physical units,
            can be list for anisotropic input (default: None)
        boundary_distances [np.ndarray] - distance to object boundaries
            can be pre-computed for teasar (default: None)
        method [str] - method used for skeletonization (default: thinning)
        method_params [kwargs] - parameter for skeletonization method.
    """
    impl = METHODS.get(method, None)
    if impl is None:
        raise ValueError("Inalid method %s, expect one of %s" % (method,
                                                                 ", ".join(get_method_names())))
    params = DEFAULT_PARAMS.get(method, {})
    params.update(method_params)

    if resolution is None:
        resolution = [1, 1, 1]
    if isinstance(resolution, int):
        resolution = 3 * [resolution]

    nodes, edges = impl(obj, resolution, boundary_distances, **params)
    return nodes, edges
