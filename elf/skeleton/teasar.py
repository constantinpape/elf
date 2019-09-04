import numpy as np
import nifty.skeletons as nskel
from scipy.ndimage import distance_transform_edt


def teasar(obj, resolution, boundary_distances=None, max_num_paths=None,
           penalty_scale=1000, penalty_exponent=16):
    """
    Skeletonize object with teasar.

    Introduced in "TEASAR: Tree-structured Extraction Algoriyhm for Accurate and Robust Skeletons":
    https://www3.cs.stonybrook.edu/~bender/pub/PG2000_teasar.pdf

    Arguments:
        obj [np.ndarray] - binary object mask
        resolution [list] - size of the voxels in physical unit
        boundary_distances [np.ndarray] - pre-compted boundary distance map (default: None)
    """

    max_num_paths = np.iinfo('uint32').max if max_num_paths is None else max_num_paths

    skel = Teasar(obj, resolution, boundary_distances)
    paths = []
    while len(paths) < max_num_paths:
        path = skel.get_next_path()
        if path is None:
            return get_nodes_and_edges_from_paths(paths)
        paths.append(path)
    return get_nodes_and_edges_from_paths(paths)


def get_nodes_and_edges_from_paths(paths):
    pass


class Teasar:
    """ Implement computation steps of TEASAR algorithm.
    """
    def __init__(self, obj, resolution, boundary_distances=None,
                 penalty_scale=5000, penalty_exponent=16):
        self.obj = obj
        self.resolution = list(resolution)
        self.boundary_distances = distance_transform_edt(obj, sampling=resolution) if\
            boundary_distances is None else boundary_distances
        self.root_node = self.find_root_node()
        self.distances = self.compute_distance_field(self.root_node)
        self.penalized_distances = self.compute_penalized_distances(penalty_scale, penalty_exponent)
        # TODO array to keep track of invalidated labels

    @property
    def shape(self):
        return self.obj.shape

    def get_path(self, src, target):
        src_coord = src if isinstance(src, tuple) else np.unravel_index(src, self.shape)
        trgt_coord = target if isinstance(target, tuple) else np.unravel_index(target, self.shape)
        path = nskel.dijkstra(self.penalized_distances, list(src_coord), list(trgt_coord))
        return path

    # TODO implement
    def get_next_path(self):
        pass

    def compute_penalized_distances(self, scale, exponent):
        distance_norm = (self.boundary_distances.max()**1.01)
        penalized_distances = scale * (1 - self.boundary_distances / distance_norm) ** exponent +\
            self.distances / self.distances.max()
        return penalized_distances

    def compute_distance_field(self, node):
        coord = np.unravel_index(node, self.shape)
        distances = nskel.euclidean_distance(self.obj, list(coord), self.resolution)
        return distances

    def find_root_node(self):
        """ Find the root node and return its id (= raveled coordinate).
        Here, we take the furthest node from the node with maximal boundary distance.
        """

        # find the max boundary distance node
        max_node = np.argmax(self.boundary_distances)

        # compute the distance field from the max node and return its argmaximum
        distances = self.compute_distance_field(max_node)
        root_node = np.argmax(distances)
        return root_node

    def get_pathlength(self, path):
        return nskel.pathlength(list(self.obj.shape), path, self.resolution)
