from typing import Optional, Tuple, Union

import numpy as np
import nifty

#
# Parser for swc skeleton format
# http://research.mssm.edu/cnic/swc.html.


def read_swc(
    input_path: str, return_radius: bool = False, return_type: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read skeleton stored in swc format.

    For details on the swc format for skeletons, see
    http://research.mssm.edu/cnic/swc.html.
    This function expects the swc catmaid flavor.

    Args:
        input_path: Path to the swc file.
        retun_radius: Return radius measurements.
        retun_type: Return type variable.

    Returns:
        The skeleton ids.
        The skeleton coordinates.
        The skeleton parents.
    """
    ids, coords, parents = [], [], []
    radii, types = [], []
    # open file and get outputs
    with open(input_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            # skip headers or break
            if line.startswith('#') or line == '':
                continue

            # parse this line
            values = line.split()
            # extract coordinate, node-id and parent-id
            coords.append([float(val) for val in values[2:5]])
            ids.append(int(values[0]))
            parents.append(int(values[-1]))

            # extract radius
            if return_radius:
                radii.append(float(values[5]))

            # extract type
            if return_type:
                types.append(int(values[1]))

    # TODO return edges instead of parents
    if return_radius:
        return ids, coords, parents, radii
    if return_type:
        return ids, coords, parents, types
    if return_radius and return_type:
        return ids, coords, parents, radii, types
    return ids, coords, parents


def write_swc(
    output_path: str,
    nodes: np.ndarray,
    edges: np.ndarray,
    resolution: Optional[Union[float, Tuple[float, float, float]]] = None,
    invert_coords: bool = False,
):
    """Write skeleton to a swc file.

    For details on the swc format for skeletons, see
    http://research.mssm.edu/cnic/swc.html.
    This writes the swc catmaid flavor.

    Args:
        output_path: Path to the output swc file.
        nodes: The coordinates of the skeleton trace.
        edges: The edges between skeleton nodes.
        resolution: Pixel resolution.
        invert_coords: Whether to invert the coordinates.
            To switch between xyz (expected by swc) and zyx (numpy default) order.
    """
    # map coords to resolution and invert if necessary
    if resolution is not None:
        if isinstance(resolution, float):
            resolution = 3 * [resolution]
        assert len(resolution) == 3, str(len(resolution))
        nodes *= np.array(resolution)
    if invert_coords:
        nodes = nodes[:, ::-1]

    n_nodes = nodes.shape[0]
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(edges)
    # swc format per node
    # node-id
    # type (hard-coded to 0 = undefined here)
    # coordinates
    # radius (hard-coded to 0.0 here)
    # parent id

    # Implement in numba (can it handle nifty? otherwise use different graph impl)?
    with open(output_path, "w") as f:
        for node_id in range(n_nodes):

            ngbs = [adj[0] for adj in graph.nodeAdjacency(node_id)]

            # only a single neighbor -> terminal node and no parent
            # also, for some reasons ngbs can be empty
            if len(ngbs) in (0, 1):
                parent = -1
            # two neighbors -> path node
            # more than two neighbors -> junction
            else:
                # TODO can we just assume that we get consistent output if we set parent to min ???
                parent = np.min(ngbs)
            coord = nodes[node_id]
            line = "%i 0 %f %f %f 0.0 %i \n" % (node_id, coord[0], coord[1], coord[2], parent)
            f.write(line)
