import numpy as np
import nifty

#
# Parser for swc skeleton format
# http://research.mssm.edu/cnic/swc.html.


def read_swc(input_path, return_radius=False, return_type=False):
    """ Read skeleton stored in .swc

    For details on the swc format for skeletons, see
    http://research.mssm.edu/cnic/swc.html.
    This function expects the swc catmaid flavor.

    Arguments:
        input_path [str]: path to swc file
        retun_radius [bool]: return radius measurements (default: False)
        retun_type [bool]: return type variable (default: False)
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


def write_swc(output_path, nodes, edges, resolution=None, invert_coords=False):
    """ Write skeleton to .swc

    For details on the swc format for skeletons, see
    http://research.mssm.edu/cnic/swc.html.
    This writes the swc catmaid flavor.

    Arguments:
        output_path [str]: output_path for swc file
        skel_vol [np.ndarray]: binary volume containing the skeleton
        resolution [list or float]: pixel resolution (default: None)
        invert_coords [bool]: whether to invert the coordinates
            This may be useful because swc expects xyz, but input is zyx (default: False)
    """
    # map coords to resolution and invert if necessary
    if resolution is not None:
        if isinstance(resolution, float):
            resolution = 3 * [resolution]
        assert len(resolution) == 3, str(len(resolution))
        nodes *= np.array(resolution)
    if invert_nodes:
        nodes = nodes[:, ::-1]

    n_nodes = nodes.shape[0]
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(edges)

    # TODO if this becomes a bottle-neck think about moving to numba, cython or c++
    with open(output_path, 'w') as f:
        # swc: node-id
        #      type (hard-coded to 0 = undefined)
        #      coordinates
        #      radius (hard-coded to 0.0)
        #      parent id
        for node_id in range(n_nodes):

            # TODO I am not 100 % sure about this
            ngbs = [adj.first for adj in graph.nodeAdjacency(nodeId)]

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
            line = '%i 0 %f %f %f 0.0 %i \n' % (node_id, coord[0], coord[1], coord[2], parent)
            f.write(line)
