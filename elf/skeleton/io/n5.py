import numpy as np


def read_n5(ds, skel_id):
    """ Read skeleton stored in custom n5-based format

    The skeleton data is stored via varlen chunks: each chunk contains
    the data for one skeleton and stores:
    (n_skel_points, coord_z_0, coord_y_0, coord_x_0, ..., coord_z_n, coord_y_n, coord_x_n,
     n_edges, edge_0_u, edge_0_v, ..., edge_n_u, edge_n_v)

    Arguments:
        ds [z5py.Dataset]: input dataset
        skel_id [int]: id of the object corresponding to the skeleton
    """
    # read data from chunk
    data = ds.read_chunk((skel_id,))

    # check if the chunk is empty
    if data is None:
        return None, None

    # read number of points and coordinates
    n_points = data[0]
    offset = 1
    coord_len = int(3 * n_points)
    nodes = data[offset:offset+coord_len].reshape((n_points, 3))
    offset += coord_len
    # read number of edges and edges
    n_edges = data[offset]
    offset += 1
    edge_len = int(2 * n_edges)
    assert len(data) == offset + edge_len, "%i, %i" % (len(data), offset + edge_len)
    edges = data[offset:offset+edge_len].reshape((n_edges, 2))
    return nodes, edges


def write_n5(ds, skel_id, nodes, edges, coordinate_offset=None):
    """ Write skeleton to custom n5-based format

    The skeleton data is stored via varlen chunks: each chunk contains
    the data for one skeleton and stores:
    [n_skel_points, coord_z_0, coord_y_0, coord_x_0, ..., coord_z_n, coord_y_n, coord_x_n,
     n_edges, edge_0_u, edge_0_v, ..., edge_n_u, edge_n_v]

    Arguments:
        ds [z5py.Dataset]: output dataset
        skel_id [int]: id of the object corresponding to the skeleton
        nodes [np.ndarray]: node coordinates
        edges [np.ndarray]: edges
        coordinate_offset [listlike]: offset to coordinate (default: None)
    """
    # check if we have offset and add up if we do
    if coordinate_offset is not None:
        assert len(coordinate_offset) == 3
        nodes += np.array(coordinate_offset)

    # make serialization for number of points and coordinates
    n_points = nodes.shape[0]
    data = [np.array([n_points]), nodes.flatten()]

    # add number of edges and edges to the serialization
    n_edges = len(edges)
    data.extend([np.array([n_edges]), edges.flatten()])

    data = np.concatenate(data, axis=0)
    ds.write_chunk((skel_id,), data.astype('uint64'), True)
