import time
import numpy as np
import vigra
import warnings

import nifty
from nifty import graph as ngraph
from nifty.graph import rag as nrag
from nifty.graph import accumulate_affinities_mean_and_length
from affogato.affinities import compute_affinities

from .features import compute_grid_graph_affinity_features


class AccumulatorLongRangeAffs:
    """@private
    """
    def __init__(self, offsets,
                 used_offsets=None,
                 offsets_weights=None,
                 verbose=True,
                 n_threads=-1,
                 invert_affinities=False,
                 statistic='mean',
                 offset_probabilities=None,
                 return_dict=False):

        offsets = check_offsets(offsets)

        # Parse inputs:
        if used_offsets is not None:
            assert len(used_offsets) < offsets.shape[0]
            if offset_probabilities is not None:
                offset_probabilities = np.require(offset_probabilities, dtype='float32')
                assert len(offset_probabilities) == len(offsets)
                offset_probabilities = offset_probabilities[used_offsets]
            if offsets_weights is not None:
                offsets_weights = np.require(offsets_weights, dtype='float32')
                assert len(offsets_weights) == len(offsets)
                offsets_weights = offsets_weights[used_offsets]

        self.offsets_probabilities = offset_probabilities
        self.used_offsets = used_offsets
        self.offsets_weights = offsets_weights

        self.used_offsets = used_offsets
        self.return_dict = return_dict
        self.statistic = statistic

        assert isinstance(n_threads, int)

        self.offsets = offsets
        self.verbose = verbose
        self.n_threads = n_threads
        self.invert_affinities = invert_affinities
        self.offset_probabilities = offset_probabilities

    def __call__(self, affinities, segmentation, affinities_weights=None):
        tick = time.time()

        # Use only few channels from the affinities, if we are not using all offsets:
        offsets = self.offsets
        offsets_weights = self.offsets_weights
        if self.used_offsets is not None:
            assert len(self.used_offsets) < self.offsets.shape[0]
            offsets = self.offsets[self.used_offsets]
            affinities = affinities[self.used_offsets]
            if affinities_weights is not None:
                affinities_weights = affinities_weights[self.used_offsets]

        assert affinities.ndim == 4
        assert affinities.shape[0] == offsets.shape[0]
        if affinities_weights is not None:
            assert affinities_weights.shape == affinities.shape
            assert offsets_weights is None, "Affinity and offset weights are not supported at the same time"

        if self.invert_affinities:
            affinities = 1. - affinities

        # Build rag and compute node sizes:
        if self.verbose:
            print("Computing rag...")
            tick = time.time()

        # If there was a label -1, now its value in the rag is given by the maximum label
        # (and it will be ignored later on)
        rag, extra_dict = get_rag(segmentation, self.n_threads)
        has_background_label = extra_dict['has_background_label']

        if self.verbose:
            print("Took {} s!".format(time.time() - tick))
            tick = time.time()

        out_dict = {}
        out_dict['rag'] = rag

        # Build graph including long-range connections:
        if self.verbose:
            print("Building (lifted) graph...")

        # -----------------------
        # Lifted edges:
        # -----------------------
        # Get rid of local offsets:
        is_direct_neigh_offset, indices_local_offsets = find_indices_direct_neighbors_in_offsets(offsets)
        lifted_offsets = offsets[np.logical_not(is_direct_neigh_offset)]

        add_lifted_edges = True
        if isinstance(self.offset_probabilities, np.ndarray):
            lifted_probs = self.offset_probabilities[np.logical_not(is_direct_neigh_offset)]
            # Check if we should add lifted edges at all:
            add_lifted_edges = any(lifted_probs != 0.)
            if add_lifted_edges:
                assert all(lifted_probs == 1.0), "Offset probabilities different from one are not supported" \
                                                 "when starting from a segmentation."

        lifted_graph, is_local_edge = build_lifted_graph_from_rag(
            rag,
            lifted_offsets,
            number_of_threads=self.n_threads,
            has_background_label=has_background_label,
            add_lifted_edges=add_lifted_edges
        )

        if self.verbose:
            print("Took {} s!".format(time.time() - tick))
            print("Computing edge_features...")
            tick = time.time()

        # TODO: here offsets_probs and offsets_weights are not consistent... (and make it support edge_mask)
        #   very likely I need to create a mask from probs if present and then use it for both (to keep it consistent)
        # Compute edge sizes and accumulate average:
        edge_indicators, edge_sizes = accumulate_affinities_mean_and_length(
            affinities,
            offsets,
            segmentation if not has_background_label else extra_dict['updated_segmentation'],
            graph=lifted_graph,
            offset_weights=offsets_weights,
            affinities_weights=affinities_weights,
            ignore_label=None if not has_background_label else extra_dict['background_label'],
            number_of_threads=self.n_threads
        )

        out_dict['graph'] = lifted_graph
        out_dict['edge_indicators'] = edge_indicators
        out_dict['edge_sizes'] = edge_sizes

        if not self.return_dict:
            edge_features = np.stack([edge_indicators, edge_sizes, is_local_edge])
            return lifted_graph, edge_features
        else:
            out_dict['is_local_edge'] = is_local_edge
            return out_dict


def get_rag(segmentation, nb_threads):
    """@private
    """

    """If the segmentation has values equal to -1, those are interpreted as background pixels.

    When this rag is build, the node IDs will be taken from segmentation and the background_node will have ID
    previous_max_label+1

    In `build_lifted_graph_from_rag`, the background node and all the edges connecting to it are ignored while creating
    the new (possibly lifted) undirected graph.
    """

    # Check if the segmentation has a background label that should be ignored in the graph:
    min_label = segmentation.min()
    if min_label >= 0:
        out_dict = {'has_background_label': False}
        return nrag.gridRag(segmentation.astype(np.uint32), numberOfThreads=nb_threads), out_dict
    else:
        assert min_label == -1, "The only accepted background label is -1"
        max_valid_label = segmentation.max()
        assert max_valid_label >= 0, "A label image with only background label was passed!"
        mod_segmentation = segmentation.copy()
        background_mask = segmentation == min_label
        mod_segmentation[background_mask] = max_valid_label + 1

        # Build rag including background:
        out_dict = {'has_background_label': True,
                    'updated_segmentation': mod_segmentation,
                    'background_label': max_valid_label + 1}
        return nrag.gridRag(mod_segmentation.astype(np.uint32), numberOfThreads=nb_threads), out_dict


def build_lifted_graph_from_rag(rag,
                                offsets,
                                number_of_threads=-1,
                                has_background_label=False,
                                add_lifted_edges=True):
    """@private
    """

    """If has_background_label is true, it assumes that it has label rag.numberOfNodes - 1 (See function `get_rag`)
    The background node and all the edges connecting to it are ignored when creating
    the new (possibly lifted) undirected graph.
    """
    # TODO: in order to support an edge_mask,
    # getting the lifted edges is the easy part, but then I also need to accumulate
    # affinities properly (and ignore those not in the mask)
    # TODO: add options `set_only_local_connections_as_mergeable`
    # similarly to `build_pixel_long_range_grid_graph_from_offsets`

    if not has_background_label:
        nb_local_edges = rag.numberOfEdges
        final_graph = rag
    else:
        # Find edges not connected to the background:
        edges = rag.uvIds()
        background_label = rag.numberOfNodes - 1
        valid_edges = edges[np.logical_and(edges[:, 0] != background_label, edges[:, 1] != background_label)]

        # Construct new graph without the background:
        new_graph = nifty.graph.undirectedGraph(rag.numberOfNodes - 1)
        new_graph.insertEdges(valid_edges)

        nb_local_edges = valid_edges.shape[0]
        final_graph = new_graph

    if not add_lifted_edges:
        return final_graph, np.ones((nb_local_edges,), dtype='bool')
    else:
        if not has_background_label:
            local_edges = rag.uvIds()
            final_graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
            final_graph.insertEdges(local_edges)

        # Find lifted edges:
        # Note that this function could return the same lifted edge multiple times, so I need to add them to the graph
        # to see how many will be actually added
        possibly_lifted_edges = ngraph.rag.compute_lifted_edges_from_rag_and_offsets(rag,
                                                                                     offsets,
                                                                                     numberOfThreads=number_of_threads)

        # Delete lifted edges connected to the background label:
        if has_background_label:
            possibly_lifted_edges = possibly_lifted_edges[
                np.logical_and(possibly_lifted_edges[:, 0] != background_label,
                               possibly_lifted_edges[:, 1] != background_label)]

        final_graph.insertEdges(possibly_lifted_edges)
        total_nb_edges = final_graph.numberOfEdges

        is_local_edge = np.zeros(total_nb_edges, dtype=np.int8)
        is_local_edge[:nb_local_edges] = 1

        return final_graph, is_local_edge


def edge_mask_from_offsets_prob(shape, offsets_probabilities, edge_mask=None):
    """@private
    """
    shape = tuple(shape) if not isinstance(shape, tuple) else shape

    offsets_probabilities = np.require(offsets_probabilities, dtype='float32')
    nb_offsets = offsets_probabilities.shape[0]
    edge_mask = np.ones((nb_offsets,) + shape, dtype='bool') if edge_mask is None else edge_mask
    assert (offsets_probabilities.min() >= 0.0) and (offsets_probabilities.max() <= 1.0)

    # Randomly sample some edges to add to the graph:
    edge_mask = []
    for off_prob in offsets_probabilities:
        edge_mask.append(np.random.random(shape) <= off_prob)
    edge_mask = np.logical_and(np.stack(edge_mask, axis=-1), edge_mask)

    return edge_mask


def from_foreground_mask_to_edge_mask(foreground_mask, offsets, mask_used_edges=None):
    """@private
    """
    _, valid_edges = compute_affinities(foreground_mask.astype('uint64'), offsets.tolist(), True, 0)

    if mask_used_edges is not None:
        return np.logical_and(valid_edges, mask_used_edges)
    else:
        return valid_edges.astype('bool')


# TODO use elf.features functionality instead
def build_pixel_long_range_grid_graph_from_offsets(image_shape,
                                                   offsets,
                                                   affinities,
                                                   offsets_probabilities=None,
                                                   mask_used_edges=None,
                                                   offset_weights=None,
                                                   set_only_direct_neigh_as_mergeable=True,
                                                   foreground_mask=None):
    """@private
    """
    # TODO: add support for foreground mask (masked nodes are removed from final undirected graph

    image_shape = tuple(image_shape) if not isinstance(image_shape, tuple) else image_shape
    offsets = check_offsets(offsets)

    if foreground_mask is not None:
        # Mask edges connected to background:
        mask_used_edges = from_foreground_mask_to_edge_mask(foreground_mask, offsets, mask_used_edges=mask_used_edges)

    # Create temporary grid graph:
    grid_graph = ngraph.undirectedGridGraph(image_shape)

    # Compute edge mask from offset probs:
    if offsets_probabilities is not None:
        if mask_used_edges is not None:
            warnings.warn("!!! Warning: both edge mask and offsets probabilities were used!!!")
        mask_used_edges = edge_mask_from_offsets_prob(image_shape, offsets_probabilities, mask_used_edges)

    uv_ids, edge_weights = compute_grid_graph_affinity_features(grid_graph, affinities,
                                                                offsets=offsets, mask=mask_used_edges)

    nb_nodes = grid_graph.numberOfNodes
    projected_node_ids_to_pixels = grid_graph.projectNodeIdsToPixels()

    if foreground_mask is not None:
        # Mask background nodes and relabel node ids continuous before to create final graph:
        projected_node_ids_to_pixels += 1
        projected_node_ids_to_pixels[np.invert(foreground_mask)] = 0
        projected_node_ids_to_pixels, new_max_label, mapping = vigra.analysis.relabelConsecutive(
            projected_node_ids_to_pixels,
            keep_zeros=True)
        # The following assumes that previously computed edges has alreadyu been masked
        uv_ids += 1
        vigra.analysis.applyMapping(uv_ids, mapping, out=uv_ids)
        nb_nodes = new_max_label + 1

    # Create new undirected graph with all edges (including long-range ones):
    graph = ngraph.UndirectedGraph(nb_nodes)
    graph.insertEdges(uv_ids)

    # By default every edge is local/mergable:
    is_local_edge = np.ones(graph.numberOfEdges, dtype='bool')
    if set_only_direct_neigh_as_mergeable:
        # Get edge ids of local edges:
        # Warning: do not use grid_graph.projectEdgeIdsToPixels because edges
        # ids could be inconsistent with those created
        # with compute_grid_graph_affinity_features assuming the given offsets!
        is_dir_neighbor, _ = find_indices_direct_neighbors_in_offsets(offsets)
        projected_local_edge_ids = grid_graph.projectEdgeIdsToPixelsWithOffsets(np.array(offsets))[is_dir_neighbor]
        is_local_edge = np.isin(np.arange(edge_weights.shape[0]),
                                projected_local_edge_ids[projected_local_edge_ids != -1].flatten(),
                                assume_unique=True)

    edge_sizes = np.ones(graph.numberOfEdges, dtype='float32')
    # TODO: use np.unique or similar on edge indices, but only if offset_weights are given (expensive)
    # Get local edges:
    # grid_graph.projectEdgeIdsToPixels()
    assert offset_weights is None, "Not implemented yet"

    return graph, projected_node_ids_to_pixels, edge_weights, is_local_edge, edge_sizes


def check_offsets(offsets):
    """@private
    """
    if isinstance(offsets, (list, tuple)):
        offsets = np.array(offsets)
    else:
        assert isinstance(offsets, np.ndarray)
    assert offsets.ndim == 2
    return offsets


def find_indices_direct_neighbors_in_offsets(offsets):
    """@private
    """
    offsets = check_offsets(offsets)
    indices_dir_neighbor = []
    is_dir_neighbor = np.empty(offsets.shape[0], dtype='bool')
    for i, off in enumerate(offsets):
        is_dir_neighbor[i] = (np.abs(off).sum() == 1)
        if np.abs(off).sum() == 1:
            indices_dir_neighbor.append(i)
    return is_dir_neighbor, indices_dir_neighbor
