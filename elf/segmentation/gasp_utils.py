import time
import warnings

import bioimage_cpp as bic
import numpy as np

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
            raise NotImplementedError(
                "Per-pixel affinity weights are not supported by the bioimage-cpp accumulator."
            )
        if offsets_weights is not None:
            raise NotImplementedError(
                "Per-offset weights are not supported by the bioimage-cpp accumulator."
            )

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
        rag_segmentation = extra_dict['updated_segmentation'] if has_background_label else segmentation

        if self.verbose:
            print("Took {} s!".format(time.time() - tick))
            tick = time.time()

        out_dict = {}
        out_dict['rag'] = rag

        # Build graph including long-range connections:
        if self.verbose:
            print("Building (lifted) graph...")

        # Determine whether to add lifted edges based on offsets and probabilities.
        is_direct_neigh_offset, _ = find_indices_direct_neighbors_in_offsets(offsets)
        add_lifted_edges = True
        if isinstance(self.offset_probabilities, np.ndarray):
            lifted_probs = self.offset_probabilities[np.logical_not(is_direct_neigh_offset)]
            add_lifted_edges = any(lifted_probs != 0.)
            if add_lifted_edges:
                assert all(lifted_probs == 1.0), "Offset probabilities different from one are not supported" \
                                                 "when starting from a segmentation."

        lifted_graph, is_local_edge = build_lifted_graph_from_rag(
            rag,
            rag_segmentation,
            offsets,
            number_of_threads=self.n_threads,
            has_background_label=has_background_label,
            add_lifted_edges=add_lifted_edges,
        )

        if self.verbose:
            print("Took {} s!".format(time.time() - tick))
            print("Computing edge_features...")
            tick = time.time()

        # Accumulate (mean, size) features per edge:
        edge_indicators, edge_sizes = _accumulate_affinities_mean_and_length(
            rag, rag_segmentation, affinities, offsets,
            lifted_uvs=_lifted_uvs_from_graph(rag, lifted_graph),
            local_edge_mask=_local_edge_mask(rag, has_background_label),
            n_threads=self.n_threads,
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


def _local_edge_mask(rag, has_background_label):
    """@private

    Boolean mask over RAG edges that selects edges not touching the background node.
    """
    uvs = np.asarray(rag.uv_ids())
    if not has_background_label:
        return np.ones(uvs.shape[0], dtype=bool)
    bg_label = int(rag.number_of_nodes) - 1
    return ~((uvs[:, 0] == bg_label) | (uvs[:, 1] == bg_label))


def _lifted_uvs_from_graph(rag, lifted_graph):
    """@private

    Lifted-edge uv ids contained in `lifted_graph` but not in `rag`. The lifted graph
    was built by inserting RAG edges first, then lifted edges, so the suffix is exactly
    the lifted set.
    """
    n_rag_edges = int(rag.number_of_edges)
    n_total = int(lifted_graph.number_of_edges)
    if n_total <= n_rag_edges:
        return np.zeros((0, 2), dtype="uint64")
    return np.asarray(lifted_graph.uv_ids(), dtype="uint64")[n_rag_edges:]


def _accumulate_affinities_mean_and_length(rag, segmentation, affinities, offsets,
                                           lifted_uvs, local_edge_mask, n_threads):
    """@private

    Accumulate (mean, size) per edge. Local features come from
    ``bic.graph.features.affinity_features`` (all offsets that land on a RAG edge),
    lifted features from ``bic.graph.features.lifted_affinity_features`` (long-range
    only). RAG edges touching the background node are dropped via ``local_edge_mask``.
    """
    nthr = 0 if n_threads is None or n_threads < 0 else int(n_threads)
    local_feats = bic.graph.features.affinity_features(
        rag, segmentation, affinities, list(map(list, offsets)), number_of_threads=nthr,
    )
    local_feats = local_feats[local_edge_mask]

    if lifted_uvs.shape[0] > 0:
        lifted_feats = bic.graph.features.lifted_affinity_features(
            segmentation, affinities, list(map(list, offsets)), lifted_uvs, number_of_threads=nthr,
        )
        feats = np.concatenate([local_feats, lifted_feats], axis=0)
    else:
        feats = local_feats

    return feats[:, 0].astype("float32"), feats[:, 1].astype("float32")


def get_rag(segmentation, nb_threads):
    """@private

    If the segmentation has values equal to -1, those are interpreted as background pixels.

    When this rag is built, the node IDs are taken from segmentation and the background node has ID
    previous_max_label + 1.

    In `build_lifted_graph_from_rag`, the background node and all the edges connecting to it are
    ignored while creating the new (possibly lifted) undirected graph.
    """
    nthr = 0 if nb_threads is None or nb_threads < 0 else int(nb_threads)
    min_label = segmentation.min()
    if min_label >= 0:
        rag = bic.graph.region_adjacency_graph(segmentation.astype(np.uint32), number_of_threads=nthr)
        return rag, {'has_background_label': False}

    assert min_label == -1, "The only accepted background label is -1"
    max_valid_label = segmentation.max()
    assert max_valid_label >= 0, "A label image with only background label was passed!"
    mod_segmentation = segmentation.copy()
    background_mask = segmentation == min_label
    mod_segmentation[background_mask] = max_valid_label + 1

    out_dict = {'has_background_label': True,
                'updated_segmentation': mod_segmentation.astype(np.uint32),
                'background_label': int(max_valid_label + 1)}
    rag = bic.graph.region_adjacency_graph(out_dict['updated_segmentation'], number_of_threads=nthr)
    return rag, out_dict


def build_lifted_graph_from_rag(rag,
                                segmentation,
                                offsets,
                                number_of_threads=-1,
                                has_background_label=False,
                                add_lifted_edges=True):
    """@private

    If has_background_label is true, the background node has label rag.number_of_nodes - 1
    (see `get_rag`). The background node and all the edges connecting to it are ignored when
    creating the new (possibly lifted) undirected graph.
    """
    nthr = 0 if number_of_threads is None or number_of_threads < 0 else int(number_of_threads)
    local_uvs = np.asarray(rag.uv_ids(), dtype="uint64")
    local_mask = _local_edge_mask(rag, has_background_label)
    local_uvs = local_uvs[local_mask]
    n_local_edges = local_uvs.shape[0]

    n_nodes = int(rag.number_of_nodes) - (1 if has_background_label else 0)

    if not add_lifted_edges:
        final_graph = bic.graph.UndirectedGraph.from_edges(n_nodes, local_uvs)
        return final_graph, np.ones((n_local_edges,), dtype='int8')

    lifted_uvs = bic.graph.features.lifted_edges_from_affinities(
        rag, segmentation, list(map(list, offsets)), number_of_threads=nthr,
    )
    lifted_uvs = np.asarray(lifted_uvs, dtype="uint64")
    if has_background_label:
        bg_label = int(rag.number_of_nodes) - 1
        lifted_mask = ~((lifted_uvs[:, 0] == bg_label) | (lifted_uvs[:, 1] == bg_label))
        lifted_uvs = lifted_uvs[lifted_mask]

    if lifted_uvs.shape[0] == 0:
        final_graph = bic.graph.UndirectedGraph.from_edges(n_nodes, local_uvs)
        is_local_edge = np.ones(final_graph.number_of_edges, dtype=np.int8)
        return final_graph, is_local_edge

    all_uvs = np.concatenate([local_uvs, lifted_uvs], axis=0)
    final_graph = bic.graph.UndirectedGraph.from_edges(n_nodes, all_uvs)

    total_nb_edges = int(final_graph.number_of_edges)
    is_local_edge = np.zeros(total_nb_edges, dtype=np.int8)
    is_local_edge[:n_local_edges] = 1
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
    _, valid_edges = bic.affinities.compute_affinities(
        foreground_mask.astype('uint64'), list(map(list, offsets)), ignore_label=0,
    )

    if mask_used_edges is not None:
        return np.logical_and(valid_edges, mask_used_edges)
    return valid_edges.astype('bool')


def _masked_grid_affinity_features(grid_graph, affinities, offsets, mask):
    """@private

    Compute aggregated (mean) edge weights for a grid graph using a per-channel
    edge mask. `mask` has shape (n_offsets, *spatial); only pixel-pair entries
    where the mask is True contribute. Returns (uv_ids, weights).
    """
    shape = tuple(grid_graph.shape)
    n_nodes = int(grid_graph.number_of_nodes)
    node_ids = np.arange(n_nodes, dtype="uint64").reshape(shape)
    offsets_arr = check_offsets(offsets)
    is_dir, _ = find_indices_direct_neighbors_in_offsets(offsets_arr)

    # Local edges via the grid graph's projection (direct-neighbor offsets only).
    dir_idx = np.where(is_dir)[0]
    if dir_idx.size > 0:
        dir_offsets = offsets_arr[dir_idx]
        dir_affs = affinities[dir_idx]
        dir_mask = mask[dir_idx]
        proj, _ = grid_graph.project_edge_ids_to_pixels_with_offsets(dir_offsets, mask=dir_mask.astype(bool))
        valid = proj != -1
        ids = proj[valid].astype("int64")
        w = dir_affs[valid].astype("float64")
        if ids.size > 0:
            uniq, inv = np.unique(ids, return_inverse=True)
            sums = np.bincount(inv, weights=w)
            cnts = np.bincount(inv)
            mean_local = (sums / cnts).astype("float32")
            grid_uvs = np.asarray(grid_graph.uv_ids(), dtype="uint64")
            local_uvs = grid_uvs[uniq]
        else:
            local_uvs = np.zeros((0, 2), dtype="uint64")
            mean_local = np.zeros((0,), dtype="float32")
    else:
        local_uvs = np.zeros((0, 2), dtype="uint64")
        mean_local = np.zeros((0,), dtype="float32")

    # Lifted edges from non-direct-neighbor offsets, aggregated manually.
    lifted_uvs_list, lifted_w_list = [], []
    for ci in np.where(~is_dir)[0]:
        off = offsets_arr[ci]
        src_slice = tuple(slice(max(0, -o), shape[d] - max(0, o)) for d, o in enumerate(off))
        dst_slice = tuple(slice(max(0, o), shape[d] - max(0, -o)) for d, o in enumerate(off))
        sub_mask = mask[ci][src_slice].astype(bool)
        if not sub_mask.any():
            continue
        u = node_ids[src_slice][sub_mask]
        v = node_ids[dst_slice][sub_mask]
        w = affinities[ci][src_slice][sub_mask]
        u_min = np.minimum(u, v)
        v_max = np.maximum(u, v)
        lifted_uvs_list.append(np.column_stack([u_min, v_max]).astype("uint64"))
        lifted_w_list.append(w.astype("float32"))

    if lifted_uvs_list:
        all_lifted = np.concatenate(lifted_uvs_list, axis=0)
        all_lifted_w = np.concatenate(lifted_w_list).astype("float64")
        enc = all_lifted[:, 0] * np.uint64(n_nodes) + all_lifted[:, 1]
        uniq, inv = np.unique(enc, return_inverse=True)
        sums = np.bincount(inv, weights=all_lifted_w)
        cnts = np.bincount(inv)
        mean_lifted = (sums / cnts).astype("float32")
        lifted_uvs = np.column_stack([uniq // np.uint64(n_nodes), uniq % np.uint64(n_nodes)]).astype("uint64")
    else:
        lifted_uvs = np.zeros((0, 2), dtype="uint64")
        mean_lifted = np.zeros((0,), dtype="float32")

    edges = np.concatenate([local_uvs, lifted_uvs], axis=0)
    weights = np.concatenate([mean_local, mean_lifted])
    return edges, weights


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
    image_shape = tuple(image_shape) if not isinstance(image_shape, tuple) else image_shape
    offsets = check_offsets(offsets)

    if foreground_mask is not None:
        # Mask edges connected to background:
        mask_used_edges = from_foreground_mask_to_edge_mask(foreground_mask, offsets, mask_used_edges=mask_used_edges)

    # Create temporary grid graph:
    grid_graph = bic.graph.grid_graph(image_shape)

    # Compute edge mask from offset probs:
    if offsets_probabilities is not None:
        if mask_used_edges is not None:
            warnings.warn("!!! Warning: both edge mask and offsets probabilities were used!!!")
        mask_used_edges = edge_mask_from_offsets_prob(image_shape, offsets_probabilities, mask_used_edges)

    if mask_used_edges is not None:
        uv_ids, edge_weights = _masked_grid_affinity_features(
            grid_graph, affinities, offsets, np.asarray(mask_used_edges),
        )
    else:
        uv_ids, edge_weights = compute_grid_graph_affinity_features(
            grid_graph, affinities, offsets=list(map(list, offsets)),
        )

    nb_nodes = int(grid_graph.number_of_nodes)
    # Grid graphs in bioimage-cpp use NumPy C-order node ids.
    projected_node_ids_to_pixels = np.arange(int(np.prod(image_shape)), dtype="uint64").reshape(image_shape)

    if foreground_mask is not None:
        # Mask background nodes and relabel node ids continuous before to create final graph:
        projected_node_ids_to_pixels = projected_node_ids_to_pixels + 1
        projected_node_ids_to_pixels[np.invert(foreground_mask)] = 0
        projected_node_ids_to_pixels, forward_map, _ = bic.segmentation.relabel_sequential(
            projected_node_ids_to_pixels,
        )
        new_max_label = int(projected_node_ids_to_pixels.max())
        # Shift uv ids by 1 to match the +1 shift applied to node ids, then remap.
        uv_ids = (np.asarray(uv_ids, dtype=forward_map.dtype) + 1)
        uv_ids = forward_map[uv_ids]
        nb_nodes = new_max_label + 1

    # Create new undirected graph with all edges (including long-range ones):
    graph = bic.graph.UndirectedGraph.from_edges(nb_nodes, np.asarray(uv_ids, dtype="uint64"))

    # By default every edge is local / mergeable:
    is_local_edge = np.ones(int(graph.number_of_edges), dtype='bool')
    if set_only_direct_neigh_as_mergeable:
        # Get edge ids of local edges via the grid graph projection.
        is_dir_neighbor, _ = find_indices_direct_neighbors_in_offsets(offsets)
        projection, _ = grid_graph.project_edge_ids_to_pixels_with_offsets(np.asarray(offsets))
        projected_local_edge_ids = projection[is_dir_neighbor]
        valid_ids = projected_local_edge_ids[projected_local_edge_ids != -1].ravel()
        is_local_edge = np.isin(np.arange(edge_weights.shape[0]),
                                valid_ids, assume_unique=True)

    edge_sizes = np.ones(int(graph.number_of_edges), dtype='float32')
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
