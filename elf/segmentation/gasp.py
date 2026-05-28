import time
import warnings
from typing import Dict, List, Optional, Tuple

import bioimage_cpp as bic
import numpy as np

from . import gasp_utils
from .multicut import compute_edge_costs


_LINKAGE_ALIASES = {
    "mean": "mean", "average": "mean", "avg": "mean",
    "max": "max", "single_linkage": "max",
    "min": "min", "complete_linkage": "min",
    "mutex_watershed": "mutex_watershed", "abs_max": "abs_max",
    "sum": "sum",
}
_UNSUPPORTED_LINKAGES = {
    "quantile", "rank", "generalized_mean", "gmean", "smooth_max", "smax",
}


def _normalize_linkage(criterion: str) -> str:
    """@private"""
    if criterion in _UNSUPPORTED_LINKAGES:
        raise NotImplementedError(
            f"Linkage criterion '{criterion}' is not supported on the bioimage-cpp backend."
        )
    try:
        return _LINKAGE_ALIASES[criterion]
    except KeyError as err:
        raise ValueError(f"Unknown linkage criterion: {criterion}") from err


def _node_labels_to_edge_labels(graph, node_labels: np.ndarray) -> np.ndarray:
    """@private"""
    uv = np.asarray(graph.uv_ids())
    return (node_labels[uv[:, 0]] != node_labels[uv[:, 1]]).astype("uint8")


def _map_node_labels_to_pixels(projection: np.ndarray, node_labels: np.ndarray) -> np.ndarray:
    """@private

    Project per-node labels back to a pixel grid using ``projection``, an array of node ids
    with ``-1`` marking masked-out pixels. Replicates ``nifty.tools.mapFeaturesToLabelArray``
    with ``fill_value=-1, ignore_label=-1`` for a 1-channel feature vector.
    """
    proj = np.asarray(projection)
    out = np.full(proj.shape, -1, dtype=np.int64)
    valid = proj != -1
    out[valid] = node_labels[proj[valid]]
    return out


def run_GASP(
    graph,
    signed_edge_weights: np.ndarray,
    linkage_criteria: str = "mean",
    add_cannot_link_constraints: bool = False,
    edge_sizes: Optional[np.ndarray] = None,
    is_mergeable_edge: Optional[np.ndarray] = None,
    use_efficient_implementations: bool = True,
    verbose: bool = False,
    linkage_criteria_kwargs: Optional[Dict] = None,
    merge_constrained_edges_at_the_end: bool = False,
    export_agglomeration_data: bool = False,
    print_every: int = 100000,
) -> Tuple[np.ndarray, float]:
    """Run the Generalized Algorithm for Agglomerative Clustering on Signed Graphs (GASP).

    Args:
        graph: An undirected graph (``bioimage_cpp.graph.UndirectedGraph`` or
            ``RegionAdjacencyGraph``).
        signed_edge_weights: Signed edge weights for clustering.
            Attractive weights are positive; repulsive weights are negative.
        linkage_criteria: Linkage criterion / update rule used during agglomeration.
            Available criteria: 'mean'/'average'/'avg', 'max'/'single_linkage',
            'min'/'complete_linkage', 'mutex_watershed'/'abs_max', 'sum'.
        add_cannot_link_constraints: Not supported on the bioimage-cpp backend; must be False
            (cannot-link constraints are implicit for the 'mutex_watershed' linkage).
        edge_sizes: Size of the graph edges.
        is_mergeable_edge: Boolean mask marking edges that may trigger a merge. Non-mergeable
            edges are processed only to install cluster-level cannot-link constraints.
        use_efficient_implementations: In the following special cases, efficient
            implementations are used:
            - 'abs_max' criterion: graph-based mutex watershed.
            - 'max' criterion: connected components on positive-weight edges.
        verbose: Not supported on the bioimage-cpp backend; must be False.
        linkage_criteria_kwargs: Not supported on the bioimage-cpp backend; must be None.
        merge_constrained_edges_at_the_end: Not supported on the bioimage-cpp backend; must be False.
        export_agglomeration_data: Not supported on the bioimage-cpp backend; must be False.
        print_every: Not supported on the bioimage-cpp backend.

    Returns:
        The node labels representing the final clustering.
        The runtime.
    """
    if add_cannot_link_constraints:
        raise NotImplementedError(
            "add_cannot_link_constraints=True is not supported on the bioimage-cpp backend; "
            "use linkage_criteria='mutex_watershed' for hard cannot-link constraints."
        )
    if verbose:
        raise NotImplementedError("verbose=True is not supported on the bioimage-cpp backend.")
    if linkage_criteria_kwargs is not None:
        raise NotImplementedError(
            "linkage_criteria_kwargs is not supported on the bioimage-cpp backend."
        )
    if merge_constrained_edges_at_the_end:
        raise NotImplementedError(
            "merge_constrained_edges_at_the_end=True is not supported on the bioimage-cpp backend."
        )
    if export_agglomeration_data:
        raise NotImplementedError(
            "export_agglomeration_data=True is not supported on the bioimage-cpp backend."
        )
    del print_every  # accepted for backwards-compatibility; bic policies have no verbose hook

    criterion = _normalize_linkage(linkage_criteria)
    signed_edge_weights = np.asarray(signed_edge_weights)

    if use_efficient_implementations and criterion in ("mutex_watershed", "abs_max", "max"):
        if is_mergeable_edge is not None and not np.asarray(is_mergeable_edge).all():
            print("WARNING: Efficient implementations only work when all edges are mergeable. "
                  "In this mode, lifted and local edges will be treated equally, so there could be final clusters "
                  "consisting of multiple components 'disconnennted' in the image plane.")

        n_nodes = int(graph.number_of_nodes)
        uv_ids = np.asarray(graph.uv_ids(), dtype="uint64")
        mutex_edges = signed_edge_weights < 0.0

        tick = time.time()
        if criterion in ("mutex_watershed", "abs_max"):
            attractive_uvs = uv_ids[~mutex_edges]
            attractive_graph = bic.graph.UndirectedGraph.from_edges(n_nodes, attractive_uvs)
            node_labels = bic.graph.mutex_watershed.mutex_watershed_clustering(
                attractive_graph,
                signed_edge_weights[~mutex_edges].astype("float32"),
                uv_ids[mutex_edges],
                (-signed_edge_weights[mutex_edges]).astype("float32"),
            )
        else:  # criterion == "max"
            node_labels = bic.graph.connected_components(graph, edge_mask=~mutex_edges)
        runtime = time.time() - tick
        return node_labels, runtime

    policy = bic.graph.agglomeration.GaspClusterPolicy(num_clusters_stop=1, linkage=criterion)
    tick = time.time()
    node_labels = policy.optimize(
        graph,
        signed_edge_weights.astype("float64"),
        edge_sizes=None if edge_sizes is None else np.asarray(edge_sizes, dtype="float64"),
        is_mergeable=None if is_mergeable_edge is None else np.asarray(is_mergeable_edge, dtype=bool),
    )
    runtime = time.time() - tick
    return node_labels, runtime


class GaspFromAffinities:
    """Run the Generalized Algorithm for Signed Graph Agglomerative Partitioning from affinities.

    Affinities are usually computed from an image.
    The clustering can be both initialized from pixels and superpixels.

    Args:
        offsets: Offsets indiicating the pixel directions of the respective affinity channels.
            Example with three direct neighbors in 3D:
            [ [-1, 0, 0],
              [0, -1, 0],
              [0, 0, -1]  ]
        beta_bias: Boundary bias term. Add bias to the edge weights.
        superpixel_generator: Callable with inputs (affinities, *args_superpixel_gen).
            If None, run_GASP() is initialized from pixels.
        run_GASP_kwargs: Additional arguments to be passed to run_GASP().
        n_threads: Number of threads for parallelization.
        verbose: Whether to be verbose.
        invert_affinities: Whether to invert the affinities.
        offsets_probabilities: List specifying the probabilities with which each type of edge-connection
            should be added to the graph. By default all connections are added.
        used_offsets: List specifying which offsets (i.e. which channels in the affinities array)
            should be considered to accumulate the average over the initial superpixel-boundaries.
            By default all offsets are used.
        offsets_weights: List specifying how each offset (i.e. a type of edge-connection in
            the graph) should be weighted in the average-accumulation during the accumulation,
            related to the input `edge_sizes` of run_GASP(). By default all edges are weighted equally.
    """
    def __init__(
        self,
        offsets: List[List[int]],
        beta_bias: float = 0.5,
        superpixel_generator: Optional[callable] = None,
        run_GASP_kwargs: Optional[Dict] = None,
        n_threads: Optional[int] = 1,
        verbose: bool = False,
        invert_affinities: bool = False,
        offsets_probabilities: Optional[List[float]] = None,
        use_logarithmic_weights: bool = False,
        used_offsets: Optional[List[int]] = None,
        offsets_weights: Optional[List[float]] = None,
        return_extra_outputs: bool = False,
        set_only_direct_neigh_as_mergeable: bool = True,
        ignore_edge_sizes: bool = False,
    ):
        offsets = gasp_utils.check_offsets(offsets)
        self.offsets = offsets

        # Parse inputs:
        if offsets_probabilities is not None:
            if isinstance(offsets_probabilities, (float, int)):
                is_offset_direct_neigh, _ = gasp_utils.find_indices_direct_neighbors_in_offsets(offsets)
                offsets_probabilities = np.ones((offsets.shape[0],), dtype="float32") * offsets_probabilities
                # Direct neighbors should be always added:
                offsets_probabilities[is_offset_direct_neigh] = 1.
            else:
                offsets_probabilities = np.require(offsets_probabilities, dtype="float32")
                assert len(offsets_probabilities) == len(offsets)

        if offsets_weights is not None:
            offsets_weights = np.require(offsets_weights, dtype="float32")
            assert len(offsets_weights) == len(offsets)

        if used_offsets is not None:
            assert len(used_offsets) < offsets.shape[0]
            if offsets_probabilities is not None:
                offsets_probabilities = offsets_probabilities[used_offsets]
            if offsets_weights is not None:
                offsets_weights = offsets_weights[used_offsets]

        self.offsets_probabilities = offsets_probabilities
        self.used_offsets = used_offsets
        self.offsets_weights = offsets_weights

        assert isinstance(n_threads, int)
        self.n_threads = n_threads

        assert isinstance(invert_affinities, bool)
        self.invert_affinities = invert_affinities

        assert isinstance(verbose, bool)
        self.verbose = verbose

        run_GASP_kwargs = run_GASP_kwargs if isinstance(run_GASP_kwargs, dict) else {}
        self.run_GASP_kwargs = run_GASP_kwargs

        assert (beta_bias <= 1.0) and (
                beta_bias >= 0.), "The beta bias parameter is expected to be in the interval (0,1)"
        self.beta_bias = beta_bias

        assert isinstance(use_logarithmic_weights, bool)
        self.use_logarithmic_weights = use_logarithmic_weights

        self.superpixel_generator = superpixel_generator
        self.return_extra_outputs = return_extra_outputs
        self.set_only_direct_neigh_as_mergeable = set_only_direct_neigh_as_mergeable

        assert not ignore_edge_sizes, "This option is deprecated"

    def __call__(
        self,
        affinities: np.ndarray,
        *args_superpixel_gen,
        mask_used_edges=None,
        affinities_weights=None,
        foreground_mask=None
    ) -> Tuple[np.ndarray, float]:
        """Run GASP segmentation.

        Args:
            affinities: Affinity map, array with shape (nb_offsets, ) + shape_image,
                where the shape of the image can be 2D or 3D.
                Passed values should be in interval [0, 1], where 1-values should represent intra-cluster connections
                (high affinity, merge) and 0-values inter-cluster connections (low affinity, boundary evidence, split).
            args_superpixel_gen: Additional arguments passed to the superpixel generator.

        Returns:
            The segmentation.
            The runtime.
        """
        assert isinstance(affinities, np.ndarray)
        assert affinities.ndim == 4, "Need affinities with 4 channels, got %i" % affinities.ndim
        if self.invert_affinities:
            affinities_ = 1. - affinities
        else:
            affinities_ = affinities

        if self.superpixel_generator is not None:
            superpixel_segmentation = self.superpixel_generator(
                affinities_, *args_superpixel_gen, foreground_mask=foreground_mask
            )
            return self.run_GASP_from_superpixels(affinities_, superpixel_segmentation,
                                                  mask_used_edges=mask_used_edges,
                                                  affinities_weights=affinities_weights,
                                                  foreground_mask=foreground_mask)
        else:
            return self.run_GASP_from_pixels(affinities_, mask_used_edges=mask_used_edges,
                                             affinities_weights=affinities_weights, foreground_mask=foreground_mask)

    def run_GASP_from_pixels(self, affinities, mask_used_edges=None, foreground_mask=None, affinities_weights=None):
        """@private
        """
        if affinities_weights is not None:
            raise NotImplementedError("affinities_weights are not supported on the bioimage-cpp backend.")
        assert affinities.shape[0] == len(self.offsets)
        offsets = self.offsets
        if self.used_offsets is not None:
            affinities = affinities[self.used_offsets]
            offsets = offsets[self.used_offsets]
            if mask_used_edges is not None:
                mask_used_edges = mask_used_edges[self.used_offsets]

        image_shape = affinities.shape[1:]

        # Check if I should use efficient implementation of the MWS:
        run_kwargs = self.run_GASP_kwargs
        export_agglomeration_data = run_kwargs.get("export_agglomeration_data", False)
        if run_kwargs.get("use_efficient_implementations", True) and \
           run_kwargs.get("linkage_criteria") in ["mutex_watershed", "abs_max"]:
            if mask_used_edges is not None:
                raise NotImplementedError(
                    "Edge masks are not supported by the efficient pixel-level mutex watershed."
                )
            assert not export_agglomeration_data, "Exporting extra agglomeration data is not possible when using " \
                                                  "the efficient implementation of MWS."
            if self.set_only_direct_neigh_as_mergeable:
                warnings.warn("With efficient implementation of MWS, it is not possible to set only direct neighbors"
                              "as mergeable.")
            # Reorder so that direct-neighbor (attractive) channels come first.
            is_dir_neighbor, _ = gasp_utils.find_indices_direct_neighbors_in_offsets(offsets)
            attractive_idx = np.where(is_dir_neighbor)[0]
            mutex_idx = np.where(~is_dir_neighbor)[0]
            ordered_idx = np.concatenate([attractive_idx, mutex_idx])
            ordered_offsets = offsets[ordered_idx]
            ordered_affs = affinities[ordered_idx]

            # Apply beta_bias: attractive weight = affinity - beta, mutex weight = beta - affinity.
            weights = np.empty_like(ordered_affs, dtype="float32")
            n_attractive = int(attractive_idx.size)
            weights[:n_attractive] = ordered_affs[:n_attractive] - self.beta_bias
            weights[n_attractive:] = self.beta_bias - ordered_affs[n_attractive:]

            tick = time.time()
            segmentation = bic.segmentation.mutex_watershed(
                weights, list(map(list, ordered_offsets)),
                number_of_attractive_channels=n_attractive,
                mask=foreground_mask,
            )
            runtime = time.time() - tick
            segmentation = segmentation.astype(np.int64)

            if self.return_extra_outputs:
                MC_energy = self.get_multicut_energy_segmentation(segmentation, affinities, offsets)
                out_dict = {"runtime": runtime, "multicut_energy": MC_energy}
                return segmentation, out_dict
            return segmentation, runtime

        # Build graph:
        if self.verbose:
            print("Building graph...")
        graph, projected_node_ids_to_pixels, edge_weights, is_local_edge, edge_sizes = \
            gasp_utils.build_pixel_long_range_grid_graph_from_offsets(
                image_shape,
                offsets,
                affinities,
                offsets_probabilities=self.offsets_probabilities,
                mask_used_edges=mask_used_edges,
                offset_weights=self.offsets_weights,
                foreground_mask=foreground_mask,
                set_only_direct_neigh_as_mergeable=self.set_only_direct_neigh_as_mergeable
            )

        # Compute log costs:
        log_costs = compute_edge_costs(1 - edge_weights, beta=self.beta_bias)
        if self.use_logarithmic_weights:
            signed_weights = log_costs
        else:
            signed_weights = edge_weights - self.beta_bias

        # Run GASP:
        if self.verbose:
            print("Start agglo...")

        node_seg, runtime = run_GASP(graph,
                                     signed_weights,
                                     edge_sizes=edge_sizes,
                                     is_mergeable_edge=is_local_edge,
                                     **self.run_GASP_kwargs)

        segmentation = _map_node_labels_to_pixels(projected_node_ids_to_pixels, node_seg)

        if self.return_extra_outputs:
            frustration = self.get_frustration(graph, node_seg, signed_weights)
            MC_energy = self.get_multicut_energy(graph, node_seg, signed_weights, edge_sizes)
            out_dict = {"multicut_energy": MC_energy,
                        "runtime": runtime,
                        "graph": graph,
                        "is_local_edge": is_local_edge,
                        "edge_sizes": edge_sizes,
                        "edge_weights": signed_weights,
                        "frustration": frustration}
            return segmentation, out_dict
        return segmentation, runtime

    def run_GASP_from_superpixels(self, affinities, superpixel_segmentation, foreground_mask=None,
                                  mask_used_edges=None, affinities_weights=None):
        """@private
        """
        if affinities_weights is not None:
            raise NotImplementedError(
                "affinities_weights are not supported on the bioimage-cpp backend."
            )
        assert mask_used_edges is None, "Edge mask cannot be used when starting from a segmentation."
        assert self.set_only_direct_neigh_as_mergeable, "Not implemented atm from superpixels"
        featurer = gasp_utils.AccumulatorLongRangeAffs(
            self.offsets,
            offsets_weights=self.offsets_weights,
            used_offsets=self.used_offsets,
            verbose=self.verbose,
            n_threads=self.n_threads,
            invert_affinities=False,
            statistic="mean",
            offset_probabilities=self.offsets_probabilities,
            return_dict=True,
        )

        # Compute graph and edge weights by accumulating over the affinities:
        featurer_outputs = featurer(affinities, superpixel_segmentation)
        graph = featurer_outputs["graph"]
        edge_indicators = featurer_outputs["edge_indicators"]
        edge_sizes = featurer_outputs["edge_sizes"]
        is_local_edge = featurer_outputs["is_local_edge"]

        # Optionally, use logarithmic weights and apply bias parameter
        log_costs = compute_edge_costs(1 - edge_indicators, beta=self.beta_bias)
        if self.use_logarithmic_weights:
            signed_weights = log_costs
        else:
            signed_weights = edge_indicators - self.beta_bias

        node_labels, runtime = run_GASP(graph, signed_weights,
                                        edge_sizes=edge_sizes,
                                        is_mergeable_edge=is_local_edge,
                                        **self.run_GASP_kwargs)

        # Map node labels back to the original superpixel segmentation:
        final_segm = _map_node_labels_to_pixels(superpixel_segmentation, node_labels)

        # If there was a background label, reset it to zero:
        min_label = final_segm.min()
        if min_label < 0:
            assert min_label == -1
            background_mask = final_segm == min_label
            zero_mask = final_segm == 0
            if np.any(zero_mask):
                max_label = final_segm.max()
                warnings.warn("Zero segment remapped to {} in final segmentation".format(max_label + 1))
                final_segm[zero_mask] = max_label + 1
            final_segm[background_mask] = 0

        if self.return_extra_outputs:
            MC_energy = self.get_multicut_energy(graph, node_labels, signed_weights, edge_sizes)
            out_dict = {"multicut_energy": MC_energy,
                        "runtime": runtime,
                        "graph": graph,
                        "is_local_edge": is_local_edge,
                        "edge_sizes": edge_sizes}
            return final_segm, out_dict
        return final_segm, runtime

    def get_multicut_energy(self, graph, node_segm, edge_weights, edge_sizes=None):
        """@private
        """
        if edge_sizes is None:
            edge_sizes = np.ones_like(edge_weights)
        edge_labels = _node_labels_to_edge_labels(graph, node_segm)
        return (edge_weights * edge_labels * edge_sizes).sum()

    def get_multicut_energy_segmentation(self, pixel_segm, affinities, offsets, edge_mask=None):
        """@private
        """
        if edge_mask is None:
            edge_mask = np.ones_like(affinities, dtype="bool")

        log_affinities = compute_edge_costs(1 - affinities, beta=self.beta_bias)

        # Find affinities "on cut":
        affs_not_on_cut, _ = bic.affinities.compute_affinities(
            pixel_segm.astype("uint64"), list(map(list, offsets)), ignore_label=0,
        )
        return log_affinities[np.logical_and(affs_not_on_cut == 0, edge_mask)].sum()

    def get_frustration(self, graph, node_segm, edge_weights):
        """@private
        """
        edge_labels = _node_labels_to_edge_labels(graph, node_segm)
        pos_frus = ((edge_weights > 0) * edge_labels).sum()
        neg_frus = ((edge_weights < 0) * (1 - edge_labels)).sum()
        return [pos_frus, neg_frus]


class SegmentationFeeder:
    """Simple superpixel generator for GASP segmentation.

    Expects affinities and initial segmentation (with optional foreground mask), can be used as "superpixel_generator"..
    """
    def __call__(self, affinities, segmentation, foreground_mask=None):
        """@private
        """
        if foreground_mask is not None:
            assert foreground_mask.shape == segmentation.shape
            segmentation = segmentation.astype("int64")
            segmentation = np.where(foreground_mask, segmentation, np.ones_like(segmentation) * (-1))
        return segmentation
