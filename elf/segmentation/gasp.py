import numpy as np
import time
from typing import Dict, List, Optional, Tuple

from affogato import segmentation as aff_segm

import nifty.graph.agglo as nifty_agglo
from nifty.graph import components, UndirectedGraph

import warnings

from nifty import tools as ntools
from affogato.affinities import compute_affinities
from affogato.segmentation import compute_mws_segmentation_from_affinities

from . import gasp_utils
from .multicut import compute_edge_costs


def run_GASP(
    graph: UndirectedGraph,
    signed_edge_weights: np.array,
    linkage_criteria: str = "mean",
    add_cannot_link_constraints: bool = False,
    edge_sizes: Optional[np.ndarray] = None,
    is_mergeable_edge: np.ndarray = None,
    use_efficient_implementations: bool = True,
    verbose: bool = False,
    linkage_criteria_kwargs: Optional[Dict] = None,
    merge_constrained_edges_at_the_end: bool = False,
    export_agglomeration_data: bool = False,
    print_every: int = 100000,
) -> Tuple[np.ndarray, float]:
    """Run the Generalized Algorithm for Agglomerative Clustering on Signed Graphs (GASP).

    Args:
        graph: Instance of a graph, e.g. nifty.graph.UndirectedGraph, nifty.graph.undirectedLongRangeGridGraph
            or nifty.graph.rag.gridRag.
        signed_edge_weights: Signed edge weights for clustering.
            Attractive weights are positive; repulsive weights are negative.
        linkage_criteria: Specifies the linkage criteria / update rule used during agglomeration.
            List of available criteria:
                - 'mean', 'average', 'avg'
                - 'max', 'single_linkage'
                - 'min', 'complete_linkage'
                - 'mutex_watershed', 'abs_max'
                - 'sum'
                - 'quantile', 'rank' keeps statistics in a histogram, with parameters:
                        * q : float (default 0.5 equivalent to the median)
                        * numberOfBins: int (default: 40)
                - 'generalized_mean', 'gmean' with parameters:
                        * p : float (default: 1.0)
                        * https://en.wikipedia.org/wiki/Generalized_mean
                - 'smooth_max', 'smax' with parameters:
                        * p : float (default: 0.0)
                        * https://en.wikipedia.org/wiki/Smooth_maximum
        add_cannot_link_constraints: Whether to add cannot link constraints for negative edges.
        edge_sizes: Size of the graph edges.
            Depending on the linkage criteria, they can be used during the agglomeration to weight differently
            the edges  (e.g. with sum or avg linkage criteria). Commonly used with regionAdjGraphs when edges
            represent boundaries of different length between segments / super-pixels. By default, all edges have
            the same weighting.
        is_mergeable_edge: Specifies if an edge can be merged or not.
            Sometimes some edges represent direct-neighbor relations
            and others describe long-range connections. If a long-range connection /edge is assigned to
            `is_mergeable_edge = False`, then the two associated nodes are not merged until they become
            direct neighbors and they get connected in the image-plane.
            By default all edges are mergeable.
        use_efficient_implementations: In the following special cases, efficient implementations are used:
            - 'abs_max' criteria: Mutex Watershed (https://github.com/hci-unihd/mutex-watershed.git)
            - 'max' criteria without cannot-link constraints: maximum spanning tree
        verbose: Whether to be verbose.
        linkage_criteria_kwargs: Additional optional parameters passed to the chosen linkage criteria.
        print_every: After how many agglomeration iteration to print in verbose mode.

    Returns:
        The node labels representing the final clustering.
        The runtime.
    """

    if use_efficient_implementations and (linkage_criteria in ["mutex_watershed", "abs_max"] or
                                          (linkage_criteria == "max" and not add_cannot_link_constraints)):
        assert not export_agglomeration_data, "Exporting extra agglomeration data is not possible when using " \
                                            "the efficient implementation."
        if is_mergeable_edge is not None:
            if not is_mergeable_edge.all():
                print("WARNING: Efficient implementations only works when all edges are mergeable. "
                      "In this mode, lifted and local edges will be treated equally, so there could be final clusters "
                      "consisting of multiple components 'disconnennted' in the image plane.")
        # assert is_mergeable_edge is None, "Efficient implementations only works when all edges are mergeable"
        nb_nodes = graph.numberOfNodes
        uv_ids = graph.uvIds()
        mutex_edges = signed_edge_weights < 0.

        tick = time.time()
        # These implementations use the convention where all edge weights are positive
        assert aff_segm is not None, "For the efficient implementation of GASP, affogato module is needed"
        if linkage_criteria in ["mutex_watershed", "abs_max"]:
            node_labels = aff_segm.compute_mws_clustering(nb_nodes,
                                                          uv_ids[np.logical_not(mutex_edges)],
                                                          uv_ids[mutex_edges],
                                                          signed_edge_weights[np.logical_not(mutex_edges)],
                                                          -signed_edge_weights[mutex_edges])
        else:
            graph_components = components(graph)
            graph_components.buildFromEdgeLabels(mutex_edges)
            node_labels = graph_components.componentLabels()
        runtime = time.time() - tick
    else:
        cluster_policy = nifty_agglo.get_GASP_policy(
            graph, signed_edge_weights,
            edge_sizes=edge_sizes,
            linkage_criteria=linkage_criteria,
            linkage_criteria_kwargs=linkage_criteria_kwargs,
            add_cannot_link_constraints=add_cannot_link_constraints,
            is_mergeable_edge=is_mergeable_edge,
            merge_constrained_edges_at_the_end=merge_constrained_edges_at_the_end,
            collect_stats_for_exported_data=export_agglomeration_data
        )
        agglomerativeClustering = nifty_agglo.agglomerativeClustering(cluster_policy)

        # Run clustering:
        tick = time.time()
        agglomerativeClustering.run(verbose=verbose, printNth=print_every)
        runtime = time.time() - tick

        # Collect results:
        node_labels = agglomerativeClustering.result()

        if export_agglomeration_data:
            exported_data = cluster_policy.exportAgglomerationData()

    if export_agglomeration_data:
        out_dict = {"agglomeration_data": exported_data}
        return node_labels, runtime, out_dict
    else:
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
        assert affinities_weights is None, "Not yet implemented from pixels"
        assert affinities.shape[0] == len(self.offsets)
        offsets = self.offsets
        if self.used_offsets is not None:
            affinities = affinities[self.used_offsets]
            offsets = offsets[self.used_offsets]
            if mask_used_edges is not None:
                mask_used_edges = mask_used_edges[self.used_offsets]

        image_shape = affinities.shape[1:]

        # affinities = affinities - np.mean(affinities, axis=(1,2,3))[:,np.newaxis,np.newaxis, np.newaxis]
        # Check if I should use efficient implementation of the MWS:
        run_kwargs = self.run_GASP_kwargs
        export_agglomeration_data = run_kwargs.get("export_agglomeration_data", False)
        # TODO: add implementation of single-linkage from pixels using affogato.segmentation.connected_components
        if run_kwargs.get("use_efficient_implementations", True) and\
           run_kwargs.get("linkage_criteria") in ["mutex_watershed", "abs_max"]:
            assert compute_mws_segmentation_from_affinities is not None, \
                "Efficient MWS implementation not available. Update the affogato repository "
            assert not export_agglomeration_data, "Exporting extra agglomeration data is not possible when using " \
                                                  "the efficient implementation of MWS."
            if self.set_only_direct_neigh_as_mergeable:
                warnings.warn("With efficient implementation of MWS, it is not possible to set only direct neighbors"
                              "as mergeable.")
            tick = time.time()
            segmentation, valid_edge_mask = compute_mws_segmentation_from_affinities(
                affinities, offsets, beta_parameter=self.beta_bias,
                foreground_mask=foreground_mask, edge_mask=mask_used_edges,
                return_valid_edge_mask=True)
            runtime = time.time() - tick
            if self.return_extra_outputs:
                MC_energy = self.get_multicut_energy_segmentation(segmentation, affinities,
                                                                  offsets, valid_edge_mask)
                out_dict = {"runtime": runtime, "multicut_energy": MC_energy}
                return segmentation, out_dict
            else:
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
            # signed_weights = edge_weights + 0.3
            signed_weights = edge_weights - self.beta_bias

        # Run GASP:
        if self.verbose:
            print("Start agglo...")

        outputs = run_GASP(graph,
                           signed_weights,
                           edge_sizes=edge_sizes,
                           is_mergeable_edge=is_local_edge,
                           verbose=self.verbose,
                           **self.run_GASP_kwargs)

        if export_agglomeration_data:
            nodeSeg, runtime, exported_data = outputs
        else:
            exported_data = {}
            nodeSeg, runtime = outputs

        segmentation = ntools.mapFeaturesToLabelArray(
            projected_node_ids_to_pixels,
            np.expand_dims(nodeSeg, axis=-1),
            nb_threads=self.n_threads,
            fill_value=-1.,
            ignore_label=-1,
        )[..., 0].astype(np.int64)

        if self.return_extra_outputs:
            frustration = self.get_frustration(graph, nodeSeg, signed_weights)
            MC_energy = self.get_multicut_energy(graph, nodeSeg, signed_weights, edge_sizes)
            out_dict = {"multicut_energy": MC_energy,
                        "runtime": runtime,
                        "graph": graph,
                        "is_local_edge": is_local_edge,
                        "edge_sizes": edge_sizes,
                        "edge_weights": signed_weights,
                        "frustration": frustration}
            if export_agglomeration_data:
                out_dict.update(exported_data)
            return segmentation, out_dict
        else:
            if export_agglomeration_data:
                warnings.warn("In order to export agglomeration data, also set the `return_extra_outputs` to True")
            return segmentation, runtime

    def run_GASP_from_superpixels(self, affinities, superpixel_segmentation, foreground_mask=None,
                                  mask_used_edges=None, affinities_weights=None):
        """@private
        """
        # TODO: compute affiniteis_weights automatically from segmentation if needed
        # When I will implement the mask_edge, remeber to crop it depending on the used offsets
        assert mask_used_edges is None, "Edge mask cannot be used when starting from a segmentation."
        assert self.set_only_direct_neigh_as_mergeable, "Not implemented atm from superpixels"
        featurer = gasp_utils.AccumulatorLongRangeAffs(self.offsets,
                                                       offsets_weights=self.offsets_weights,
                                                       used_offsets=self.used_offsets,
                                                       verbose=self.verbose,
                                                       n_threads=self.n_threads,
                                                       invert_affinities=False,
                                                       statistic="mean",
                                                       offset_probabilities=self.offsets_probabilities,
                                                       return_dict=True)

        # Compute graph and edge weights by accumulating over the affinities:
        featurer_outputs = featurer(affinities, superpixel_segmentation,
                                    affinities_weights=affinities_weights)
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

        # Run GASP:
        export_agglomeration_data = self.run_GASP_kwargs.get("export_agglomeration_data", False)
        outputs = \
            run_GASP(graph, signed_weights,
                     edge_sizes=edge_sizes,
                     is_mergeable_edge=is_local_edge,
                     verbose=self.verbose,
                     **self.run_GASP_kwargs)

        if export_agglomeration_data:
            node_labels, runtime, exported_data = outputs
        else:
            exported_data = {}
            node_labels, runtime = outputs

        # Map node labels back to the original superpixel segmentation:
        final_segm = ntools.mapFeaturesToLabelArray(
            superpixel_segmentation,
            np.expand_dims(node_labels, axis=-1),
            nb_threads=self.n_threads,
            fill_value=-1.,
            ignore_label=-1,
        )[..., 0].astype(np.int64)

        # If there was a background label, reset it to zero:
        min_label = final_segm.min()
        if min_label < 0:
            assert min_label == -1

            # Move bacground label to 0, and map 0 segment (if any) to MAX_LABEL+1.
            # In this way, most of the final labels will stay consistent with the graph and given over-segmentation
            background_mask = final_segm == min_label
            zero_mask = final_segm == 0
            if np.any(zero_mask):
                max_label = final_segm.max()
                warnings.warn("Zero segment remapped to {} in final segmentation".format(max_label+1))
                final_segm[zero_mask] = max_label + 1
            final_segm[background_mask] = 0

        if self.return_extra_outputs:
            MC_energy = self.get_multicut_energy(graph, node_labels, signed_weights, edge_sizes)
            out_dict = {"multicut_energy": MC_energy,
                        "runtime": runtime,
                        "graph": graph,
                        "is_local_edge": is_local_edge,
                        "edge_sizes": edge_sizes}
            if export_agglomeration_data:
                out_dict.update(exported_data)
            return final_segm, out_dict
        else:
            if export_agglomeration_data:
                warnings.warn("In order to export agglomeration data, also set the `return_extra_outputs` to True")
            return final_segm, runtime

    def get_multicut_energy(self, graph, node_segm, edge_weights, edge_sizes=None):
        """@private
        """
        if edge_sizes is None:
            edge_sizes = np.ones_like(edge_weights)
        edge_labels = graph.nodeLabelsToEdgeLabels(node_segm)
        return (edge_weights * edge_labels * edge_sizes).sum()

    def get_multicut_energy_segmentation(self, pixel_segm, affinities, offsets, edge_mask=None):
        """@private
        """
        if edge_mask is None:
            edge_mask = np.ones_like(affinities, dtype="bool")

        log_affinities = compute_edge_costs(1 - affinities, beta=self.beta_bias)

        # Find affinities "on cut":
        affs_not_on_cut, _ = compute_affinities(pixel_segm.astype("uint64"), offsets.tolist(), False, 0)
        return log_affinities[np.logical_and(affs_not_on_cut == 0, edge_mask)].sum()

    def get_frustration(self, graph, node_segm, edge_weights):
        """@private
        """
        edge_labels = graph.nodeLabelsToEdgeLabels(node_segm)
        pos_frus = ((edge_weights > 0) * edge_labels).sum()
        neg_frus = ((edge_weights < 0) * (1-edge_labels)).sum()
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
