import os
import pickle
from copy import deepcopy
from typing import Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np

from .import features as elf_feats
from .import learning as elf_learn
from .import lifted_multicut as elf_lmc
from .import multicut as elf_mc
from .import watershed as elf_ws

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES = {"raw-edge-features", "raw-region-features", "boundary-edge-features"}
"""@private
"""
DEFAULT_WS_KWARGS = {"threshold": 0.25, "sigma_seeds": 2., "sigma_weights": 2.,
                     "min_size": 100, "alpha": 0.9, "pixel_pitch": None,
                     "apply_nonmax_suppression": False}
"""@private
"""
DEFAULT_RF_KWARGS = {"ignore_label": None, "n_estimators": 200, "max_depth": 10}
"""@private
"""


def _compute_watershed(boundaries, use_2dws, mask, ws_kwargs, n_threads):
    if use_2dws:
        bd_ = boundaries if isinstance(boundaries, np.ndarray) else boundaries[0]
        return elf_ws.stacked_watershed(bd_, mask=mask, n_threads=n_threads, **ws_kwargs)[0]
    else:
        return elf_ws.distance_transform_watershed(boundaries, mask=mask, **ws_kwargs)[0]


def _compute_xyz_boundary_features(rag, boundaries, watershed, use_2dws, n_threads):
    bd_xy, bd_z = boundaries
    z_edges = elf_feats.compute_z_edge_mask(rag, watershed)
    feats = elf_feats.compute_boundary_features_with_filters(rag, bd_xy, use_2dws, n_threads=n_threads)
    feats_z = elf_feats.compute_boundary_features_with_filters(rag, bd_z, use_2dws, n_threads=n_threads)
    feats[z_edges] = feats_z[z_edges]
    return feats


def _compute_features(raw, boundaries, watershed, feature_names, use_2dws, n_threads):
    if (len(FEATURE_NAMES - set(feature_names)) > 0) or (len(feature_names) == 0):
        raise ValueError("Invalid feature set")
    if isinstance(boundaries, np.ndarray):
        if raw.shape != boundaries.shape:
            raise ValueError("Shapes %s and %s do not match" % (str(raw.shape), str(boundaries.shape)))
    if raw.shape != watershed.shape:
        raise ValueError("Shapes %s and %s do not match" % (str(raw.shape), str(watershed.shape)))

    rag = elf_feats.compute_rag(watershed, n_threads=n_threads)

    features = []
    if "raw-edge-features" in feature_names:
        feats = elf_feats.compute_boundary_features_with_filters(rag, raw, use_2dws,
                                                                 n_threads=n_threads)
        features.append(feats)
    if "boundary-edge-features" in feature_names:
        if isinstance(boundaries, np.ndarray):
            feats = elf_feats.compute_boundary_features_with_filters(rag, boundaries, use_2dws,
                                                                     n_threads=n_threads)
        else:
            if not use_2dws:
                raise ValueError("Separate boundary maps for xy and z edges are only available if use_2dws is True")
            if len(boundaries) != 2 or\
               (not isinstance(boundaries[0], np.ndarray)) or\
               (not isinstance(boundaries[1], np.ndarray)):
                raise ValueError("Invalid boundary maps")
            feats = _compute_xyz_boundary_features(rag, boundaries, watershed, use_2dws, n_threads)
        features.append(feats)
    if "raw-region-features" in feature_names:
        feats = elf_feats.compute_region_features(rag.uvIds(), raw, watershed.astype("uint32"), n_threads=n_threads)
        features.append(feats)

    # for now, we always append the length as one other feature
    # eventually, it would be nice to add topolgy features, cf.
    # https://github.com/ilastik/nature_methods_multicut_pipeline/blob/master/software/multicut_src/DataSet.py#L954
    # https://github.com/DerThorsten/nifty/blob/master/src/python/lib/graph/rag/accumulate.cxx#L361
    edge_len = elf_feats.compute_boundary_mean_and_length(rag, raw, n_threads=n_threads)[:, 1]
    features.append(edge_len[:, None])

    features = np.concatenate(features, axis=1)
    assert len(features) == rag.numberOfEdges
    return rag, features


def _compute_features_and_labels(
    raw, boundaries, watershed, labels, mask, use_2dws, feature_names, ignore_label, n_threads
):
    rag, features = _compute_features(raw, boundaries, watershed, feature_names, use_2dws, n_threads)
    edge_mask = np.ones(len(features), dtype="bool")

    if ignore_label is None:
        edge_labels = elf_learn.compute_edge_labels(rag, labels, n_threads=n_threads)
    else:
        edge_labels, edge_mask1 = elf_learn.compute_edge_labels(rag, labels, ignore_label, n_threads)
        edge_mask = np.logical_and(edge_mask, edge_mask1)

    if mask is not None:
        edge_mask2 = (rag.uvIds() != 0).any(axis=1)
        edge_mask = np.logical_and(edge_mask, edge_mask2)

    features, edge_labels = features[edge_mask], edge_labels[edge_mask]
    assert len(features) == len(edge_labels)

    if use_2dws:
        z_edges = elf_feats.compute_z_edge_mask(rag, watershed)
        z_edges = z_edges[edge_mask]
        assert len(z_edges) == len(features)
    else:
        z_edges = None

    return features, edge_labels, z_edges


def _get_solver(solver):
    if isinstance(solver, str):
        solver = elf_mc.get_multicut_solver(solver)
    elif not callable(solver):
        raise ValueError("Invalid multicut solver")
    return solver


def _mask_edges(rag, edge_costs):
    uv_ids = rag.uvIds()
    ignore_edges = (uv_ids == 0).any(axis=1)
    edge_costs[ignore_edges] = - np.abs(edge_costs).max()
    return edge_costs


#
# training workflows
#


def edge_training(
    raw: Union[np.ndarray, List[np.ndarray]],
    boundaries: Union[np.ndarray, List[np.ndarray]],
    labels: Union[np.ndarray, List[np.ndarray]],
    use_2dws: bool,
    watershed: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    mask: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    feature_names: List[str] = FEATURE_NAMES,
    ws_kwargs: Dict = DEFAULT_WS_KWARGS,
    learning_kwargs: Dict = DEFAULT_RF_KWARGS,
    n_threads: Optional[int] = None,
) -> "RandomForestClassifier":
    """Train random forest classifier for edges.

    Args:
        raw: Raw data or list of raw data arrays.
        boundaries: Boundary maps or list of boundary maps.
        labels: The groundtruth segmentation(s).
        use_2dws: Whether to run watersheds in 2d and stack them or if the watersheds passed are 2d and stacked.
        watershed: Watershed segmentation(s). Will be computed if None is passed.
        mask: Mask(s) to exclude from training data.
        feature_names: The names of features that will be computed.
        ws_kwargs: The keyword arguments for watershed function.
        learning_kwargs: The keyword arguments for random forest.
        n_threads: The number of threads.

    Returns:
        The trained random forest classifier.
    """

    # we store the ignore label(s) in the random forest kwargs,
    # but they need to be removed before this is passed to the rf implementation
    rf_kwargs = deepcopy(learning_kwargs)
    ignore_label = rf_kwargs.pop("ignore_label", None)

    # if the raw data is a numpy array, we assume that we have a single training set
    if isinstance(raw, np.ndarray):
        if (not isinstance(boundaries, np.ndarray)) or (not isinstance(labels, np.ndarray)):
            raise ValueError("Expect raw data, boundaries and labels to be either all numpy arrays or lists")
        if mask is not None and not isinstance(mask, np.ndarray):
            raise ValueError("Invalid mask")

        # compute the watersheds segmentation if it was not passed
        if watershed is None:
            watershed = _compute_watershed(boundaries, use_2dws, mask, ws_kwargs, n_threads)
        features, edge_labels, z_edges = _compute_features_and_labels(raw, boundaries, watershed,
                                                                      labels, mask,
                                                                      use_2dws, feature_names,
                                                                      ignore_label, n_threads)

    # otherwise, we assume to get listlike data for raw data, boundaries and labels,
    # corresponding to multiple training data-sets
    else:
        if not (len(raw) == len(boundaries) == len(labels)):
            raise ValueError("Expect same number of raw data, boundary and label arrays")
        if watershed is not None and len(watershed) != len(raw):
            raise ValueError("Expect same number of watershed arrays as raw data")
        if mask is not None and not len(mask) == len(raw):
            raise ValueError("Expect same number of mask arrays as raw data")

        features = []
        edge_labels = []
        z_edges = []
        # compute features and labels for all training data-sets
        for train_id, (this_raw, this_boundaries, this_labels) in enumerate(zip(raw, boundaries, labels)):
            # compute the watersheds segmentation if it was not passed
            if watershed is None:
                this_mask = None if mask is None else mask[train_id]
                this_watershed = _compute_watershed(this_boundaries, use_2dws, this_mask, ws_kwargs, n_threads)
            else:
                this_watershed = watershed[train_id]
                this_mask = None

            this_features, this_edge_labels, this_z_edges = _compute_features_and_labels(this_raw, this_boundaries,
                                                                                         this_watershed, this_labels,
                                                                                         this_mask, use_2dws,
                                                                                         feature_names,
                                                                                         ignore_label, n_threads)
            features.append(this_features)
            edge_labels.append(this_edge_labels)
            # we only get z-edges if we have 2d watersheds
            if use_2dws:
                assert this_z_edges is not None
                z_edges.append(this_z_edges)

        features = np.concatenate(features, axis=0)
        edge_labels = np.concatenate(edge_labels, axis=0)
        if use_2dws:
            z_edges = np.concatenate(z_edges, axis=0)

    assert len(features) == len(edge_labels), "%i, %i" % (len(features), len(edge_labels))
    # train the random forest (2 separate random forests for in-plane and between plane edges for stacked 2d watersheds)
    if use_2dws:
        assert len(features) == len(z_edges)
        rf = elf_learn.learn_random_forests_for_xyz_edges(
            features, edge_labels, z_edges, n_threads=n_threads, **rf_kwargs
        )
    else:
        rf = elf_learn.learn_edge_random_forest(features, edge_labels, n_threads=n_threads, **rf_kwargs)
    return rf


def _load_rf(rf, use_2dws):

    def _load(rfp):
        if isinstance(rfp, str):
            if not os.path.exists(rfp):
                raise ValueError("Invalid random foerst path %s" % rfp)
            with open(rfp, 'rb') as f:
                rfp = pickle.load(f)
        return rfp

    if use_2dws:
        if len(rf) != 2:
            raise ValueError("Expect two RFs")
        rf_xy, rf_z = rf
        rf_xy, rf_z = _load(rf_xy), _load(rf_z)
        return rf_xy, rf_z
    else:
        rf = _load(rf)
        return rf

#
# multicut workflows
#


def multicut_segmentation(
    raw: np.ndarray,
    boundaries: np.ndarray,
    rf: Union[str, "RandomForestClassifier"],
    use_2dws: bool,
    multicut_solver: Union[str, callable] = "kernighan-lin",
    watershed: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    feature_names: List[str] = FEATURE_NAMES,
    weighting_scheme: Optional[str] = None,
    ws_kwargs: Dict = DEFAULT_WS_KWARGS,
    solver_kwargs: Dict = {},
    beta: float = 0.5,
    n_threads: Optional[int] = None,
    return_intermediates: bool = False,
    cost_callback: Optional[callable] = None,
) -> np.ndarray:
    """Compute instance segmentation with multicut based on edge costs derived from random forest predictions.

    Args:
        raw: The raw data.
        boundaries: The boundary maps.
        rf: The edge classifier, may also be the path to a pickled rf.
        use_2dws: Whether to run watersheds in 2d and stack them or if the watersheds passed are 2d and stacked.
        multicut_solver: Name of multicut solver in elf.segmentation.multicut or custom solver function.
        watershed: Watershed segmentation. Will be computed if None is passed.
        mask: Mask to ignore in segmentation.
        feature_names: Names of features that will be computed.
        weighting_scheme: Strategy to weight multicut edge costs by size.
        ws_kwargs: Keyword arguments for watershed function.
        solver_kwargs: Keyword arguments for multicut solver.
        beta: The boundary bias.
        n_threads: The number of threads.
        return_intermediates: Whether to also return intermediate results.
        cost_callback: Callback to modify the costs. It will be passed the costs, rag and watershed.

    Returns:
        The multicut segmentation.
    """

    # get the multicut solver
    solver = _get_solver(multicut_solver)

    # load random forest(s) if they are given as path
    if use_2dws:
        rf_xy, rf_z = _load_rf(rf, use_2dws)
    else:
        rf = _load_rf(rf, use_2dws)

    # compute watersheds if none were given
    if watershed is None:
        print("Compute watershed ...")
        watershed = _compute_watershed(boundaries, use_2dws, mask, ws_kwargs, n_threads)

    # compute rag and features
    print("Compute rag and features ...")
    rag, features = _compute_features(raw, boundaries, watershed,
                                      feature_names, use_2dws, n_threads)

    # use random forest to compute edge probabilties.
    # if we have stacked 2d watersheds, we expect two random forests,
    # one for the in-plane (xy) and one for the between-plane (z) edges
    if use_2dws:
        print("Predict edges for 2d watershed ...")
        z_edges = elf_feats.compute_z_edge_mask(rag, watershed)
        edge_probs = elf_learn.predict_edge_random_forests_for_xyz_edges(rf_xy, rf_z,
                                                                         features, z_edges,
                                                                         n_threads)
    else:
        print("Predict edges for 3d watershed ...")
        edge_probs = elf_learn.predict_edge_random_forest(rf, features, n_threads)
        z_edges = None

    # derive the edge costs from random forst probabilities
    print("Compute edge sizes ...")
    edge_sizes = elf_feats.compute_boundary_mean_and_length(rag, raw, n_threads)[:, 1]
    print("Compute edge costs ...")
    edge_costs = elf_mc.compute_edge_costs(edge_probs, edge_sizes=edge_sizes, beta=beta,
                                           z_edge_mask=z_edges, weighting_scheme=weighting_scheme)

    # apply the cost call back if given
    # and mask edges connecting to the mask label if mask was given
    if cost_callback is not None:
        edge_costs = cost_callback(edge_costs, rag, watershed)
    if mask is not None:
        edge_costs = _mask_edges(rag, edge_costs)

    # compute multicut and project to pixels
    # we pass the watershed to the solver as well, because it is needed for
    # the blockwise-multicut solver
    print("Run multicut solver ...")
    node_labels = solver(rag, edge_costs, n_threads=n_threads,
                         segmentation=watershed, **solver_kwargs)
    print("Project to volume ...")
    seg = elf_feats.project_node_labels_to_pixels(rag, node_labels, n_threads)

    if return_intermediates:
        return {"watershed": watershed,
                "rag": rag,
                "features": features,
                "edge_costs": edge_costs,
                "node_labels": node_labels,
                "segmentation": seg}
    else:
        return seg


def multicut_workflow(
    train_raw: Union[np.ndarray, List[np.ndarray]],
    train_boundaries: Union[np.ndarray, List[np.ndarray]],
    train_labels: Union[np.ndarray, List[np.ndarray]],
    raw: np.ndarray,
    boundaries: np.ndarray,
    use_2dws: bool,
    multicut_solver: Union[str, callable] = "kernighan-lin",
    train_watershed: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    watershed: np.ndarray = None,
    train_mask: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    mask: Optional[np.ndarray] = None,
    feature_names: List[str] = FEATURE_NAMES,
    weighting_scheme: Optional[str] = None,
    ws_kwargs: Dict = DEFAULT_WS_KWARGS,
    learning_kwargs: Dict = DEFAULT_RF_KWARGS,
    solver_kwargs: Dict = {},
    beta: float = 0.5,
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """Run workflow for multicut segmentation based on boundary maps with edge weights learned via random forest.

    Based on "Multicut brings automated neurite segmentation closer to human performance":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/217205318/beier_17_multicut.pdf

    Args:
        train_raw: Raw data for training data-sets, can be list if there are multiple training sets.
        train_boundaries: Boundary maps for training data-sets, can be list if there are multiple training sets.
        train_labels: Segmentation ground-truth for training data-sets, can be list if there are multiple training sets.
        raw: The raw data of the data-set to be segmented.
        boundaries: The boundary maps of the data-set to be segmented.
        use_2dws: Whether to run watersheds in 2d and stack them or if the watersheds passed are 2d and stacked.
        multicut_solver: Name of multicut solver in elf.segmentation.multicut or custom solver function.
        train_watershed: Watershed segmentation for the training data-sets can be list if there are
            multiple training sets, will be computed if None is passed.
        watershed: Watershed segmentation for the data to be segmented. Will be computed if None is passed.
        train_mask: Mask to ignre in waterheds for training datasets.
        mask: Mask to ignore in segmentation.
        feature_names: The names of features that will be computed.
        weighting_scheme: Strategy to weight multicut edge costs by size.
        ws_kwargs: Keyword arguments for watershed function.
        learning_kwargs: Keyword arguments for the random forest.
        solver_kwargs: Keyword arguments for multicut solver.
        beta: The boundary bias.
        n_threads: The number of threads.

    Returns:
        THe multicut segemntation.
    """
    # train random forest for edge classification based on the training data-sets
    rf = edge_training(train_raw, train_boundaries, train_labels, use_2dws,
                       watershed=train_watershed, mask=train_mask, feature_names=feature_names,
                       ws_kwargs=ws_kwargs, learning_kwargs=learning_kwargs,
                       n_threads=n_threads)

    # run multicut to obtain segmenation for the data to be segmented
    seg = multicut_segmentation(raw, boundaries, rf,
                                use_2dws, multicut_solver,
                                watershed=watershed, mask=mask, feature_names=feature_names,
                                weighting_scheme=weighting_scheme, ws_kwargs=ws_kwargs,
                                solver_kwargs=solver_kwargs, beta=beta, n_threads=n_threads)
    return seg


def _simple_mc_problem(input_, use_2dws, watershed, mask, weighting_scheme, ws_kwargs,
                       beta, offsets, n_threads, cost_callback):
    # check if input is affinities or boundaries and determine the spatial shape
    have_affs = input_.ndim == 4
    spatial_shape = input_.shape[1:] if have_affs else input_.shape
    if have_affs:
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]] if offsets is None else offsets
        if input_.shape[0] != len(offsets):
            raise ValueError("Incosistent number of offsets and affinity channels: %i, %i" % (len(offsets),
                                                                                              input_.shape[0]))
    if mask is not None:
        if mask.shape != spatial_shape or mask.dtype != np.dtype('bool'):
            raise ValueError("Invalid mask was passed")

    # compute watersheds if they were not passed
    if watershed is None:
        if have_affs:
            bmap = np.max(input_[1:3], axis=0) if use_2dws else np.max(input_[:3], axis=0)
        else:
            bmap = input_
        watershed = _compute_watershed(bmap, use_2dws, mask, ws_kwargs, n_threads)
        assert watershed.shape == spatial_shape
    else:
        if watershed.shape != spatial_shape:
            raise ValueError("Invalid shape of watersheds: got %s, expected %s" % str(watershed.shape,
                                                                                      spatial_shape))

    # compute rag and edge costs
    rag = elf_feats.compute_rag(watershed, n_threads=n_threads)
    if have_affs:
        edge_probs = elf_feats.compute_affinity_features(rag, input_, offsets, n_threads=n_threads)
    else:
        edge_probs = elf_feats.compute_boundary_mean_and_length(rag, input_, n_threads=n_threads)

    if use_2dws:
        z_edges = elf_feats.compute_z_edge_mask(rag, watershed)
    else:
        z_edges = None

    edge_probs, edge_sizes = edge_probs[:, 0], edge_probs[:, -1]
    edge_costs = elf_mc.compute_edge_costs(edge_probs, edge_sizes=edge_sizes, beta=beta,
                                           weighting_scheme=weighting_scheme, z_edge_mask=z_edges)

    if cost_callback is not None:
        edge_costs = cost_callback(edge_costs, rag, watershed)

    if mask is not None:
        edge_costs = _mask_edges(rag, edge_costs)
    return watershed, rag, edge_probs, edge_costs


def simple_multicut_workflow(
    input_: np.ndarray,
    use_2dws: bool,
    multicut_solver: Union[str, callable] = "kernighan-lin",
    watershed: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    weighting_scheme: Optional[str] = None,
    ws_kwargs: Dict = DEFAULT_WS_KWARGS,
    solver_kwargs: Dict = {},
    beta: float = 0.5,
    offsets: List[List[int]] = None,
    n_threads: int = None,
    return_intermediates: bool = False,
    cost_callback: Optional[callable] = None
) -> np.ndarray:
    """Run simplified multicut segmentation workflow from affinity or boundary maps.

    Adapted from "Multicut brings automated neurite segmentation closer to human performance":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/217205318/beier_17_multicut.pdf

    Args:
        input_: The input data, either boundary or affinity map.
        use_2dws: Whether to run watersheds in 2d and stack them, or if the watersheds passed are 2d and stacked.
        multicut_solver: Name of multicut solver in elf.segmentation.multicut or custom solver function.
        watershed: Watershed over-segmentation, if None is passed, the overr-segmentation will be computed.
        mask: Binary mask to exclude from the segmentation; values that are False will be excluded.
        weighting_scheme: Strategy to weight multicut edge costs by size.
        ws_kwargs: Keyword arguments for the watershed function.
        solver_kwargs: Keyword arguments for multicut solver.
        beta: The boundary bias.
        offsets: The pixel offsets for affintities, by default assume nearest neighbor offsets.
        n_threads: The number of threads.
        return_intermediates: Whether to also return intermediate results.
        cost_callback: Callback to modify the costs. It will be passed the costs, rag and watershed.

    Returns:
        The multicut segmentation.
    """
    # get the multicut solver
    solver = _get_solver(multicut_solver)

    # create the multicut problem
    watershed, rag, edge_probs, edge_costs = _simple_mc_problem(
        input_, use_2dws, watershed, mask, weighting_scheme, ws_kwargs,
        beta, offsets, n_threads, cost_callback
    )

    # solve the multicut problem and project back to segmentation
    # we pass the watershed to the solver as well, because it is needed for
    # the blockwise-multicut solver
    node_labels = solver(rag, edge_costs, n_threads=n_threads,
                         segmentation=watershed, **solver_kwargs)
    seg = elf_feats.project_node_labels_to_pixels(rag, node_labels, n_threads)

    if return_intermediates:
        return {"watershed": watershed,
                "rag": rag,
                "probs": edge_probs,
                "costs": edge_costs,
                "node_labels": node_labels,
                "segmentation": seg}
    else:
        return seg


#
# lifted multicut workflows
# Possiible additional workflows::
# - lifted_multicut_from_probabilities_workflow
# - learning based lifted multicut workflows
#


def _get_lifted_solver(solver):
    if isinstance(solver, str):
        solver = elf_lmc.get_lifted_multicut_solver(solver)
    elif not callable(solver):
        raise ValueError("Invalid lifted multicut solver")
    return solver


def lifted_multicut_from_segmentation_workflow(
    input_: np.ndarray,
    segmentation: np.ndarray,
    use_2dws: bool,
    overlap_threshold: float,
    graph_depth: int,
    same_segment_cost: float,
    different_segment_cost: float,
    lifted_multicut_solver: Union[str, callable] = "kernighan-lin",
    watershed: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    weighting_scheme: Optional[str] = None,
    ws_kwargs: Dict = DEFAULT_WS_KWARGS,
    mode: str = "all",
    solver_kwargs: Dict = {},
    beta: float = 0.5,
    offsets: List[List[int]] = None,
    n_threads: Optional[int] = None,
    return_intermediates: bool = False,
    cost_callback: Optional[callable] = None,
) -> np.ndarray:
    """ Run simple lifted multicut segmentation workflow from affinity or boundary maps and using lifted edges
    derived from priors by mapping a separate segmentation to the watersheds.

    Adapted from "Leveraging domain knowledge to improve microscopy image segmentation with lifted multicuts":
    https://doi.org/10.3389/fcomp.2019.00006

    Args:
        input_: The input data, either boundary or affinity map.
        segmentation: The segmentation used to derive priors for lifted edges.
        use_2dws: Whether to run watersheds in 2d and stack them, or if the watersheds passed are 2d and stacked.
        overlap_threshold: Minimal overlap to assign a segment id to node.
        graph_depth: Maximal graph depth up to which lifted edges will be included.
        same_segment_cost: Costs for edges between nodes with same segment id attribution.
        different_segment_cost: Costs for edges between nodes with different segment id attribution.
        lifted_multicut_solver: Name of multicut solver in elf.segmentation.multicut or custom solver function.
        watershed: Watershed over-segmentation, if None is passed, the over-segmentation will be computed.
        mask: Binary mask to exclude from the segmentation; values that are False will be excluded.
        weighting_scheme: Strategy to weight multicut edge costs by size.
        ws_kwargs: Keyword arguments for the watershed function.
        mode: Mode for insertion of lifted edges. One of:
            "all" - lifted edges will be inserted in between all nodes with attribution.
            "different" - lifted edges will only be inserted in between nodes attributed to different classes.
            "same" - lifted edges will only be inserted in between nodes attribted to the same class.
        solver_kwargs: Keyword arguments for the lifted multicut solver.
        beta: The boundary bias.
        offsets: The pixel offsets for affintities, by default assume nearest neighbor offsets.
        n_threads: The number of threads for parallelization.
        return_intermediates: Whether to also return intermediate results.
        cost_callback: Callback to modify the costs. It will be passed the costs, rag and watershed.

    Retruns:
        The lifted multicut segmentation.
    """
    # get the multicut solver
    solver = _get_lifted_solver(lifted_multicut_solver)

    # create the multicut problem
    watershed, rag, edge_probs, edge_costs = _simple_mc_problem(
        input_, use_2dws, watershed, mask, weighting_scheme, ws_kwargs,
        beta, offsets, n_threads, cost_callback
    )

    # get the lifted edges and costs
    lifted_uvs, lifted_costs = elf_feats.lifted_problem_from_segmentation(rag, watershed, segmentation,
                                                                          overlap_threshold, graph_depth,
                                                                          same_segment_cost, different_segment_cost,
                                                                          mode=mode, n_threads=n_threads)

    # solve the multicut problem and project back to segmentation
    # we pass the watershed to the solver as well, because it is needed for
    # the blockwise-multicut solver
    node_labels = solver(rag, edge_costs, lifted_uvs, lifted_costs,
                         n_threads=n_threads, segmentation=watershed, **solver_kwargs)
    seg = elf_feats.project_node_labels_to_pixels(rag, node_labels, n_threads)

    if return_intermediates:
        return {"watershed": watershed,
                "rag": rag,
                "probs": edge_probs,
                "costs": edge_costs,
                "lifted_edges": lifted_uvs,
                "lifted_costs": lifted_costs,
                "node_labels": node_labels,
                "segmentation": seg}
    else:
        return seg
