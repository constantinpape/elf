from copy import deepcopy
import numpy as np

from .import features as elf_feats
from .import learning as elf_learn
from .import multicut as elf_mc
from .import watershed as elf_ws

FEATURE_NAMES = {'raw-edge-features', 'raw-region-features', 'boundary-edge-features'}
DEFAULT_WS_KWARGS = {'threshold': 0.25, 'sigma_seeds': 2., 'sigma_weights': 2.,
                     'min_size': 100, 'alpha': 0.9, 'pixel_pitch': None,
                     'apply_nonmax_suppression': False}
DEFAULT_RF_KWARGS = {'ignore_label': None, 'n_estimators': 200, 'max_depth': 10}


# TODO:
# - add callbacks to customize features, costs etc. to workflows for more customization
# - lmc workflows
# - multicut_workflow_from_config(config_file)


def _compute_watershed(boundaries, use_2dws, mask, ws_kwargs, n_threads):
    if use_2dws:
        return elf_ws.stacked_watershed(boundaries, mask=mask, n_threads=n_threads, **ws_kwargs)[0]
    else:
        return elf_ws.distance_transform_watershed(boundaries, mask=mask, **ws_kwargs)[0]


def _compute_features(raw, boundaries, watershed, feature_names, use_2dws, n_threads):
    if (len(FEATURE_NAMES - set(feature_names)) > 0) or (len(feature_names) == 0):
        raise ValueError("Invalid feature set")
    if raw.shape != boundaries.shape:
        raise ValueError("Shapes %s and %s do not match" % (str(raw.shape), str(boundaries.shape)))
    if raw.shape != watershed.shape:
        raise ValueError("Shapes %s and %s do not match" % (str(raw.shape), str(watershed.shape)))

    rag = elf_feats.compute_rag(watershed, n_threads=n_threads)

    features = []
    if 'raw-edge-features' in feature_names:
        feats = elf_feats.compute_boundary_features_with_filters(rag, raw, use_2dws,
                                                                 n_threads=n_threads)
        features.append(feats)
    if 'boundary-edge-features' in feature_names:
        feats = elf_feats.compute_boundary_features_with_filters(rag, boundaries, use_2dws,
                                                                 n_threads=n_threads)
        features.append(feats)
    if 'raw-region-features' in feature_names:
        feats = elf_feats.compute_region_features(rag.uvIds(), raw, watershed,
                                                  n_threads=n_threads)
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


def _compute_features_and_labels(raw, boundaries, watershed, labels,
                                 use_2dws, feature_names, ignore_label, n_threads):
    rag, features = _compute_features(raw, boundaries, watershed, feature_names, use_2dws, n_threads)

    if ignore_label is None:
        edge_labels = elf_learn.compute_edge_labels(rag, labels, n_threads=n_threads)
    else:
        edge_labels, edge_mask = elf_learn.compute_edge_labels(rag, labels, ignore_label, n_threads)
        features, edge_labels = features[edge_mask], edge_labels[edge_mask]

    if use_2dws:
        z_edges = elf_feats.compute_z_edge_mask(rag, watershed)
    else:
        z_edges = None

    return features, edge_labels, z_edges


def _get_solver(multicut_solver):
    if isinstance(multicut_solver, str):
        solver = elf_mc.get_multicut_solver(multicut_solver)
    else:
        if not callable(multicut_solver):
            raise ValueError("Invalid multicut solver")
        solver = multicut_solver
    return solver


def _mask_edges(rag, edge_costs):
    uv_ids = rag.uvIds()
    ignore_edges = (uv_ids == 0).any(axis=1)
    edge_costs[ignore_edges] = - np.abs(edge_costs).max()
    return edge_costs


# TODO support mask for training
def edge_training(raw, boundaries, labels, use_2dws, watershed=None,
                  feature_names=FEATURE_NAMES, ws_kwargs=DEFAULT_WS_KWARGS,
                  learning_kwargs=DEFAULT_RF_KWARGS, n_threads=None):
    """ Train random forest classifier for edges.

    Arguments:
        raw [np.ndarray or list[np.ndarray]] - raw data or list of raw data arrays
        boundaries [np.ndarray or list[np.ndarray]] - boundary maps or list of boundary maps
        labels [np.ndarray or list[np.ndarray]] - groundtruth segmentation
        use_2dws [bool] - whether to run watersheds in 2d and stack them
            or if the watersheds passed are 2d and stacked
        watershed [np.ndarray or list[np.ndarray]] - watershed segmentation.
            will be computed if None is passed (default: None)
        feature_names [list] - names of features that will be computed (default: all available)
        ws_kwargs [dict] - keyword arguments for watershed function
        learning_kwargs [dict] - keyword arguments for random forest
        n_threads [int] - number of threads (default: None)
    """

    # we store the ignore label(s) in the random forest kwargs,
    # but they need to be removed before this is passed to the rf implementation
    rf_kwargs = deepcopy(learning_kwargs)
    ignore_label = rf_kwargs.pop('ignore_label', None)

    # if the raw data is a numpy array, we assume that we have a single training set
    if isinstance(raw, np.ndarray):
        if (not isinstance(boundaries, np.ndarray)) or (not isinstance(labels, np.ndarray)):
            raise ValueError("Expect raw data, boundaries and labels to be either all numpy arrays or lists")
        # if mask is not None and not isinstance(mask, np.ndarray):
        #     raise ValueError("Invalid mask")

        # compute the watersheds segmentation if it was not passed
        if watershed is None:
            watershed = _compute_watershed(boundaries, use_2dws, None, ws_kwargs, n_threads)
        features, edge_labels, z_edges = _compute_features_and_labels(raw, boundaries, watershed, labels,
                                                                      use_2dws, feature_names,
                                                                      ignore_label, n_threads)
    # otherwise, we assume to get listlike data for raw data, boundaries and labels,
    # corresponding to multiple training data-sets
    else:
        if not (len(raw) == len(boundaries) == len(labels)):
            raise ValueError("Expect same number of raw data, boundary and label arrays")
        if watershed is not None and len(watershed) != len(raw):
            raise ValueError("Expect same number of watershed arrays as raw data")
        # if mask is not None and not len(mask) == len(raw):
        #     raise ValueError("Expect same number of mask arrays as raw data")

        features = []
        edge_labels = []
        z_edges = []
        # compute features and labels for all training data-sets
        for train_id, (this_raw, this_boundaries, this_labels) in enumerate(zip(raw, boundaries, labels)):
            # compute the watersheds segmentation if it was not passed
            if watershed is None:
                # this_mask = None if mask is None else mask[train_id]
                this_watershed = _compute_watershed(this_boundaries, use_2dws, None, ws_kwargs, n_threads)
            else:
                this_watershed = watershed[train_id]

            this_features, this_edge_labels, this_z_edges = _compute_features_and_labels(this_raw, this_boundaries,
                                                                                         this_watershed, this_labels,
                                                                                         use_2dws, feature_names,
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

    assert len(features) == len(edge_labels), "%i, %i" % (len(features),
                                                          len(edge_labels))

    # train the random forest (2 separate random forests for in-plane and between plane edges for stacked 2d watersheds)
    if use_2dws:
        assert len(features) == len(z_edges)
        rf = elf_learn.learn_random_forests_for_xyz_edges(features, edge_labels, z_edges, n_threads=n_threads,
                                                          **rf_kwargs)
    else:
        rf = elf_learn.learn_edge_random_forest(features, edge_labels, n_threads=n_threads, **rf_kwargs)
    return rf


def multicut_segmentation(raw, boundaries, rf,
                          use_2dws, multicut_solver, watershed=None, mask=None,
                          feature_names=FEATURE_NAMES, weighting_scheme=None,
                          ws_kwargs=DEFAULT_WS_KWARGS, solver_kwargs={},
                          beta=0.5, n_threads=None, return_intermediates=False):
    """ Instance segmentation with multicut with edge costs
    derived from random forest predictions.

    Arguments:
        raw [np.ndarray] - raw data
        boundaries [np.ndarray] - boundary maps
        rf [RandomForestClassifier] - edge classifier
        use_2dws [bool] - whether to run watersheds in 2d and stack them
            or if the watersheds passed are 2d and stacked
        multicut_solver [str or callable] - name of multicut solver in elf.segmentation.multicut
            or custom solver function
        watershed [np.ndarray] - watershed segmentation.
            will be computed if None is passed (default: None)
        mask [np.ndarray] - mask to ignore in segmentation (default: None)
        feature_names [list] - names of features that will be computed (default: all available)
        weighting_scheme [str] - strategy to weight multicut edge costs by size (default: None)
        ws_kwargs [dict] - keyword arguments for watershed function
        solver_kwargs [dict] - keyword arguments for multicut solver (default: {})
        beta [float] - boundary bias (default: 0.5)
        n_threads [int] - number of threads (default: None)
        return_intermediates [bool] - whether to also return intermediate results (default: False)
    """

    # get the multicut solver
    solver = _get_solver(multicut_solver)

    # compute watersheds if none were given
    if watershed is None:
        watershed = _compute_watershed(boundaries, use_2dws, mask, ws_kwargs, n_threads)

    # compute rag and features
    rag, features = _compute_features(raw, boundaries, watershed,
                                      feature_names, use_2dws, n_threads)

    # use random forest to compute edge probabilties.
    # if we have stacked 2d watersheds, we expect two random forests,
    # one for the in-plane (xy) and one for the between-plane (z) edges
    if use_2dws:
        rf_xy, rf_z = rf
        z_edges = elf_feats.compute_z_edge_mask(rag, watershed)
        edge_probs = elf_learn.predict_edge_random_forests_for_xyz_edges(rf_xy, rf_z,
                                                                         features, z_edges,
                                                                         n_threads)
    else:
        edge_probs = elf_learn.predict_edge_random_forest(rf, features, n_threads)
        z_edges = None

    # derive the edge costs from random forst probabilities
    edge_sizes = elf_feats.compute_boundary_mean_and_length(rag, raw, n_threads)[:, 1]
    costs = elf_mc.compute_edge_costs(edge_probs, edge_sizes=edge_sizes, beta=beta,
                                      z_edge_mask=z_edges, weighting_scheme=weighting_scheme)
    if mask is not None:
        costs = _mask_edges(rag, costs)

    # compute multicut and project to pixels
    # we pass the watershed to the solver as well, because it is needed for
    # the blockwise-multicut solver
    node_labels = solver(rag, costs, n_threads=n_threads,
                         segmentation=watershed, **solver_kwargs)
    seg = elf_feats.project_node_labels_to_pixels(rag, node_labels, n_threads)

    if return_intermediates:
        return {'watershed': watershed,
                'rag': rag,
                'features': features,
                'costs': costs,
                'node_labels': node_labels,
                'segmentation': seg}
    else:
        return seg


def multicut_workflow(train_raw, train_boundaries, train_labels,
                      raw, boundaries, use_2dws, multicut_solver,
                      train_watershed=None, watershed=None, mask=None,
                      feature_names=FEATURE_NAMES, weighting_scheme=None,
                      ws_kwargs=DEFAULT_WS_KWARGS, learning_kwargs=DEFAULT_RF_KWARGS,
                      solver_kwargs={}, beta=0.5, n_threads=None):
    """ Run workflow for multicut segmentation based on boundary maps with edge weights learned via random forest.

    Based on "Multicut brings automated neurite segmentation closer to human performance":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/217205318/beier_17_multicut.pdf

    Arguments:
        train_raw [np.ndarray or list[np.ndarray]] - raw data for training data-sets,
            can be list if there are multiple training sets
        train_boundaries [np.ndarray or list[np.ndarray]] - boundary maps for training data-sets,
            can be list if there are multiple training sets
        train_labels [np.ndarray or list[np.ndarray]] - segmentation ground-truth for training data-sets,
            can be list if there are multiple training sets
        raw [np.ndarray] - raw data of the data-set to be segmented
        boundaries [np.ndarray] - boundary maps of the data-set to be segmented
        use_2dws [bool] - whether to run watersheds in 2d and stack them
            or if the watersheds passed are 2d and stacked
        multicut_solver [str or callable] - name of multicut solver in elf.segmentation.multicut
            or custom solver function
        train_watershed [np.ndarray] - watershed segmentation for the training data-sets
            can be list if there are multiple training sets,
            will be computed if None is passed (default: None)
        watershed [np.ndarray] - watershed segmentation for the data to be segmented.
            will be computed if None is passed (default: None)
        mask [np.ndarray] - mask to ignore in segmentation (default: None)
        feature_names [list] - names of features that will be computed (default: all available)
        weighting_scheme [str] - strategy to weight multicut edge costs by size (default: None)
        ws_kwargs [dict] - keyword arguments for watershed function
        learning_kwargs [dict] - keyword arguments for the random forest
        solver_kwargs [dict] - keyword arguments for multicut solver (default: {})
        beta [float] - boundary bias (default: 0.5)
        n_threads [int] - number of threads (default: None)
    """
    # train random forest for edge classification based on the training data-sets
    rf = edge_training(train_raw, train_boundaries, train_labels, use_2dws,
                       watershed=train_watershed, feature_names=feature_names,
                       ws_kwargs=ws_kwargs, learning_kwargs=learning_kwargs,
                       n_threads=n_threads)

    # run multicut to obtain segmenation for the data to be segmented
    seg = multicut_segmentation(raw, boundaries, rf,
                                use_2dws, multicut_solver,
                                watershed=watershed, mask=mask, feature_names=feature_names,
                                weighting_scheme=weighting_scheme, ws_kwargs=ws_kwargs,
                                solver_kwargs=solver_kwargs, beta=beta, n_threads=n_threads)
    return seg


def simple_multicut_workflow(input_, use_2dws, multicut_solver, watershed=None, mask=None,
                             weighting_scheme=None, ws_kwargs=DEFAULT_WS_KWARGS,
                             solver_kwargs={}, beta=0.5, offsets=None, n_threads=None,
                             return_intermediates=False):
    """ Run simplified multicut segmentation workflow from affinity or boundary maps.

    Adapted from "Multicut brings automated neurite segmentation closer to human performance":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/217205318/beier_17_multicut.pdf

    Arguments:
        input_ [np.ndarray] - input data, either boundary or affinity map
        use_2dws [bool] - whether to run watersheds in 2d and stack them
            or if the watersheds passed are 2d and stacked
        multicut_solver [str or callable] - name of multicut solver in elf.segmentation.multicut
            or custom solver function
        watershed [np.ndarray] - watershed over-segmentation, if None is passed,
            the overr-segmentation witll be computted (default: None)
        mask [np.ndarray] - binary mask to exclude from the segmentation;
            values that are False will be excluded (default: None)
        weighting_scheme [str] - strategy to weight multicut edge costs by size (default: None)
        ws_kwargs [dict] - keyword arguments for the watershed function (default: None)
        solver_kwargs [dict] - keyword arguments for multicut solver (default: {})
        beta [float] - boundary bias (default: 0.5)
        offsets [list] - pixel offsets for affintities, by default assume nearest neighbor offsets (default: None)
        n_threads [int] - number of threads (default: None)
        return_intermediates [bool] - whether to also return intermediate results (default: False)
    """
    # get the multicut solver
    solver = _get_solver(multicut_solver)

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

    if mask is not None:
        edge_costs = _mask_edges(rag, edge_costs)

    # solve the multicut problem and project back to segmentation
    # we pass the watershed to the solver as well, because it is needed for
    # the blockwise-multicut solver
    node_labels = solver(rag, edge_costs, n_threads=n_threads,
                         segmentation=watershed, **solver_kwargs)
    seg = elf_feats.project_node_labels_to_pixels(rag, node_labels, n_threads)

    if return_intermediates:
        return {'watershed': watershed,
                'rag': rag,
                'probs': edge_probs,
                'costs': edge_costs,
                'node_labels': node_labels,
                'segmentation': seg}
    else:
        return seg
