import multiprocessing
import numpy as np
import nifty.graph.rag as nrag
from sklearn.ensemble import RandomForestClassifier


def compute_edge_labels(rag, gt, ignore_label=None, n_threads=None):
    """ Compute edge labels by mapping ground-truth segmentation to graph nodes.

    Arguments:
        rag [RegionAdjacencyGraph] - region adjacency graph
        gt [np.ndarray] - ground-truth segmentation
        ignore_label [int or np.ndarray] - label id(s) in ground-truth
            to ignore in learning (default: None)
        n_threads [int] - number of threads (default: None)
    """
    node_labels = nrag.gridRagAccumulateLabels(gt)
    uv_ids = rag.uv_ids()

    edge_labels = (node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]).astype('uint8')

    if ignore_label is not None:
        edge_mask = np.isin(uv_ids, ignore_label)
        edge_mask = edge_mask.sum(axis=1) == 0
        assert len(edge_labels) == len(edge_mask)
        return edge_labels, edge_mask

    return edge_labels


def learn_edge_random_forest(features, labels, edge_mask=None, n_threads=None,
                             **rf_kwargs):
    """ Learn random forest for edge classification.

    Arguments:
        features [np.ndarray] - edge features
        labels [np.ndarray] - edge labels
        edge_mask [np.ndarray] - mask of edges to ignore in training (default: None)
        rf_kwargs [kwargs] - keyword arguments for sklearn.ensemble.RandomForestClassifier
    """
    if len(features) != len(labels):
        raise ValueError("Incomatble feature and label dimensions")
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads

    rf = RandomForestClassifier(**rf_kwargs)
    rf.n_jobs = n_threads
    if edge_mask is None:
        features_, labels_ = features, labels
    else:
        if len(features) != len(edge_mask):
            raise ValueError("Incomatble feature and label dimensions")
        features_, labels_ = features_[edge_mask], labels_[edge_mask]

    rf.fit(features_, labels_)
    rf.n_jobs = 1
    return rf


def learn_random_forests_for_xyz_edges(features, labels, z_edges,
                                       edge_mask=None, n_threads=None, **rf_kwargs):
    """ Learn random forests for classification of xy-and-z edges separately.

    Arguments:
        features [np.ndarray] - edge features
        labels [np.ndarray] - edge labels
        z_edges [np.ndarray] - mask for z edges
        edge_mask [np.ndarray] - mask of edges to ignore in training (default: None)
        rf_kwargs [kwargs] - keyword arguments for sklearn.ensemble.RandomForestClassifier
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads

    xy_edges = np.logical_not(z_edges)
    xy_edge_mask = None if edge_mask is None else edge_mask[xy_edges]
    rf_xy = learn_edge_random_forest(features[xy_edges], labels[xy_edges],
                                     edge_mask=xy_edge_mask,
                                     n_threads=n_threads, **rf_kwargs)

    z_edge_mask = None if edge_mask is None else edge_mask[z_edges]
    rf_z = learn_edge_random_forest(features[z_edges], labels[z_edges], edge_mask=z_edge_mask,
                                    n_threads=n_threads, **rf_kwargs)

    return rf_xy, rf_z
