import multiprocessing
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import nifty.graph.rag as nrag
from sklearn.ensemble import RandomForestClassifier


def compute_edge_labels(
    rag, gt: np.ndarray, ignore_label: Optional[Union[int, Sequence[int]]] = None, n_threads: Optional[int] = None
) -> np.ndarray:
    """Compute edge labels by mapping ground-truth segmentation to graph nodes.

    Args:
        rag: The region adjacency graph.
        gt: The ground-truth segmentation.
        ignore_label: Label id(s) in the ground-truth to ignore in learning.
        n_threads: The number of threads.

    Returns:
        The edge labels.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads

    node_labels = nrag.gridRagAccumulateLabels(rag, gt, n_threads)
    uv_ids = rag.uvIds()

    edge_labels = (node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]).astype("uint8")

    if ignore_label is not None:
        mapped_uv_ids = node_labels[uv_ids]
        edge_mask = np.isin(mapped_uv_ids, ignore_label)
        edge_mask = edge_mask.sum(axis=1) == 0
        assert len(edge_labels) == len(edge_mask)
        return edge_labels, edge_mask

    return edge_labels


def learn_edge_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    edge_mask: Optional[np.ndarray] = None,
    n_threads: Optional[int] = None,
    **rf_kwargs,
) -> RandomForestClassifier:
    """Learn random forest for edge classification.

    Args:
        features: The edge features.
        labels: The edge labels.
        edge_mask: The mask of edges to ignore in training.
        n_threads: The number of threads.
        rf_kwargs: Keyword arguments for sklearn.ensemble.RandomForestClassifier.

    Returns:
        The trained random forest.
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


def learn_random_forests_for_xyz_edges(
    features: np.ndarray,
    labels: np.ndarray,
    z_edges: np.ndarray,
    edge_mask: Optional[np.ndarray] = None,
    n_threads: Optional[int] = None,
    **rf_kwargs
) -> Tuple[RandomForestClassifier, RandomForestClassifier]:
    """Learn random forests for classification of xy-and-z edges separately.

    Args:
        features: The edge features.
        labels: The edge labels.
        z_edges: The mask for z edges.
        edge_mask: The mask of edges to ignore in training.
        n_threads: The number of threads.
        rf_kwargs: Keyword arguments for sklearn.ensemble.RandomForestClassifier.

    Returns:
        Trained random forest classifer for in-plane edges.
        Trained random forest classifer for between-plane edges.
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


def predict_edge_random_forest(
    rf: RandomForestClassifier, features: np.ndarray, n_threads: Optional[int] = None
) -> np.ndarray:
    """Predict edge probablities with random forest.

    Args:
        rf: The random forest classifier.
        features: The edge features.
        n_threads: The number of threads.

    Returns:
        The edge probabilities.
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    prev_njobs = rf.n_jobs
    rf.n_jobs = n_threads

    edge_probs = rf.predict_proba(features)[:, 1]
    assert len(edge_probs) == len(features)

    rf.n_jobs = prev_njobs
    return edge_probs


def predict_edge_random_forests_for_xyz_edges(
    rf_xy: RandomForestClassifier,
    rf_z: RandomForestClassifier,
    features: np.ndarray,
    z_edge_mask: np.ndarray,
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """Predict edge probablities with random forests for xy- and z- edges separately.

    Args:
        rf_xy: The random forest trained for xy edges.
        rf_z: The random forest trained for z edges.
        features: The edge features.
        z_edges: The mask for z edges.
        n_threads: The number of threads.

    Returns:
        The edge probabilities.
    """
    edge_probs = np.zeros(len(features))

    xy_edge_mask = np.logical_not(z_edge_mask)
    edge_probs[xy_edge_mask] = predict_edge_random_forest(rf_xy, features[xy_edge_mask], n_threads)

    edge_probs[z_edge_mask] = predict_edge_random_forest(rf_z, features[z_edge_mask], n_threads)

    return edge_probs
