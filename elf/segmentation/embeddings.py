import numpy as np
import vigra

from scipy.ndimage import shift
from sklearn.decomposition import PCA

from skimage.segmentation import slic


def embedding_pca(embeddings, n_components=3, as_rgb=True):
    """
    """

    if as_rgb and n_components != 3:
        raise ValueError("")

    pca = PCA(n_components=n_components)
    embed_dim = embeddings.shape[0]
    shape = embeddings.shape[1:]

    embed_flat = embeddings.reshape(embed_dim, -1).T
    embed_flat = pca.fit_transform(embed_flat).T
    embed_flat = embed_flat.reshape((n_components,) + shape)

    if as_rgb:
        embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
        embed_flat = embed_flat.astype('uint8')

    return embed_flat


# TODO which slic implementation to use? vigra or skimage?
# TODO can we run slic on the full embeddign space without RGB PCA?
# TODO is there a slic implementation that can do this?
# TODO slic parameter?
def embedding_slic(embeddings, run_pca=True):
    """
    """

    if run_pca:
        embeddings = embedding_pca(embeddings, as_rgb=True)
    embeddings = embeddings.transpose((1, 2, 0)) if embeddings.ndim == 3 else\
        embeddings.transpose((1, 2, 3, 0))
    seg = slic(embeddings, convert2lab=True)
    # print(embeddings.shape)
    # seg = slicSuperpixels(embeddings[..., 0], intensityScaling=1., seedDistance=1)[0]
    return seg


def _embeddings_to_probabilities(embed1, embed2, delta, embedding_axis):
    probs = (2 * delta - np.linalg.norm(embed1 - embed2, axis=embedding_axis)) / (2 * delta)
    probs = np.maximum(probs, 0) ** 2
    return probs


def edge_probabilities_from_embeddings(embeddings, segmentation, rag, delta):
    # TODO this looks inefficient :(
    n_nodes = rag.numberOfNodes
    embed_dim = embeddings.shape[0]

    segmentation = segmentation.astype('uint32')
    mean_embeddings = np.zeros((n_nodes, embed_dim), dtype='float32')
    for cid in range(embed_dim):
        mean_embed = vigra.analysis.extractRegionFeatures(embeddings[cid],
                                                          segmentation, features=['mean'])['mean']
        mean_embeddings[:, cid] = mean_embed

    uv_ids = rag.uvIds()
    embed_u = mean_embeddings[uv_ids[:, 0]]
    embed_v = mean_embeddings[uv_ids[:, 1]]
    edge_probabilities = 1. - _embeddings_to_probabilities(embed_u, embed_v, delta, embedding_axis=1)
    return edge_probabilities


# could probably be implemented more efficiently with shift kernels
# instead of explicit call to shift
# (or implement in C++ to save memory)
def embeddings_to_affinities(embeddings, offsets, delta, invert=False):
    """ Convert embeddings to affinities.

    Computes the affinity according to the formula
    a_ij = max((2 * delta - ||x_i - x_j||) / 2 * delta, 0) ** 2,
    where delta is the push force used in training the embeddings.
    Introduced in "Learning Dense Voxel Embeddings for 3D Neuron Reconstruction":
    https://arxiv.org/pdf/1909.09872.pdf

    Arguments:
        embeddings [np.ndarray] - the array with embeddings
        offsets [list] - the offset vectors for which to compute affinities
        delta [float] - the delta factor used in the push force when training the embeddings
        invert [bool] - whether to invert the affinites (default=False)
    """
    ndim = embeddings.ndim - 1
    if not all(len(off) == ndim for off in offsets):
        raise ValueError("Incosistent dimension of offsets and embeddings")

    n_channels = len(offsets)
    shape = embeddings.shape[1:]
    affinities = np.zeros((n_channels,) + shape, dtype='float32')

    for cid, off in enumerate(offsets):
        # we need to  shift in the other direction in order to
        # get the correct offset
        # also, we need to add a zero shift in the first axis
        shift_off = [0] + [-o for o in off]
        # we could also shift via np.pad and slicing
        shifted = shift(embeddings, shift_off, order=0, prefilter=False)
        affs = _embeddings_to_probabilities(embeddings, shifted, delta, embedding_axis=0)
        affinities[cid] = affs

    if invert:
        affinities = 1. - affinities

    return affinities
