from typing import Tuple

import numpy as np

# The skeletonize_3d function was removed in recent versions of scikit-image in favor
# of skeletonize, which now handles both 2d and 3d inputs. To keep compatability with older
# versions we do this try except here.
try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    from skimage.morphology import skeletonize as skeletonize_3d

from skan import csr


def thinning(
    obj: np.array,
    resolution: Tuple[float, ...],
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Skeletonize object with thinning based method.

    Wrapper around implementation from
    https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.skeletonize

    Args:
        obj: Binary object mask.
        resolution: Size of the voxels in physical unit.
        args: Additional positional arguments. For signature compatability, will be ignored.
        kwargs: Additional keyword arguments. For signature compatability, will be ignored.

    Returns:
        The nodes of the skeleton.
        The edges between skeleton nodes.
    """

    # Skeletonize with thinning.
    vol = skeletonize_3d(obj)

    # Use skan to extact skeleon node coordinates and edges.
    adj_mat, nodes = csr.skeleton_to_csgraph(vol, spacing=resolution)

    # Convert nodes from tuple to numpy array.
    nodes = np.concatenate([n[:, None] for n in nodes], axis=1).astype("uint64")

    # Convert graph to uv-list representation.
    n_nodes = len(nodes)
    graph = csr.csr_to_nbgraph(adj_mat)
    edges = np.array(
        [[u, v] for u in range(n_nodes) for v in graph.neighbors(u) if u < v], dtype="uint64"
    )

    # Return node coordinate list and edges.
    return nodes, edges
