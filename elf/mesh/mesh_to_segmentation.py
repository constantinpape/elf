import tempfile
import warnings
from typing import Optional, Tuple

import numpy as np
import vigra
from tqdm import tqdm

from elf.parallel import label
from .io import write_obj

try:
    from madcad.hashing import PositionMap
    from madcad.io import read
except ImportError:
    PositionMap = None


def vertices_and_faces_to_segmentation(
    vertices, faces, resolution=(1.0, 1.0, 1.0), shape=None, verbose=False, block_shape=None
):
    """@private
    """
    with tempfile.NamedTemporaryFile(suffix=".obj") as f:
        tmp_path = f.name
        write_obj(tmp_path, vertices, faces)
        seg = mesh_to_segmentation(tmp_path, resolution, shape=shape, verbose=verbose, block_shape=block_shape)
    return seg


def mesh_to_segmentation(
    mesh_file: str,
    resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    reverse_coordinates: bool = False,
    shape: Tuple[int, int, int] = None,
    verbose: bool = False,
    block_shape: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Transform a mesh into a volumetric binary segmentation mask.

    Requires madcad and pywavefront as dependency.

    Args:
        mesh_file: Path to the mesh in obj format.
        resolution: Pixel resolution of the vertex coordinates.
        reverse_coordinates: Whether to reverse the coordinate order.
        shape: Shape of the output volume. If None, the maximal extent of the mesh coordinates will be used.
        verbose: Whether to activate verbose output.
        block_shape: Block_shape to parallelize the computation.

    Returns:
        The binary segmentation mask.
    """
    if PositionMap is None:
        raise RuntimeError("Need madcad dependency for mesh_to_seg functionality.")

    mesh = read(mesh_file)
    hasher = PositionMap(1)

    voxels = set()
    for face in tqdm(mesh.faces, disable=not verbose):
        voxel = hasher.keysfor(mesh.facepoints(face))
        if reverse_coordinates:
            voxel = [vox[::-1] for vox in voxel]
        voxel = [tuple(int(vv / res) for vv, res in zip(vox, resolution)) for vox in voxel]
        voxels.update(voxel)

    voxels = np.array(list(voxels))
    if verbose:
        print("Number of coordinates", len(voxels))

    max_vox = voxels.max(axis=0)
    if shape is None:
        shape = np.ceil(max_vox) + 1
    elif any(mv >= sh for mv, sh in zip(max_vox, shape)):
        warnings.warn(f"Clipping voxel coordinates {max_vox} that are larger than the shape {shape}.")
        voxels[:, 0] = np.clip(voxels[:, 0], 0, shape[0] - 1)
        voxels[:, 1] = np.clip(voxels[:, 1], 0, shape[1] - 1)
        voxels[:, 2] = np.clip(voxels[:, 2], 0, shape[2] - 1)

    if verbose:
        print("Computing segmentation volume of shape", shape)

    seg = np.ones(shape, dtype="uint8")
    coords = tuple(voxels[:, ii] for ii in range(voxels.shape[1]))
    seg[coords] = 0

    if block_shape is None:
        seg_out = vigra.analysis.labelVolumeWithBackground(seg) == 2
    else:
        seg_out = np.zeros_like(seg)
        seg_out = label(seg, seg_out, with_background=True, block_shape=block_shape) == 2

    seg_out[coords] = 1
    return seg_out
