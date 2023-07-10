import tempfile
import warnings

import numpy as np
import vigra
from tqdm import tqdm

from .io import write_obj

try:
    from madcad.hashing import PositionMap
    from madcad.io import read
except ImportError:
    PositionMap = None


def vertices_and_faces_to_segmentation(
    vertices, faces, resolution=[1.0, 1.0, 1.0], shape=None, verbose=False
):
    with tempfile.NamedTemporaryFile(suffix=".obj") as f:
        tmp_path = f.name
        write_obj(tmp_path, vertices, faces)
        seg = mesh_to_segmentation(tmp_path, resolution, shape=shape, verbose=verbose)
    return seg


def mesh_to_segmentation(mesh_file, resolution=[1.0, 1.0, 1.0],
                         reverse_coordinates=False, shape=None, verbose=False):
    """ Compute segmentation volume from mesh.

    Requires madcad and pywavefront as dependency.

    Arguments:
        mesh_file [str] - path to mesh in obj format
        resolution [list[float]] - pixel resolution of the vertex coordinates
        reverse_coordinates [bool] - whether to reverse the coordinate order (default: False)
        shape [tuple[int]] - shape of the output volume.
            If None, the maximal extent of the mesh coordinates will be used as shape (default: None)
        verbose [bool] - whether to activate verbose output (default: False)
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
        voxel = [
            tuple(
                int(vv / res) for vv, res in zip(vox, resolution)
            )
            for vox in voxel
        ]
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
    seg = vigra.analysis.labelVolumeWithBackground(seg) == 2
    seg[coords] = 1
    return seg
