import numpy as np
import vigra
from tqdm import tqdm

try:
    from madcad.hashing import PositionMap
    from madcad.io import read
except ImportError:
    PositionMap = None


def mesh_to_segmentation(mesh_file, resolution,
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
    faces = tqdm(mesh.faces) if verbose else mesh.faces
    for face in faces:
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
        shape = max_vox + 1
    else:
        assert all(mv < sh for mv, sh in zip(max_vox, shape)), f"{max_vox}, {shape}"
    if verbose:
        print("Computing segmentation volume of shape", shape)

    seg = np.ones(shape, dtype='uint8')
    coords = tuple(
        voxels[:, ii] for ii in range(voxels.shape[1])
    )
    seg[coords] = 0
    seg = vigra.analysis.labelVolumeWithBackground(seg)
    return seg
