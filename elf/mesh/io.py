import os
from typing import Optional, Tuple, Union

import numpy as np


def read_numpy(path: Union[os.PathLike, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read mesh from compressed numpy format.

    Args:
        path: The file with the mesh saved as compressed numpy file.

    Returns:
        The vertices of the mesh.
        The faces of the mesh.
        The normals of the mesh.
    """
    mesh = np.load(path)
    return mesh["verts"], mesh["faces"], mesh["normals"]


def write_numpy(path: Union[os.PathLike, str], verts: np.ndarray, faces: np.ndarray, normals: np.ndarray):
    """Write mesh to compressed numpy format.

    Args:
        path: The path for saving the mesh.
        verts: The vertices of the mesh.
        faces: The faces of the mesh.
        normals: The normals of the mesh.
    """
    np.savez_compressed(path, verts=verts, faces=faces, normals=normals)


# TODO support different format for faces
def read_obj(path: Union[os.PathLike, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read mesh from an obj file.

    Args:
        path: The file with the mesh saved as obj.

    Returns:
        The vertices of the mesh.
        The faces of the mesh.
        The normals of the mesh.
        The face normals of the mesh.
    """
    verts = []
    faces = []
    normals = []
    face_normals = []
    with open(path) as f:
        for line in f:
            # normal
            if line.startswith("vn"):
                normals.append([float(ll) for ll in line.split()[1:]])
            # vertex texture, hard-coded to vt 0.0 0.0 in paintera
            elif line.startswith("vt"):
                pass
            # vertex
            elif line.startswith("v"):
                verts.append([float(ll) for ll in line.split()[1:]])
            # face
            elif line.startswith("f"):
                this_face = [int(ll.split("/")[0]) for ll in line.split()[1:]]
                if len(this_face) > 3:
                    this_face = this_face[:3]
                faces.append(this_face)
                try:
                    face_normals.append([int(ll.split("/")[2]) for ll in line.split()[1:]])
                except IndexError:
                    pass

    return np.array(verts), np.array(faces), np.array(normals), np.array(face_normals)


# TODO support different format for faces
def write_obj(
    path: Union[os.PathLike, str],
    verts: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    face_normals: Optional[np.ndarray] = None,
    zero_based_face_index: bool = False,
):
    """Write mesh to an obj file.

    Args:
        path: The path for saving the mesh.
        verts: The vertices of the mesh.
        faces: The faces of the mesh.
        normals: The normals of the mesh.
        face_normals: The face normals of the mesh.
        zero_based_face_index: Whether to use 0- or 1- based indexing for the faces.
    """
    with open(path, "w") as f:
        for vert in verts:
            f.write(" ".join(map(str, ["v"] + vert.tolist())))
            f.write("\n")

        f.write("\n")

        if normals is not None:
            for normal in normals:
                f.write(" ".join(map(str, ["vn"] + normal.tolist())))
                f.write("\n")

        f.write("\n")
        f.write("vt 0.0 0.0\n")
        f.write("\n")

        if not zero_based_face_index:
            faces += 1

        if face_normals is None:
            for face in faces:
                f.write("f " + " ".join(map(str, face)))
                f.write("\n")
        else:
            for face, normal in zip(faces, face_normals):
                f.write(" ".join(["f"] + ["/".join([str(fa), "1", str(no)]) for fa, no in zip(face, normal)]))
                f.write("\n")


def read_ply(path: Union[os.PathLike, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Read mesh from ply data format.

    Args:
        path: The file with the mesh saved as ply.

    Returns:
        The vertices of the mesh.
        The faces of the mesh.
    """
    verts = []
    faces = []

    is_header = True
    n_verts, n_faces = None, None
    line_id = 0

    with open(path) as f:
        for line in f:
            # parse the header
            if is_header:
                if line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])
                elif line.startswith("element face"):
                    n_faces = int(line.split()[-1])
                elif line.startswith("end_header"):
                    assert n_verts is not None
                    assert n_faces is not None
                    is_header = False
            else:
                if line_id < n_verts:  # parse a vertex
                    verts.append(list(map(float, line.split()[:3])))
                else:  # parse a face
                    face = line.split()
                    n = int(face[0])
                    face = list(map(int, face[1:n+1]))
                    faces.append(face)
                if len(faces) == n_faces:
                    break
                line_id += 1

    assert len(verts) == n_verts
    assert len(faces) == n_faces
    return np.array(verts), np.array(faces)


# https://web.archive.org/web/20161221115231/http://www.cs.virginia.edu/~gfx/Courses/2001/Advanced.spring.01/plylib/Ply.txt
def write_ply(path: Union[os.PathLike, str], verts: np.ndarray, faces: np.ndarray):
    """Write mesh to ply data format.

    Args:
        path: The path for saving the mesh.
        verts: The vertices of the mesh.
        faces: The faces of the mesh.
    """

    header = f"""ply
format ascii 1.0
element vertex {len(verts)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
    # indexing is zero-based
    if 0 not in faces:
        faces = faces - 1
    with open(path, "w") as f:
        f.write(header)
        for vert in verts:
            line = " ".join(map(str, vert.tolist()))
            f.write(line)
            f.write("\n")
        for face in faces:
            this_face = [len(face)] + face.tolist()
            line = " ".join(map(str, this_face))
            f.write(line)
            f.write("\n")
