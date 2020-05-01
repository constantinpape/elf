import numpy as np


def read_numpy(path):
    """ Read mesh from compressed numpy format
    """
    mesh = np.load(path)
    return mesh['verts'], mesh['faces'], mesh['normals']


def write_numpy(path, verts, faces, normals):
    """ Write mesh to compressed numpy format
    """
    np.savez_compressed(path,
                        verts=verts,
                        faces=faces,
                        normals=normals)


# TODO support different format for faces
def read_obj(path):
    """ Read mesh from obj
    """
    verts = []
    faces = []
    normals = []
    face_normals = []
    with open(path) as f:
        for line in f:
            # normal
            if line.startswith('vn'):
                normals.append([float(ll) for ll in line.split()[1:]])
            # vertex texture, hard-coded to vt 0.0 0.0 in paintera
            elif line.startswith('vt'):
                pass
            # vertex
            elif line.startswith('v'):
                verts.append([float(ll) for ll in line.split()[1:]])
            # face
            elif line.startswith('f'):
                faces.append([int(ll.split('/')[0]) for ll in line.split()[1:]])
                face_normals.append([int(ll.split('/')[2]) for ll in line.split()[1:]])

    return (np.array(verts), np.array(faces),
            np.array(normals), np.array(face_normals))


# TODO support different format for faces
def write_obj(path, verts, faces, normals, face_normals=None):
    """ Write mesh to obj
    """
    with open(path, 'w') as f:
        for vert in verts:
            f.write(" ".join(map(str, ['v'] + vert.tolist())))
            f.write("\n")

        f.write("\n")

        for normal in normals:
            f.write(" ".join(map(str, ['vn'] + normal.tolist())))
            f.write("\n")

        f.write("\n")
        f.write("vt 0.0 0.0\n")
        f.write("\n")

        if face_normals is None:
            for face in faces:
                f.write("f " + " ".join(map(str, face)))
                f.write("\n")
        else:
            for face, normal in zip(faces, face_normals):
                f.write(" ".join(["f"] + ["/".join([str(fa), "1", str(no)])
                                          for fa, no in zip(face, normal)]))
                f.write("\n")
