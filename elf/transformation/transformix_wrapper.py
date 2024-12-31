import os
import subprocess
from shutil import rmtree
from typing import List, Tuple, Union

import numpy as np


def _set_ld_library_path(elastix_folder):
    lib_path = os.environ.get("LD_LIBRARY_PATH", "")
    elastix_lib_path = os.path.join(elastix_folder, "lib")
    os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{elastix_lib_path}"


def _write_coordinates(coordinates, out_file):
    if isinstance(coordinates, np.ndarray):
        ndim = coordinates.shape[1]
    elif isinstance(coordinates, (list, tuple)):
        ndim = len(coordinates[0])

    if ndim not in (2, 3):
        raise ValueError(f"Expect 2d or 3d input coordinates, got {ndim}")

    n_coords = len(coordinates)
    with open(out_file, "w") as f:
        f.write("index\n")
        f.write(f"{n_coords}\n")
        # TODO this can probably be vectorized
        for coord in coordinates:
            coord_str = " ".join(map(str, coord[::-1]))
            f.write(f"{coord_str}\n")


def _read_coordinates(coord_file):
    coords = []
    with open(coord_file) as f:
        for line in f:
            parsed = line.split(";")[4].lstrip().rstrip()
            parsed = parsed.split()
            x = float(parsed[3])
            y = float(parsed[4])
            try:
                z = float(parsed[5])
                coord = [z, y, x]
            except ValueError:
                coord = [y, x]
            coords.append(coord)
    return np.array(coords)


def transform_coordinates(
    coordinates: Union[np.ndarray, List, Tuple],
    transformation_file: str,
    elastix_folder: str,
) -> np.ndarray:
    """Transform coordinates with transformix.

    Args:
        coordinates: The coordinate list to transform.
        transformation_file: The file with the elastix transformation parameter.
        elastix_folder: The folder with the elastix files.

    Returns:
        The transformed coordinates.
    """

    tmp_folder = "./coords_out"
    os.makedirs(tmp_folder, exist_ok=True)
    coord_in_file = os.path.join(tmp_folder, "coords_temp.txt")

    _write_coordinates(coordinates, coord_in_file)
    transformix_bin = os.path.join(elastix_folder, "bin", "transformix")

    _set_ld_library_path(elastix_folder)
    cmd = [transformix_bin,
           "-def", coord_in_file,
           "-out", tmp_folder,
           "-tp", transformation_file]
    subprocess.run(cmd)

    coord_out_file = os.path.join(tmp_folder, "outputpoints.txt")
    out = _read_coordinates(coord_out_file)

    try:
        os.remove(coord_in_file)
    except OSError:
        pass
    try:
        rmtree(tmp_folder)
    except OSError:
        pass

    return out


def transform_volume():
    """@private
    """
    raise NotImplementedError
