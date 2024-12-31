import argparse
from typing import List, Optional

import numpy as np
try:
    import napari
except ImportError:
    napari = None

from ..io import open_file, is_dataset


def view_container(
    path: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    lazy: bool = False,
    show: bool = True,
) -> "napari.Viewer":
    """Display the contents of an hdf5 file or similar in napari.

    Args:
        path: The filepath to the container.
        include: Optional list of datasets to include.
            If given, only the datasets with names matching the list will be shown.
        exclude: Optional list of datasets to exclude.
        lazy: Whether to use lazy loading for the data in the container.
        show: Whether to start the napari viewer.

    Returns:
        The napari viewer.
    """
    assert napari is not None, "Requires napari"
    assert sum((include is not None, exclude is not None)) != 2

    v = napari.Viewer()

    def add_layer(v, node, name):
        print("Add:", name)
        data = node if lazy else node[:]
        if np.dtype(data.dtype) in (
            np.dtype("uint32"), np.dtype("uint64"), np.dtype("int32"), np.dtype("int64")
        ):
            v.add_labels(data, name=name)
        else:
            v.add_image(data, name=name)

    def visitor(name, node):
        if not is_dataset(node):
            return

        if include and name in include:
            add_layer(v, node, name)
        elif exclude and name not in exclude:
            add_layer(v, node, name)
        elif (exclude is None) and (include is None):
            add_layer(v, node, name)

    with open_file(path, "r") as f:
        f.visititems(visitor)
        if show:
            napari.run()
    return v


def main():
    """@private
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-i", "--include", default=None, nargs="+")
    parser.add_argument("-x", "--exclude", default=None, nargs="+")
    args = parser.parse_args()
    view_container(args.path, args.include, args.exclude)
