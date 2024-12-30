import json
import os
import warnings
from typing import Dict, List, Optional, Union

import numpy as np

from .affine import (affine_matrix_2d, affine_matrix_3d,
                     scale_from_matrix, translation_from_matrix)

SUPPORTED_NGFF_VERSIONS = ("0.4",)


def _parse_04_transformation(ngff_trafo, indices):
    assert len(ngff_trafo) <= 2

    scale, translation = None, None
    for trafo in ngff_trafo:
        trafo_type = trafo["type"]
        assert trafo_type in ("scale", "translation"), f"Expected scale or translation transform, got {trafo_type}"
        if trafo_type == "scale":
            scale = trafo["scale"]
        if trafo_type == "translation":
            translation = trafo["translation"]

    assert sum((scale is not None, translation is not None)) > 0
    if scale is not None and translation is not None:
        assert len(scale) == len(translation)

    if indices and scale:
        scale = [scale[index] for index in indices]
    if indices and translation:
        translation = [translation[index] for index in indices]

    ndim = len(translation) if scale is None else len(scale)
    if ndim == 2:
        transform = affine_matrix_2d(scale=scale, translation=translation)
    elif ndim == 3:
        transform = affine_matrix_3d(scale=scale, translation=translation)
    else:
        raise RuntimeError(f"Only support 2d or 3d affines, got {ndim}")
    return transform


def _parse_transformation(ngff_trafo, version, indices):
    if version == "0.4":
        return _parse_04_transformation(ngff_trafo, indices)
    else:
        raise RuntimeError(f"Unsupported version {version}")


def _get_04_axis_indices(multiscales, axes):
    indices = []
    for i, ax in enumerate(multiscales["axes"]):
        if ax["name"] in axes:
            indices.append(i)
    assert len(indices) == len(axes)
    return indices


def _get_axis_indices(multiscales, axes, version):
    if version == "0.4":
        return _get_04_axis_indices(multiscales, axes)
    else:
        raise RuntimeError(f"Unsupported version {version}")


def ngff_to_native(
    multiscales: Union[str, List[Dict], Dict], scale_level: int = 0, axes: Optional[str] = None
) -> np.ndarray:
    """Convert NGFF transformation to affine transformation matrix.

    Args:
        multiscales: The ngff multiscales metadata.
            Can be either a filepath to the zarr file or a dict containing the deserialzed ngff metadata.
        scale_level: The scale level for which to compute the transformation.
        axes: Subset of axes for which to compute the transformation, e.g. 'zyx' to compute only for spatial axes.

    Returns:
        The affine transformation matrix.
    """
    if isinstance(multiscales, str):
        assert os.path.exists(multiscales)
        if os.path.isdir(multiscales):
            multiscales = os.path.join(multiscales, ".zattrs")
        with open(multiscales) as f:
            multiscales = json.load(f)

    if isinstance(multiscales, dict) and len(multiscales) == 1:
        assert "multiscales" in multiscales
        multiscales = multiscales["multiscales"]
    if isinstance(multiscales, list):
        multiscales = multiscales[0]
    assert isinstance(multiscales, dict)

    if "version" in multiscales:
        version = multiscales["version"]
    else:
        version = SUPPORTED_NGFF_VERSIONS[-1]
        warnings.warn(f"Could not find version field in multiscales metadata, assuming latest version: {version}")
    if version not in SUPPORTED_NGFF_VERSIONS:
        raise RuntimeError(
            f"NGFF version {version} is not in supported versions: {SUPPORTED_NGFF_VERSIONS}"
        )

    indices = None if axes is None else _get_axis_indices(multiscales, axes, version)
    transformation = multiscales["datasets"][scale_level].get("coordinateTransformations", None)
    if transformation is not None:
        transformation = _parse_transformation(transformation, version, indices)

    if "coordinateTransformations" in multiscales:
        global_transformation = multiscales["coordinateTransformations"]
        global_transformation = _parse_transformation(global_transformation, version, indices)
        if transformation is None:
            transformation = global_transformation
        else:
            assert transformation.shape == global_transformation.shape
            transformation = transformation @ global_transformation

    return transformation


def _to_04_trafo(transformation):
    trafos = []
    scale = scale_from_matrix(transformation)
    if any(sc != 1.0 for sc in scale):
        trafos.append({"type": "scale", "scale": scale.tolist()})
    translation = translation_from_matrix(transformation)
    if any(trans != 0.0 for trans in translation):
        trafos.append({"type": "translation", "translation": translation.tolist()})
    return {"coordinateTransformations": trafos}


# TODO implement expanding to axes, e.g. expanding zyx to tczyx trafo
def native_to_ngff(transformation: np.ndarray, version: Optional[str] = None) -> Dict:
    """Convert affine transformation matrix to NGFF transformation.

    Args:
        transformation: The transformation matrix.
        version: The ngff version to use. Will use the latest supported version by default.

    Returns:
        The NGFF transformation.
    """
    if transformation.shape not in [(3, 3), (4, 4)]:
        raise ValueError(
            f"Invalid shape of the transformation matrix: {transformation.shape}, expect 3x3 or 4x4 matrix"
        )

    if version is None:
        version = SUPPORTED_NGFF_VERSIONS[-1]
    if version == "0.4":
        trafo = _to_04_trafo(transformation)
    else:
        raise RuntimeError(
            f"NGFF version {version} is not in supported versions: {SUPPORTED_NGFF_VERSIONS}"
        )
    return trafo
