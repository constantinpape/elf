from numbers import Number
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .transform_impl import transform_subvolume


def transform_coordinate(coord, scale_factor):
    """@private
    """
    return tuple(co * sc for co, sc in zip(coord, scale_factor))


def transform_subvolume_resize(
    data: ArrayLike,
    scale_factor: Union[float, Tuple[float, ...]],
    bb: Tuple[slice, ...],
    order: int = 0,
    fill_value: Number = 0,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """Resize data in subvolume.

    Args:
        data: The input data, can be a numpy array or another array-like object.
        scale_factor: Scale factor to apply to the data.
        bb: Bounding box into the output data.
        order: Interpolation order.
        fill_value: Output value for invald coordinates.
        sigma: Sigma value used for pre-smoothing the input in order to avoid aliasing effects.

    Returns:
        The resized subvolume data.
    """
    ndim = data.ndim
    scale_factor_ = ndim * [scale_factor] if isinstance(scale_factor, Number) else scale_factor
    if len(scale_factor) != ndim:
        raise ValueError("Invalid input dimension")
    trafo = partial(transform_coordinate, scale_factor=scale_factor_)
    return transform_subvolume(data, trafo, bb, order=order, fill_value=fill_value, sigma=sigma)
