from numbers import Number
from functools import partial
from .transform_impl import transform_subvolume


def transform_coordinate(coord, scale_factor):
    return tuple(co * sc for co, sc in zip(coord, scale_factor))


def transform_subvolume_resize(data, scale_factor, bb,
                               order=0, fill_value=0, sigma=None):
    """ Apply affine transformation to subvolume.

    Arguments:
        data [array_like] - input data
        scale_factor [float or iterable[float]] - scale factor to apply to the data
        bb [tuple[slice]] - bounding box into the output data
        order [int] - interpolation order (default: 0)
        fill_value [scalar] - output value for invald coordinates (default: 0)
        sigma [float] - sigma value used for pre-smoothing the input
            in order to avoid aliasing effects (default: None)
    """
    ndim = data.ndim
    scale_factor_ = ndim * [scale_factor] if isinstance(scale_factor, Number) else scale_factor
    if len(scale_factor) != ndim:
        raise ValueError("Invalid input dimension")
    trafo = partial(transform_coordinate, scale_factor=scale_factor_)
    return transform_subvolume(data, trafo, bb,
                               order=order, fill_value=fill_value,
                               sigma=sigma)
