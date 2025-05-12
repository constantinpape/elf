from numbers import Number
from typing import Optional, Union, Tuple

import numpy as np
import zarr
from numpy.typing import ArrayLike

SUPPORTED_CODECS = ("blosc", "gzip", "zstd")
"""Supported compression codecs
"""


def _check_consistency(data, dtype, shape):
    # Data was not passed, so dtype and shape are required.
    if data is None:
        if dtype is None:
            raise ValueError("You have to pass a dtype if data is not passed.")

        if shape is None:
            raise ValueError("You have to pass the shape if data is not passed.")

    # Data was passed. dtype and shape are not required.
    # If given, we check that they are consistent, otherwise we derive them from the data.
    else:
        if dtype is None:
            dtype = data.dtype
        elif np.dtype(dtype) != np.dtype(data.dtype):
            raise ValueError(
                "You have passed both data and dtype arguments and the values are inconsistent: "
                f"{dtype} != {data.dtype}."
            )

        if shape is None:
            shape = data.shape
        elif shape != data.shape:
            raise ValueError(
                "You have passed both data and shape arguments and the values are inconsistent: "
                f"{shape} != {data.shape}."
            )

    return data, dtype, shape


def _translate_compression(compression):
    if compression is None:
        return None

    if compression == "blosc":
        compressor = [zarr.codecs.BloscCodec()]
    elif compression == "gzip":
        compressor = [zarr.codecs.GzipCodec()]
    elif compression == "zstd":
        compressor = [zarr.codecs.ZstdCodec()]
    else:
        raise ValueError(
            f"The argument {compression} is not supported. Supported codecs: {','.join(SUPPORTED_CODECS)}"
        )

    return compressor


def _create_dataset_impl(
    self,
    name: str,
    data: Optional[ArrayLike] = None,
    dtype: Optional[Union[str, np.dtype]] = None,
    shape: Optional[Tuple[int, ...]] = None,
    compression: Optional[str] = None,
    fillvalue: Optional[Number] = None,
    **kwargs,
) -> zarr.Array:
    data, dtype, shape = _check_consistency(data, dtype, shape)
    compressors = _translate_compression(compression)
    array = self.create_array(
        name=name, shape=shape, dtype=dtype, fill_value=fillvalue, compressors=compressors, **kwargs
    )
    if data is not None:
        array[:] = data
    return array


def _check_dataset_consistency(array, data, dtype, shape, name):
    if data is not None:
        data, dtype, shape = _check_consistency(data, dtype, shape)
    if dtype is not None and np.dtype(dtype) != np.dtype(array.dtype):
        raise ValueError(
            f"The zarr array @ {name} already exists and has inconsistent datatype with the one you have passed: "
            f"{dtype} != {array.dtype}.")
    if shape is not None and tuple(shape) != tuple(array.shape):
        raise ValueError(
            f"The zarr array @ {name} already exists and has inconsistent shape with the one you have passed: "
            f"{shape} != {shape}.")
    if data is not None:
        array[:] = data


def _require_dataset_impl(
    self,
    name: str,
    data: Optional[ArrayLike] = None,
    dtype: Optional[Union[str, np.dtype]] = None,
    shape: Optional[Tuple[int, ...]] = None,
    compression: Optional[str] = None,
    fillvalue: Optional[Number] = None,
    **kwargs,
) -> zarr.Array:
    if name in self:
        array = self[name]
        _check_dataset_consistency(array, data, dtype, shape, name)
        return array
    # This is already monkey-patched.
    array = self.create_dataset(
        name=name, data=data, dtype=dtype, shape=shape, compression=compression, fillvalue=fillvalue, **kwargs
    )
    return array


def identity(arg):
    """@private
    """
    return arg


def noop(*args, **kwargs):
    """@private
    """
    pass


# Monkey-patch zarr v3 to support context manager behavior and
# in order to support create_dataset and require_dataset methods.
def zarr_open(*args, **kwargs):
    """@private
    """
    z = zarr.open(*args, **kwargs)

    ztype = type(z)
    if not hasattr(ztype, "__enter__"):
        ztype.__enter__ = identity
    if not hasattr(ztype, "__exit__"):
        ztype.__exit__ = noop

    # We don't need further monkey-patching if this is opening an array.
    if isinstance(z, zarr.Array):
        return z

    # The zarr group is a dataclass, so we can't do simple monkey-patching like this:
    # z.create_dataset = _create_dataset_impl
    # Hence, we have to take the slightly more complicated approach of patching the class:
    PatchedGroup = type(
        "PatchedGroup", (z.__class__,),
        {"create_dataset": _create_dataset_impl, "require_dataset": _require_dataset_impl}
    )
    object.__setattr__(z, "__class__", PatchedGroup)

    return z
