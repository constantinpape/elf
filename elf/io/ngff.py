import skimage.transform
# we use zarr here because z5py does not support nested directory for the zarr format
import zarr
from . import files

AXES_NAMES = {"t", "c", "z", "y", "x"}


def _get_chunks(axes_names):
    if len(axes_names) == 2:
        return (256, 256)
    elif len(axes_names) == 3:
        return 3*(64,) if axes_names[0] == 'z' else (1, 256, 256)
    elif len(axes_names) == 4:
        return (1, 1, 256, 256) if axes_names[:2] == ('t', 'c') else (1, 64, 64, 64)
    else:
        return (1, 1, 64, 64, 64)


def _validate_axes_names(ndim, axes_names):
    assert len(axes_names) == ndim
    val_axes = tuple(axes_names)
    if ndim == 2:
        assert val_axes == ('y', 'x')
    elif ndim == 3:
        assert val_axes in [('z', 'y', 'x'), ('c', 'y', 'x'), ('t', 'y', 'x')]
    elif ndim == 4:
        assert val_axes in [('t', 'z', 'y', 'x'), ('c', 'z' 'y', 'x'), ('t', 'c', 'y', 'x')]
    else:
        assert val_axes == ('t', 'c', 'z', 'y', 'x')


# TODO downscale only in spatial dimensions
def _downscale(data, downscaler, kwargs):
    data = downscaler(data, **kwargs).astype(data.dtype)
    return data


# TODO expose dimension separator as param
def write_ome_zarr(data, path, axes_names, name, n_scales,
                   key=None, chunks=None,
                   downscaler=skimage.transform.rescale,
                   kwargs={"scale": (0.5, 0.5, 0.5), "order": 0, "preserve_range": True}):
    """Write numpy data to ome.zarr format.
    """

    assert 2 <= data.ndim <= 5
    _validate_axes_names(data.ndim, axes_names)

    chunks = _get_chunks(axes_names) if chunks is None else chunks
    store = zarr.NestedDirectoryStore(path, dimension_separator="/")

    with zarr.open(store, mode='a') as f:
        g = f if key is None else f.require_group(key)
        g.create_dataset('s0', data=data, chunks=chunks, dimension_separator="/")
        for ii in range(1, n_scales):
            data = _downscale(data, downscaler, kwargs)
            g.create_dataset(f's{ii}', data=data, chunks=chunks, dimension_separator="/")
        function_name = f'{downscaler.__module__}.{downscaler.__name__}'
        create_ngff_metadata(g, name, axes_names,
                             type_=function_name, metadata=kwargs)


def create_ngff_metadata(g, name, axes_names, type_=None, metadata=None):
    """Create ome-ngff metadata for a multiscale dataset stored in zarr format.
    """
    assert files.is_z5py(g) or files.is_zarr(g)
    assert files.is_group(g)

    # validate the individual datasets
    ndim = g[list(g.keys())[0]].ndim
    assert all(dset.ndim == ndim for dset in g.values())
    assert all(files.is_dataset(dset) for dset in g.values())
    assert len(axes_names) == ndim
    assert len(set(axes_names) - AXES_NAMES) == 0

    ms_entry = {
        "datasets": [
            {"path": name} for name in g
        ],
        "axes": axes_names,
        "name": name,
        "version": "0.3"
    }
    if type_ is not None:
        ms_entry["type"] = type_
    if metadata is not None:
        ms_entry["metadata"] = metadata

    metadata = g.attrs.get("multiscales", [])
    metadata.append(ms_entry)
    g.attrs["multiscales"] = metadata

    # write the array dimensions for compat with xarray:
    # https://xarray.pydata.org/en/stable/internals/zarr-encoding-spec.html?highlight=zarr
    for ds in g.values():
        ds.attrs["_ARRAY_DIMENSIONS"] = axes_names
