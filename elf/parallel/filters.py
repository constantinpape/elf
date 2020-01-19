import re
import multiprocessing
from concurrent import futures
from functools import partial
from tqdm import tqdm

import nifty.tools as nt
try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff

from .common import get_block_shape
from ..util import sigma_to_halo


# helper function to choose a channel from filter output
def choose_channel(data, sigma, function, channel):
    return function(data, sigma)[..., channel]


def block_to_bb(block):
    return tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))


def get_halo(sigma, order, ndim, outer_scale=None):
    sigma_ = sigma if outer_scale is None else sigma + outer_scale
    halo = sigma_to_halo(sigma_, order)
    if isinstance(halo, float):
        halo = ndim * [halo]
    return halo


def apply_filter(data, filter_name, sigma,
                 outer_scale=None, return_channel=None,
                 out=None, block_shape=None,
                 n_threads=None, mask=None,
                 verbose=False):
    """ Apply filter to data in parallel.

    Arguments:
        data [array_like] - input data, numpy array or similar like h5py or zarr dataset
        filter_name [str] - name of the filter to apply
        sigma [float or list[float]] - sigma value for filter
        outer_scale [float] - outer scale value for structure tensor (default: None)
        return_channel [int] - channel to select for multi-scale response (default: None)
        out [array_like] - output, by default the filter is applied inplace (default: None)
        block_shape [tuple] - shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available (default: None)
        n_threads [int] - number of threads, by default all are used (default: None)
        mask [array_like] - mask to exclude data from the computation (default: None)
        verbose [bool] - verbosity flag (default: False)
    Returns:
        array_like - filter response
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads

    # order values for halo calculation, also used to check valid filters
    order_values = {
        "gaussianSmoothing": 0,
        "gaussianGradientMagnitude": 1,
        "hessianOfGaussianEigenvalues": 2,
        "structureTensorEigenvalues": 1,
        "laplacianOfGaussian": 2,
    }
    if filter_name not in order_values:
        raise ValueError(f"{filter_name} is not a valid filter")

    if mask is not None and mask.shape != data.shape:
        raise ValueError("Invalid mask shape, got %s, expected %s (= shape of first operand)" % (str(mask.shape),
                                                                                                 str(data.shape)))

    filter_function = getattr(ff, filter_name)
    if filter_name == "structureTensorEigenvalues":
        assert outer_scale is not None, "Need outer_scale for structureTensorEigenvalues"
        filter_function = partial(filter_function, outerScale=outer_scale)

    ndim = data.ndim
    shape = data.shape
    block_shape = get_block_shape(data, block_shape)
    blocking = nt.blocking(ndim * [0], shape, block_shape)

    order = order_values[filter_name]
    halo = get_halo(sigma, order, ndim, outer_scale)
    multi_channel = filter_name in ("hessianOfGaussianEigenvalues",
                                    "structureTensorEigenvalues")

    if out is None:
        if multi_channel and return_channel is None:
            raise ValueError("Cannot apply filter in-place for multi-channel response")
        out = data
    else:
        exp_shape = data.shape
        if multi_channel and return_channel is not None:
            exp_shape = (ndim,) + exp_shape
        if out.shape != exp_shape:
            raise ValueError("Output shape %s does not match expected shape %s" % (str(out.shape),
                                                                                   str(exp_shape)))

    # get the correct output shape depending on whether we have multi-channel features
    # and whether we keep all channels for thos
    if multi_channel and return_channel is not None:
        # multi-channel output, but we only keep a single channel
        # -> output-shape = shape
        assert return_channel < ndim, f"{return_channel} must be smaller than {ndim}"
        filter_function = partial(choose_channel,
                                  function=filter_function,
                                  channel=return_channel)

    def _apply_filter(block_id):
        # get the block with halo and the slicings corresponding to
        # the block with halo, the block without halo and the
        # block without halo in loocal coordinates
        block = blocking.getBlockWithHalo(blockIndex=block_id, halo=halo)
        inner_slice = block_to_bb(block.innerBlock)
        outer_slice = block_to_bb(block.outerBlock)
        inner_local_slice = block_to_bb(block.innerBlockLocal)

        if mask is not None:
            m = mask[inner_slice]
            if m.sum() == 0:
                return

        block_data = data[outer_slice]
        block_response = filter_function(block_data, sigma)[inner_local_slice]

        if mask is not None:
            block_response[m] = block_data[m]
        out[inner_slice] = block_response

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        if verbose:
            list(tqdm(tp.map(_apply_filter, range(n_blocks)), total=n_blocks))
        else:
            tp.map(_apply_filter, range(n_blocks))

    return out


def _generate_filter(filter_name):
    # cast filter name to snake case for all elf names
    elf_name = re.sub(r'(?<!^)(?=[A-Z])', '_', filter_name).lower()
    doc_str =\
        """Comppute %s response block-wise and in parallel.

        Arguments:
            data [array_like] - input data, numpy array or similar like h5py or zarr dataset
            sigma [float or list[float]] - sigma value for filter
            out [array_like] - output, by default the filter is applied inplace (default: None)
            block_shape [tuple] - shape of the blocks used for parallelisation,
                by default chunks of the input will be used, if available (default: None)
            n_threads [int] - number of threads, by default all are used (default: None)
            mask [array_like] - mask to exclude data from the computation (default: None)
            verbose [bool] - verbosity flag (default: False)
        Returns:
            array_like - filter response
        """ % elf_name

    def op(data, sigma, out=None, block_shape=None,
           n_threads=None, mask=None, verbose=False):
        apply_filter(data, filter_name, sigma, outer_scale=None,
                     return_channel=None, out=out, block_shape=block_shape,
                     n_threads=n_threads, mask=mask, verbose=verbose)

    op.__doc__ = doc_str
    op.__name__ = elf_name
    globals()[elf_name] = op


def _generate_structure_tensor(filter_name):
    # cast filter name to snake case for all elf names
    elf_name = re.sub(r'(?<!^)(?=[A-Z])', '_', filter_name).lower()
    doc_str =\
        """Comppute %s response block-wise and in parallel.

        Arguments:
            data [array_like] - input data, numpy array or similar like h5py or zarr dataset
            sigma [float or list[float]] - sigma value for filter
            outer_scale [float] - outer scale value
            return_channel [int] - return selected channel (default: None)
            out [array_like] - output, by default the filter is applied inplace (default: None)
            block_shape [tuple] - shape of the blocks used for parallelisation,
                by default chunks of the input will be used, if available (default: None)
            n_threads [int] - number of threads, by default all are used (default: None)
            mask [array_like] - mask to exclude data from the computation (default: None)
            verbose [bool] - verbosity flag (default: False)
        Returns:
            array_like - filter response
        """ % elf_name

    def op(data, sigma, outer_scale,
           return_channel=None, out=None, block_shape=None,
           n_threads=None, mask=None, verbose=False):
        apply_filter(data, filter_name, sigma, outer_scale=outer_scale,
                     return_channel=return_channel, out=out, block_shape=block_shape,
                     n_threads=n_threads, mask=mask, verbose=verbose)

    op.__doc__ = doc_str
    op.__name__ = elf_name
    globals()[elf_name] = op


def _generate_hessian(filter_name):
    # cast filter name to snake case for all elf names
    elf_name = re.sub(r'(?<!^)(?=[A-Z])', '_', filter_name).lower()
    doc_str =\
        """Comppute %s response block-wise and in parallel.

        Arguments:
            data [array_like] - input data, numpy array or similar like h5py or zarr dataset
            sigma [float or list[float]] - sigma value for filter
            return_channel [int] - return selected channel (default: None)
            out [array_like] - output, by default the filter is applied inplace (default: None)
            block_shape [tuple] - shape of the blocks used for parallelisation,
                by default chunks of the input will be used, if available (default: None)
            n_threads [int] - number of threads, by default all are used (default: None)
            mask [array_like] - mask to exclude data from the computation (default: None)
            verbose [bool] - verbosity flag (default: False)
        Returns:
            array_like - filter response
        """ % elf_name

    def op(data, sigma, return_channel=None,
           out=None, block_shape=None,
           n_threads=None, mask=None, verbose=False):
        apply_filter(data, sigma, return_channel=return_channel,
                     out=out, block_shape=block_shape,
                     n_threads=n_threads, mask=mask, verbose=verbose)

    op.__doc__ = doc_str
    op.__name__ = elf_name
    globals()[filter_name] = op


# autogenerate parallel filter implementations
_filter_names = ["gaussianSmoothing",
                 "gaussianGradientMagnitude",
                 "hessianOfGaussianEigenvalues",
                 "structureTensorEigenvalues",
                 "laplacianOfGaussian"]


for filter_name in _filter_names:
    if filter_name == "structureTensorEigenvalues":
        _generate_structure_tensor(filter_name)
    elif filter_name == "hessianOfGaussianEigenvalues":
        _generate_hessian(filter_name)
    else:
        _generate_filter(filter_name)

del _filter_names
del _generate_filter
del _generate_hessian
del _generate_structure_tensor
