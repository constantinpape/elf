import re
import multiprocessing
from concurrent import futures
from functools import partial
from tqdm import tqdm
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff

from .common import get_blocking
from ..util import sigma_to_halo


# helper function to choose a channel from filter output
def choose_channel(data, sigma, function, channel):
    """@private
    """
    return function(data, sigma)[..., channel]


def block_to_bb(block):
    """@private
    """
    return tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))


def get_halo(sigma, order, ndim, outer_scale=None):
    """@private
    """
    sigma_ = sigma if outer_scale is None else sigma + outer_scale
    halo = sigma_to_halo(sigma_, order)
    if isinstance(halo, int):
        halo = ndim * [halo]
    return halo


def apply_filter(
    data: ArrayLike,
    filter_name: str,
    sigma: float,
    outer_scale: Optional[str] = None,
    return_channel: Optional[int] = None,
    out: Optional[ArrayLike] = None,
    block_shape: Optional[Tuple[int, ...]] = None,
    n_threads: Optional[int] = None,
    mask: Optional[ArrayLike] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
) -> ArrayLike:
    """Apply filter to data in parallel.

    Args:
        data: Input data, numpy array or similar like h5py or zarr dataset
        filter_name: Name of the filter to apply.
        sigma: Sigma value for filter.
        outer_scale: Outer scale value for structure tensor.
        return_channel: Channel to select for multi-channel response.
        out: Output, by default the filter is applied inplace.
        block_shape: Shape of the blocks used for parallelisation,
            by default chunks of the input will be used, if available.
        n_threads: Number of threads, by default all are used.
        mask: Mask to exclude data from the computation.
        verbose: Verbosity flag.
        roi: Region of interest for this computation.

    Returns:
        The filter response.
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
    blocking = get_blocking(data, block_shape, roi, n_threads)

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
        if multi_channel and return_channel is None:
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

        # vigra / fastfilters are channel last, but we are channel first
        if block_response.ndim > ndim:
            inner_slice = (slice(None),) + inner_slice
            block_response = np.rollaxis(block_response, -1)
            if mask is not None:
                m = m[None]

        if mask is not None:
            block_response[m] = block_data[m]
        out[inner_slice] = block_response

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(_apply_filter, range(n_blocks)), total=n_blocks, disable=not verbose))

    return out


def _generate_filter(filter_name):
    # cast filter name to snake case for all elf names
    elf_name = re.sub(r'(?<!^)(?=[A-Z])', '_', filter_name).lower()
    doc_str =\
        """Comppute %s response block-wise and in parallel.

        Args:
            data: Input data, numpy array or similar like h5py or zarr dataset.
            sigma: Sigma value for filter.
            out: Output, by default the filter is applied inplace.
            block_shape: Shape of the blocks used for parallelisation,
                by default chunks of the input will be used, if available.
            n_threads: Number of threads, by default all are used.
            mask: Mask to exclude data from the computation.
            verbose: Verbosity flag.
            roi: Region of interest for this computation.

        Returns:
            The filter response.
        """ % elf_name

    def op(
        data: ArrayLike,
        sigma: float,
        out: Optional[ArrayLike] = None,
        block_shape: Optional[Tuple[int, ...]] = None,
        n_threads: Optional[int] = None,
        mask: Optional[ArrayLike] = None,
        verbose: bool = False,
        roi: Optional[Tuple[slice, ...]] = None,
    ) -> ArrayLike:
        return apply_filter(data, filter_name, sigma, outer_scale=None,
                            return_channel=None, out=out, block_shape=block_shape,
                            n_threads=n_threads, mask=mask, verbose=verbose, roi=roi)

    op.__doc__ = doc_str
    op.__name__ = elf_name
    globals()[elf_name] = op


def _generate_structure_tensor(filter_name):
    # cast filter name to snake case for all elf names
    elf_name = re.sub(r'(?<!^)(?=[A-Z])', '_', filter_name).lower()
    doc_str =\
        """Comppute %s response block-wise and in parallel.

        Args:
            data: Input data, numpy array or similar like h5py or zarr dataset.
            sigma: Sigma value for filter.
            outer_scale: Outer scale value.
            return_channel: Return selected channel.
            out: Output, by default the filter is applied inplace.
            block_shape: Shape of the blocks used for parallelisation,
                by default chunks of the input will be used, if available.
            n_threads: Number of threads, by default all are used.
            mask: Mask to exclude data from the computation.
            verbose: Verbosity flag.
            roi: Region of interest for this computation.

        Returns:
            The filter response.
        """ % elf_name

    def op(
        data: ArrayLike,
        sigma: float,
        outer_scale: float,
        return_channel: Optional[int] = None,
        out: Optional[ArrayLike] = None,
        block_shape: Optional[Tuple[int, ...]] = None,
        n_threads: Optional[int] = None,
        mask: Optional[ArrayLike] = None,
        verbose: bool = False,
        roi: Optional[Tuple[slice, ...]] = None,
    ) -> ArrayLike:
        return apply_filter(data, filter_name, sigma, outer_scale=outer_scale,
                            return_channel=return_channel, out=out, block_shape=block_shape,
                            n_threads=n_threads, mask=mask, verbose=verbose, roi=roi)

    op.__doc__ = doc_str
    op.__name__ = elf_name
    globals()[elf_name] = op


def _generate_hessian(filter_name):
    # cast filter name to snake case for all elf names
    elf_name = re.sub(r'(?<!^)(?=[A-Z])', '_', filter_name).lower()
    doc_str =\
        """Comppute %s response block-wise and in parallel.

        Args:
            data: Input data, numpy array or similar like h5py or zarr dataset.
            sigma: Sigma value for filter.
            return_channel: Return selected channel.
            out: Output, by default the filter is applied inplace.
            block_shape: Shape of the blocks used for parallelisation,
                by default chunks of the input will be used, if available.
            n_threads: Number of threads, by default all are used.
            mask: Mask to exclude data from the computation.
            verbose: Verbosity flag.
            roi: Region of interest for this computation.

        Returns:
            The filter response.
        """ % elf_name

    def op(
        data: ArrayLike,
        sigma: float,
        return_channel: Optional[int] = None,
        out: Optional[ArrayLike] = None,
        block_shape: Optional[Tuple[int, ...]] = None,
        n_threads: Optional[int] = None,
        mask: Optional[ArrayLike] = None,
        verbose: bool = False,
        roi: Optional[Tuple[slice, ...]] = None,
    ):
        return apply_filter(data, filter_name, sigma, return_channel=return_channel,
                            out=out, block_shape=block_shape,
                            n_threads=n_threads, mask=mask, verbose=verbose, roi=roi)

    op.__doc__ = doc_str
    op.__name__ = elf_name
    globals()[elf_name] = op


# autogenerate parallel filter implementations
_filter_names = ["gaussianSmoothing",
                 "gaussianGradientMagnitude",
                 "hessianOfGaussianEigenvalues",
                 "laplacianOfGaussian",
                 "structureTensorEigenvalues"]


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
