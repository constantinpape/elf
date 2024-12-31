from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt


def size_histogram_from_segmentation(
    segmentation: np.ndarray,
    n_bins: int = 16,
    histogram_bins: List[int] = [1],
    bin_for_threshold: Optional[int] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    ignore_background: bool = True,
) -> int:
    """Plot size histogram for the objects in the segmentation to find a threshold for size filtering.

    Args:
        segmentation: The input segmentation.
        n_bins: The umber of bins per histogram.
        histogram_bins: The ubsequent histogram bins to plot.
        bin_for_threshold: The bin in the last histogram to use for size threshold.
            If None, all histograms will be plotted.
        min_size: Minimal size considered for the histograms.
        max_size: Maximal size considered for the histograms.
        ignore_background: Whether to ignore the background label 0.

    Returns:
        The size threshold determined from the selected bin.
    """
    seg_ids, seg_sizes = np.unique(segmentation, return_counts=True)
    if ignore_background and seg_ids[0] == 0:
        seg_sizes = seg_sizes[1:]
    return plot_size_histogram(seg_sizes, n_bins=n_bins, histogram_bins=histogram_bins,
                               bin_for_threshold=bin_for_threshold, min_size=min_size,
                               max_size=max_size)


def plot_size_histogram(
    seg_sizes,
    n_bins: int = 16,
    histogram_bins: List[int] = [1],
    bin_for_threshold: Optional[int] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
) -> int:
    """Plot histogram for the sizes to find a threshold for size filtering.

    Args:
        seg_sizes: The object sizes.
        n_bins: The umber of bins per histogram.
        histogram_bins: The ubsequent histogram bins to plot.
        bin_for_threshold: The bin in the last histogram to use for size threshold.
            If None, all histograms will be plotted.
        min_size: Minimal size considered for the histograms.
        max_size: Maximal size considered for the histograms.

    Returns:
        The size threshold determined from the selected bin.
    """
    if (min_size is not None) or (max_size is not None):
        size_mask = np.ones(seg_sizes.shape, dtype='bool')
        if min_size is not None:
            size_mask[seg_sizes < min_size] = False
        if max_size is not None:
            size_mask[seg_sizes > max_size] = False
        seg_sizes = seg_sizes[size_mask]
    seg_sizes = np.sort(seg_sizes)

    if isinstance(histogram_bins, int):
        histogram_bins = [histogram_bins]

    fig, ax = plt.subplots(1 + len(histogram_bins))

    p0 = ax[0]
    _, bins, _ = p0.hist(seg_sizes, bins=n_bins)
    p0.set_title('Full Size-histogram')

    for ii, histo_bin in enumerate(histogram_bins, 1):

        size_range = bins[histo_bin]
        seg_sizes = seg_sizes[seg_sizes < size_range]

        p = ax[ii]
        _, bins, _ = p.hist(seg_sizes, bins=n_bins)
        p.set_title('Size-histogram % up to previous bin %i' % (ii, histo_bin))

    if bin_for_threshold is None:
        plt.show()
        bin_for_threshold = input("Which bin in the second histogram should be used for the size threshold? ")
        bin_for_threshold = int(bin_for_threshold)
        assert bin_for_threshold < n_bins

    size_threshold = bins[bin_for_threshold]
    print("Bin", bin_for_threshold, "corresponds to size threshold", size_threshold)
    return size_threshold
