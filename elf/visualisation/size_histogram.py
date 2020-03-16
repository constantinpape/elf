import numpy as np
import matplotlib.pyplot as plt


# TODO enable pre size-filtering
def plot_size_histogram(segmentation, n_bins=16, histogram_bins=1, bin_for_threshold=None):
    seg_sizes = np.unique(segmentation, return_counts=True)[1]
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
