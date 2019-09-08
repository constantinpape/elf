import time
import numpy as np
import z5py
from elf.label_multiset import (create_multiset_from_labels,
                                downsample_multiset, merge_multisets)


def print_times(t):
    print("Mean:", np.mean(t), "+-", np.std(t), "s")
    print("Max: ", np.max(t), "s")
    print("Min: ", np.min(t), "s")


def bench_multiset(labels, n=5):
    print("bench_multiset:")
    mset = create_multiset_from_labels(labels)
    t = []
    for _ in range(n):
        t0 = time.time()
        mset[:]
        t0 = time.time() - t0
        t.append(t0)
    print_times(t)


def get_multisets(labels, chunks):
    msets = []
    pos = []

    for ii in (0, 1):
        for jj in (0, 1):
            for kk in (0, 1):
                p = (ii, jj, kk)
                pos.append(p)
                bb = tuple(slice(pp * ch, (pp + 1) * ch) for pp, ch in zip(p, chunks))
                mset = create_multiset_from_labels(labels[bb])
                msets.append(mset)

    return msets, pos


def bench_merge(labels, n=5):
    print("bench_merge:")
    shape = labels.shape
    chunks = tuple(sh // 2 for sh in shape)
    t = []
    for _ in range(n):
        msets, grid_pos = get_multisets(labels, chunks)
        t0 = time.time()
        merge_multisets(msets, grid_pos, shape, chunks)
        t0 = time.time() - t0
        t.append(t0)
    print_times(t)


def bench_downsample(labels, n=5):
    print("bench_downsample:")
    mset = create_multiset_from_labels(labels)
    scale_factor = [2, 2, 2]
    t = []
    for _ in range(n):
        t0 = time.time()
        downsample_multiset(mset, scale_factor)
        t0 = time.time() - t0
        t.append(t0)
    print_times(t)


if __name__ == '__main__':
    path = '/g/kreshuk/pape/Work/data/cremi/example/sampleA.n5'
    key = 'volumes/segmentation/multicut'
    bb = np.s_[:32, :256, :256]
    labels = z5py.File(path)[key][bb]

    # benchmark the multiset
    bench_multiset(labels)

    # benchmark merging multiset
    bench_merge(labels)

    # benchmark downsapling multiset
    bench_downsample(labels)
