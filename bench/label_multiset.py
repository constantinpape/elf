import time
import numpy as np
import z5py
from elf.label_multiset import (LabelMultisetGrid,
                                create_multiset_from_labels,
                                create_multiset_from_multiset)


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


def get_multiset_grid(labels):
    shape = labels.shape
    chunks = tuple(sh // 2 for sh in shape)
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

    return LabelMultisetGrid(msets, pos, shape, chunks)


def bench_multiset_grid(labels, n=5):
    print("bench_multiset_grid:")
    mset = get_multiset_grid(labels)
    t = []
    for _ in range(n):
        t0 = time.time()
        mset[:]
        t0 = time.time() - t0
        t.append(t0)
    print_times(t)


def bench_create(labels, n=5):
    print("bench_create:")
    mset = create_multiset_from_labels(labels)
    scale_factor = [2, 2, 2]
    t = []
    for _ in range(n):
        t0 = time.time()
        create_multiset_from_multiset(mset, scale_factor)
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

    # benchmark the multiset grid
    bench_multiset_grid(labels)

    # benchmark creating the downscaled multiset
    bench_create(labels)
