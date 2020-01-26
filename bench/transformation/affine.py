import time
import numpy as np
from functools import partial

import h5py
from skimage.data import astronaut
from scipy.ndimage import affine_transform
from elf.transformation import compute_affine_matrix, transform_subvolume_affine


def bench_affine_2d(affine_function, N=5):
    times = []
    im = astronaut()[..., 0]
    matrix = compute_affine_matrix(scale=(.5, .5), rotation=(45,))
    for _ in range(N):
        t0 = time.time()
        affine_function(im, matrix)
        t0 = time.time() - t0
        times.append(t0)
    print(np.min(times), "s")
    # print(np.mean(times), "+-", np.std(times), "s")


def bench_all_afine_2d():
    print("Benchmarking affine 2d ...")
    for order in (0, 1):
        print("Scipy - order", order, ":")
        bench_affine_2d(partial(affine_transform, order=order))
        print("Elf  -  order", order, ":")
        bench_affine_2d(partial(transform_subvolume_affine,
                                order=order, bb=np.s_[0:512, 0:512]))


def bench_affine_3d(affine_function, N=5):
    path = '/home/pape/Work/data/isbi/isbi_train_volume.h5'
    with h5py.File(path, 'r') as f:
        data = f['raw'][:]

    times = []
    matrix = compute_affine_matrix(scale=(1., 2., 1.5), rotation=(16, 32, 7))
    for _ in range(N):
        t0 = time.time()
        affine_function(data, matrix)
        t0 = time.time() - t0
        times.append(t0)
    print(np.min(times), "s")
    # print(np.mean(times), "+-", np.std(times), "s")


def bench_all_afine_3d():
    print("Benchmarking affine 3d ...")
    for order in (0, 1):
        print("Scipy - order", order, ":")
        bench_affine_3d(partial(affine_transform, order=order))
        print("Elf  -  order", order, ":")
        bench_affine_3d(partial(transform_subvolume_affine,
                                order=order, bb=np.s_[0:30, 0:512, 0:512]))


if __name__ == '__main__':
    bench_all_afine_2d()
    bench_all_afine_3d()
