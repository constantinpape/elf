import numpy as np
import matplotlib.pyplot as plt
import z5py

from skimage.data import astronaut
from scipy.ndimage import affine_transform
from elf.transformation.affine import compute_affine_matrix
from elf.transformation.affine_impl import apply_affine_for_subvolume


def make_input():
    im = astronaut()[:, :, 0]
    print(im.shape)
    f = z5py.File('data.n5')
    f.create_dataset('astronaut', data=im, chunks=(64, 64))


def check_affine():

    f = z5py.File('data.n5')
    ds = f['astronaut']

    im = ds[:]
    slice_ = np.s_[64:256, 64:256]

    matrix = compute_affine_matrix(scale=(.5, .5), rotation=(45,))
    matrix = np.linalg.inv(matrix)
    reference = affine_transform(im, matrix, order=0)[slice_]

    transformed = apply_affine_for_subvolume(ds, matrix, slice_)

    if np.allclose(reference, transformed):
        print("Results agree")
    else:
        print("Results DO NOT agree")

    fig, ax = plt.subplots(2)
    ax[0].imshow(reference, cmap='gray')
    ax[1].imshow(transformed, cmap='gray')
    plt.show()


# make_input()
check_affine()
