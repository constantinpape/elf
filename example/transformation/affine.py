import numpy as np
import z5py
import h5py

from skimage.data import astronaut
from scipy.ndimage import affine_transform
from elf.transformation import compute_affine_matrix, transform_subvolume_with_affine


def make_input_2d():
    im = astronaut()[:, :, 0]
    print(im.shape)
    f = z5py.File('data.n5')
    f.create_dataset('astronaut', data=im, chunks=(64, 64))


def check_affine_2d():
    import matplotlib.pyplot as plt

    f = z5py.File('data.n5')
    ds = f['astronaut']

    im = ds[:]
    slice_ = np.s_[64:256, 64:256]

    matrix = compute_affine_matrix(scale=(.5, .5), rotation=(45,))
    matrix = np.linalg.inv(matrix)
    reference = affine_transform(im, matrix, order=0)[slice_]

    transformed = transform_subvolume_with_affine(ds, matrix, slice_)

    if np.allclose(reference, transformed):
        print("Results agree")
    else:
        print("Results DO NOT agree")

    fig, ax = plt.subplots(2)
    ax[0].imshow(reference, cmap='gray')
    ax[1].imshow(transformed, cmap='gray')
    plt.show()


def check_affine_3d():
    import napari

    path = '/g/kreshuk/pape/Work/data/isbi/isbi_train_volume.h5'
    with h5py.File(path, 'r') as f:
        inp = f['raw'][:]

    bb = np.s_[0:5, 0:256, 0:256]
    matrix = compute_affine_matrix(scale=(2., 2., 2.), rotation=(25., 0., 0.))

    print("Transform elf ...")
    t1 = transform_subvolume_with_affine(inp, matrix, bb)
    print("Transform scipy ...")
    t2 = affine_transform(inp, matrix, order=0)[bb]

    not_close = ~np.isclose(t1, t2)
    print("Not close elements:", not_close.sum(), "/", t2.size)
    not_close = not_close.reshape(t2.shape)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(t1, name='transformed-elf')
        viewer.add_image(t2, name='transformed-scipy')
        viewer.add_labels(not_close, name='diff-pixel')


# make_input_2d()
# check_affine_2d()

check_affine_3d()
