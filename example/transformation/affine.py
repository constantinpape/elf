import numpy as np
import h5py
from skimage.data import astronaut
from scipy.ndimage import affine_transform
from elf.transformation import compute_affine_matrix, transform_subvolume_affine


def check_affine_2d():
    """ Compare elf.transformation.affine to scipy reference implementation
    on 2d example data.
    """
    import matplotlib.pyplot as plt
    im = astronaut()[..., 0].astype('float32')

    order = 1
    slice_ = np.s_[0:64, 0:64]

    matrix = compute_affine_matrix(scale=(.5, .5), rotation=(45,))
    matrix = np.linalg.inv(matrix)

    reference = affine_transform(im, matrix, order=order, prefilter=False)[slice_]
    transformed = transform_subvolume_affine(im, matrix, slice_, order=order)
    diff = ~np.isclose(reference, transformed).reshape(reference.shape)

    if diff.sum() == 0:
        print("Results agree")
    else:
        print("Results DO NOT agree")

    fig, ax = plt.subplots(3)
    ax[0].imshow(reference, cmap='gray')
    ax[1].imshow(transformed, cmap='gray')
    ax[2].imshow(diff)
    plt.show()


def check_affine_3d():
    """ Compare elf.transformation.affine to scipy reference implementation
    on 3d example data.
    """
    import napari

    path = '/g/kreshuk/pape/Work/data/isbi/isbi_train_volume.h5'
    with h5py.File(path, 'r') as f:
        inp = f['raw'][:]

    bb = np.s_[0:5, 0:256, 0:256]
    matrix = compute_affine_matrix(scale=(2., 2., 2.), rotation=(25., 0., 0.))

    order = 1
    print("Transform scipy ...")
    reference = affine_transform(inp, matrix, order=order, prefilter=False)[bb]

    print("Transform elf ...")
    transformed = transform_subvolume_affine(inp, matrix, bb, order=order)

    not_close = ~np.isclose(reference, transformed)
    print("Not close elements:", not_close.sum(), "/", reference.size)
    not_close = not_close.reshape(reference.shape)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(reference, name='transformed-scipy')
        viewer.add_image(transformed, name='transformed-elf')
        viewer.add_labels(not_close, name='diff-pixel')


# check_affine_2d()
check_affine_3d()
