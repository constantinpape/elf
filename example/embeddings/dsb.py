import os
import numpy as np
import h5py
import imageio

import napari
import elf.segmentation.embeddings as embed


def segment_from_embeddings():
    in_folder = '/home/pape/Work/data/data_science_bowl/dsb2018/test/images'
    input_images = os.listdir(in_folder)

    test_image = input_images[0]
    test_name = os.path.splitext(test_image)[0]
    im = np.asarray(imageio.imread(os.path.join(in_folder, test_image)))

    pred_file = './predictions.h5'
    with h5py.File(pred_file, 'r') as f:
        pred = f[test_name][:]

    pca = embed.embedding_pca(pred).transpose((1, 2, 0))
    seg = embed.embedding_slic(pred)

    with napari.gui_qt():
        viewer = napari.view_image(pca, rgb=True, name='pca')
        viewer.add_image(im, name='image')
        viewer.add_labels(seg, name='segmentation')


if __name__ == '__main__':
    segment_from_embeddings()
