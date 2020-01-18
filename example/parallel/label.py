import napari
import numpy as np
from skimage.measure import label as label_reference
from elf.io import open_file
from elf.parallel.label import label
from elf.evaluation import rand_index


def merge_ids(seg, n_merges=50):
    uniques = np.unique(seg)
    merge_ids = np.random.choice(uniques,
                                 size=(n_merges, 2),
                                 replace=False)
    for ida, idb in merge_ids:
        seg[seg == idb] = ida
    return seg


# TODO implement in notebook
def label_isbi():
    path = '/home/pape/Work/data/isbi/isbi_train_volume.h5'
    with open_file(path, 'r') as f:
        seg = f['labels/gt_segmentation'][:]

    # merge disconnected segments, so that cc makes a difference
    seg = merge_ids(seg)

    # apply connected components
    blocks = (10, 100, 100)
    seg_cc = np.zeros_like(seg)
    label(seg, seg_cc, block_shape=blocks, verbose=True)

    seg_cc_ref = label_reference(seg)

    # check that the two segmentations agree
    ari, ri = rand_index(seg_cc_ref, seg_cc)
    assert ari == 0., str(ari)
    assert ri == 1., str(ri)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_labels(seg, name='segmentation')
        viewer.add_labels(seg_cc, name='labeled')


if __name__ == '__main__':
    label_isbi()
