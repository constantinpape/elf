import numpy as np
import h5py

import napari
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
import elf.segmentation.embeddings as embed


def segment_mc(pred, seg, delta):
    rag = feats.compute_rag(seg)
    edge_probs = embed.edge_probabilities_from_embeddings(pred, seg, rag, delta)
    edge_sizes = feats.compute_boundary_mean_and_length(rag, pred[0])[:, 1]
    costs = mc.transform_probabilities_to_costs(edge_probs, edge_sizes=edge_sizes)
    mc_seg = mc.multicut_kernighan_lin(rag, costs)
    mc_seg = feats.project_node_labels_to_pixels(rag, mc_seg)
    return mc_seg


def segment_from_embeddings():
    """ Example of embedding based segmentation workflow
    on embeddings trained for the ISBI2012 segmentation challenge.

    This is still WIP.
    """

    bb = np.s_[1, :, :]
    bb_emb = (slice(None),) + bb

    p = './prediction.h5'
    print("Load all ...")
    with h5py.File(p, 'r') as f:
        ds = f['data']
        pred = ds[bb_emb]

    print("Run pca ...")
    pca = embed.embedding_pca(pred).transpose((1, 2, 0))

    print("Run slic ...")
    seg = embed.embedding_slic(pred)

    print("Run mc ...")
    delta = 2.
    mc = segment_mc(pred, seg, delta)

    print("Compute affinities ...")
    offsets = [[-3, 0], [0, -3]]
    affs = embed.embeddings_to_affinities(pred, offsets, delta=delta,
                                          invert=True)

    p = '/home/pape/Work/data/isbi/isbi_test_volume.h5'
    with h5py.File(p, 'r') as f:
        raw = f['raw'][bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='image')
        viewer.add_image(pred, name='emebddings')
        viewer.add_image(pca, rgb=True, name='pca')
        viewer.add_image(affs, name='affs')
        viewer.add_labels(seg, name='slic')
        viewer.add_labels(mc, name='multicut')


if __name__ == '__main__':
    segment_from_embeddings()
