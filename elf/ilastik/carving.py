import h5py
import numpy as np
import nifty.graph.rag as nrag


def get_object_names(project):
    """@private
    """
    with h5py.File(project, "r") as f:
        object_names = list(f["carving/objects"].keys())
    return object_names


def load_watershed_and_rag(project):
    """@private
    """
    with h5py.File(project, "r") as f:
        # load the watershed; we need to transpose due to axis-tags
        ws = f["preprocessing/graph/labels"][:].T
    n_labels = int(ws.max() + 1)
    rag = nrag.gridRag(ws, numberOfLabels=n_labels)
    return ws, rag


def export_object(project, name, ws=None, rag=None):
    """@private
    """
    # TODO check for consistency in ws and rag arguments
    if ws is None:
        ws, rag = load_watershed_and_rag(project)

    with h5py.File(project, "r") as f:
        fg = f["carving/objects/%s/sv" % name][:].squeeze()
    node_labels = np.zeros(rag.numberOfNodes, dtype="uint32")
    node_labels[fg] = 1
    seg = nrag.projectScalarNodeDataToPixels(rag, node_labels)

    return seg


# TODO
# def export_all_objects(project, postprocess=None):
#     names = get_object_names(project)
#     ws, rag = load_watershed_and_rag(project)
#     for name in names:
#         seg = export_object(project, name, ws, rag)
#         # TODO save the seg
#         if postprocess is not None:
#             seg = postprocess(seg)
