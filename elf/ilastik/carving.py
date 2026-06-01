import h5py
import numpy as np
import bioimage_cpp as bic


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
    rag = bic.graph.region_adjacency_graph(ws)
    return ws, rag


def export_object(project, name, ws=None, rag=None):
    """@private
    """
    if (ws is None) != (rag is None):
        raise ValueError("ws and rag must either both be provided or both be None.")
    if ws is None:
        ws, rag = load_watershed_and_rag(project)

    with h5py.File(project, "r") as f:
        fg = f[f"carving/objects/{name}/sv"][:].squeeze()
    node_labels = np.zeros(rag.number_of_nodes, dtype="uint32")
    node_labels[fg] = 1
    seg = bic.graph.project_node_labels_to_pixels(rag, ws, node_labels)

    return seg


def export_all_objects(project, postprocess=None):
    """@private
    """
    names = get_object_names(project)
    ws, rag = load_watershed_and_rag(project)
    segmentations = {}
    for name in names:
        seg = export_object(project, name, ws=ws, rag=rag)
        if postprocess is not None:
            seg = postprocess(seg)
        segmentations[name] = seg
    return segmentations
