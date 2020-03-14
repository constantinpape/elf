from .import features as elf_feats
from .import learning as elf_learn
from .import multicut as elf_mc

FEATURE_NAMES = {'raw-edge-features', 'raw-region-features', 'boundary-edge-features'}


def _edge_training(raw, input_, labels, watersheds,
                   separate_xyz, feature_names, n_threads):
    pass


def _compute_watershed(train_input, run_2dws, n_threads):
    pass


# TODO
# TODO callbacks
def multicut_workflow(train_raw, train_input, train_labels,
                      test_input, test_raw,
                      train_watersheds=None, test_watersheds=None,
                      feature_names=FEATURE_NAMES, run_2dws=False,
                      weighting_scheme=None, multicut_solver=None,
                      n_threads=None, serialize_config=False):
    """ Run workflow for multicut segmentation based on boundary inputs.

    Based on "Multicut brings automated neurite segmentation closer to human performance":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/217205318/beier_17_multicut.pdf

    Arguments:
        train_raw [np.ndarray] -
        train_input [np.ndarray] -
        train_labels [np.ndarray] -
        test_input [np.ndarray] -
        test_raw [np.ndarray] -
        train_watersheds [np.ndarray] -
        test_watersheds [np.ndarray] -
        feature_names [listlike] -
        run_2dws [bool] -
        weighting_scheme [str] -
        multicut_solver [str] -
        n_threads [int] -
        serialize_config [bool] -
    """
    if train_raw.shape != train_labels.shape:
        raise ValueError("Shapes for train data do not match")
    if len(FEATURE_NAMES - set(feature_names)) > 0:
        raise ValueError("Invalid feature set")

    if train_watersheds is None:
        train_watersheds = _compute_watershed(train_input, run_2dws, n_threads)

    if train_watersheds.shape != train_labels.shape:
        raise ValueError("Shapes for train data do not match")

    rf = _edge_training(train_raw, train_input, train_labels,
                        train_watersheds, run_2dws, feature_names,
                        n_threads)


# ref: https://github.com/constantinpape/mu-net/blob/master/mu_net/utils/segmentation.py#L157
# TODO
def simple_multicut_workflow(input_, watersheds=None, run_2dws=False,
                             n_threads=None, serialize_config=False):
    """ Run simplified multicut segmentation workflow from affinity or boundary maps.

    Adapted from "Multicut brings automated neurite segmentation closer to human performance":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/217205318/beier_17_multicut.pdf

    Arguments:
        input_ [np.ndarray] -
        watersheds [np.ndarray] -
        run_2dws [bool] -
    """


# TODO implement:
# - lmc workflows
# - multicut_workflow_from_config(config_file)
