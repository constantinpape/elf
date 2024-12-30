import numpy as np

from skimage.segmentation import relabel_sequential
from elf.evaluation.matching import label_overlap, intersection_over_union

try:
    import napari
    from magicgui import magic_factory
except ImportError:
    napari = None
    magic_factory = None


def _compute_matches(prediction, ground_truth, overlap_matrix, iou_threshold):
    matches = overlap_matrix > iou_threshold

    # Get the TP and FP ids, by checking which rows have / don't have a match.
    pred_matches = np.any(matches, axis=1)
    tp_ids = np.where(pred_matches)[0]
    if 0 in tp_ids:
        tp_ids = tp_ids[1:]
    fp_ids = np.where(~pred_matches)[0]
    if 0 in fp_ids:
        fp_ids = fp_ids[1:]

    # Get the FN ids by checking which columns don't have a match.
    gt_matches = np.any(matches, axis=0)
    fn_ids = np.where(~gt_matches)[0]
    if 0 in fn_ids:
        fn_ids = fn_ids[1:]

    # Compute masks based on the ids.
    tp = np.isin(prediction, tp_ids)
    fp = np.isin(prediction, fp_ids)
    fn = np.isin(ground_truth, fn_ids)

    return tp, fp, fn


def run_metric_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
):
    """Visualize the metric scores over a range of thresholds.

    Args:
        image: The input image.
        prediction: The predictions generated over the input image.
        ground_truth: The true labels for the input image.
    """
    assert napari is not None and magic_factory is not None, "Requires napari"

    ground_truth = relabel_sequential(ground_truth)[0]
    prediction = relabel_sequential(prediction)[0]

    # Compute the overlaps for objects in the prediction and ground-truth.
    overlap_matrix = intersection_over_union(label_overlap(prediction, ground_truth, ignore_label=None)[0])

    # Compute the initial TPs, FPs and FNs based on an IOU threshold of 0.5.
    iou_threshold = 0.5
    tp, fp, fn = _compute_matches(prediction, ground_truth, overlap_matrix, iou_threshold)

    viewer = napari.Viewer()
    viewer.add_image(image)
    viewer.add_labels(ground_truth, name="Ground Truth")
    viewer.add_labels(prediction, name="Prediction")

    # The keyword changed from color->colormap with napari 0.5
    try:
        tp_layer = viewer.add_labels(tp, name="True Positives", color={1: "green"})
        fp_layer = viewer.add_labels(fp, name="False Positives", color={1: "red"})
        fn_layer = viewer.add_labels(fn, name="False Negatives", color={1: "blue"})
    except TypeError:
        tp_layer = viewer.add_labels(tp, name="True Positives", colormap={1: "green"})
        fp_layer = viewer.add_labels(fp, name="False Positives", colormap={1: "red"})
        fn_layer = viewer.add_labels(fn, name="False Negatives", colormap={1: "blue"})

    @magic_factory(
        call_button="Update IoU Threshold",
        iou_threshold={"widget_type": "FloatSlider", "min": 0.1, "max": 1.0, "step": 0.05}
    )
    def update_iou_threshold(iou_threshold: float = 0.5):
        new_tp, new_fp, new_fn = _compute_matches(prediction, ground_truth, overlap_matrix, iou_threshold)
        tp_layer.data = new_tp
        fp_layer.data = new_fp
        fn_layer.data = new_fn

    iou_widget = update_iou_threshold()
    viewer.window.add_dock_widget(iou_widget, name="IoU Threshold Slider")
    napari.run()
