import numpy as np


def run_metric_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
):
    """Visualize the metric scores over a range of thresholds.
    """
    import napari
    from magicgui import magic_factory

    iou_threshold = 0.5
    tp, fp, fn = _calculate_scores(ground_truth, prediction, iou_threshold)

    viewer = napari.Viewer()
    viewer.add_image(image)
    viewer.add_labels(ground_truth, name='Ground Truth')
    viewer.add_labels(prediction, name='Prediction')
    tp_layer = viewer.add_labels(tp, name='True Positives', color={1: 'green'})
    fp_layer = viewer.add_labels(fp, name='False Positives', color={1: 'red'})
    fn_layer = viewer.add_labels(fn, name='False Negatives', color={1: 'blue'})

    @magic_factory(
        call_button="Update IoU Threshold",
        iou_threshold={"widget_type": "FloatSlider", "min": 0.5, "max": 1.0, "step": 0.1}
    )
    def update_iou_threshold(iou_threshold=0.5):
        new_tp, new_fp, new_fn = _calculate_scores(ground_truth, prediction, iou_threshold)
        tp_layer.data = new_tp
        fp_layer.data = new_fp
        fn_layer.data = new_fn

    iou_widget = update_iou_threshold()
    viewer.window.add_dock_widget(iou_widget, name='IoU Threshold Slider')
    napari.run()


def _intersection_over_union(gt, predicton):
    intersection = np.logical_and(gt, predicton).sum()
    union = np.logical_or(gt, predicton).sum()
    if union == 0:
        return 0
    return intersection / union


def _calculate_scores(ground_truth, prediction, iou_threshold):
    gt_ids = np.unique(ground_truth)
    pred_ids = np.unique(prediction)

    ignore_index = 0
    gt_ids = gt_ids[gt_ids != ignore_index]
    pred_ids = pred_ids[pred_ids != ignore_index]

    shape = ground_truth.shape
    tp, fp, fn = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    matched_gt, matched_pred = set(), set()

    for pred_id in pred_ids:
        best_iou = 0
        best_gt_id = None
        for gt_id in gt_ids:
            if gt_id in matched_gt:
                continue

            iou = _intersection_over_union((ground_truth == gt_id), (prediction == pred_id))
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id

        if best_iou >= iou_threshold:
            tp = np.logical_or(tp, (prediction == pred_id))
            matched_gt.add(best_gt_id)
            matched_pred.add(pred_id)
        else:
            fp = np.logical_or(fp, (prediction == pred_id))

    for gt_id in gt_ids:
        if gt_id not in matched_gt:
            fn = np.logical_or(fn, (ground_truth == gt_id))

    return tp.astype(int), fp.astype(int), fn.astype(int)
