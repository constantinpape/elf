import imageio.v3 as imageio

from elf.visualisation import run_metric_visualization

# to simplify switching the folder
INPUT_FOLDER = "/home/anwai/data/livecell"
# INPUT_FOLDER = "/home/pape/Work/data/incu_cyte/livecell"


def _run_prediction(image_path):
    # NOTE: overwrite this function to use your own prediction pipeline.
    from micro_sam.automatic_segmentation import automatic_instance_segmentation
    prediction = automatic_instance_segmentation(input_path=image_path, model_type="vit_b_lm")
    return prediction


def check_on_livecell(input_path, gt_path):
    if input_path is None and gt_path is None:
        from micro_sam.evaluation.livecell import _get_livecell_paths
        image_paths, gt_paths = _get_livecell_paths(input_folder=INPUT_FOLDER)
        image_path, gt_path = image_paths[0], gt_paths[0]

    image = imageio.imread(image_path)
    ground_truth = imageio.imread(gt_path)

    prediction = _run_prediction(image_path)

    # Visualize metrics over the prediction and ground truth.
    run_metric_visualization(image, prediction, ground_truth)


def main(args):
    check_on_livecell(input_path=args.input_path, gt_path=args.gt_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default=None)
    parser.add_argument("-gt", "--gt_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
