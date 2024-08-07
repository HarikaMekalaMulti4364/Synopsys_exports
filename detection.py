# Copyright 2023-2024 Synopsys, Inc.
# This Synopsys software and all associated documentation are proprietary
# to Synopsys, Inc. and may only be used pursuant to the terms and conditions
# of a written license agreement with Synopsys, Inc.
# All other use, reproduction, modification, or distribution of the Synopsys
# software or the associated documentation is strictly prohibited.

import cv2
import numpy as np
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import nnac.accuracy.datasets.coco as coco_tools
from nnac.core.log import Logger
logger = Logger("Evaluation - Detection")


def plot_boxes(input_images: Path, input_files: [Path], results: list):
    path_to_output_images = Path().absolute() / "output_images"
    path_to_output_images.mkdir(parents=True, exist_ok=True)

    for item in input_files:
        item = Path(item)
        image_id = int(item.stem)
        image_name = item.name
        original_image_path = Path(input_images) / image_name
        image = cv2.imread(str(original_image_path))

        results_for_image_id = [result for result in results if result["image_id"] == image_id]
        for image_data in results_for_image_id:
            bbox = np.array(image_data["bbox"]).astype(int)
            basic_color = (0, int(255 * float(image_data["score"])), 0)
            top_left = (bbox[0], bbox[1])

            #   OpenCV accepts x1,y1,x2,y2 coordinates instead of x,y,w,h that stored in COCO json
            bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            image = cv2.rectangle(image, top_left, bottom_right, basic_color, 1)

            class_name = coco_tools.COCO2017_IDS_TO_LABELS[image_data["category_id"]]
            #   Draw text 5 pixels above bounding box
            cv2.putText(
                image,
                class_name,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                basic_color,
                1,
            )
        cv2.imwrite(str((path_to_output_images / image_name)), image)


def detection(predictions: dict, reference_data_path: Path) -> dict:
    logger.info("\nEvaluating model results using detection metric:")

    coco_ground_truth = COCO(reference_data_path)
    logger.info(f"Using {reference_data_path} as ground truth detection dataset")
    coco_tools.filter_out_images_from_dataset(coco_ground_truth, predictions)

    predictions = coco_tools.prepare_results_for_pycocotools(coco_ground_truth, predictions)

    #   TODO Store resized dataset for debug purpose?
    # path_to_json_results = "processed_bboxes.json"
    # with open(path_to_json_results, "w") as cocoDt_file:
    #     logger.info(f"Storing processed detection results in COCO format to {path_to_json_results}")
    #     json.dump(predictions, cocoDt_file)

    coco_predictions = coco_ground_truth.loadRes(predictions)
    coco_eval = COCOeval(coco_ground_truth, coco_predictions, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    #   TODO Plot detection boxes
    # plot_boxes(path_to_input_images, input_files, predictions)

    return {
        "mAP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
    }
