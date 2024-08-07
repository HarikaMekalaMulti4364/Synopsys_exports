# Copyright 2023 Synopsys, Inc.
# This Synopsys software and all associated documentation are proprietary
# to Synopsys, Inc. and may only be used pursuant to the terms and conditions
# of a written license agreement with Synopsys, Inc.
# All other use, reproduction, modification, or distribution of the Synopsys
# software or the associated documentation is strictly prohibited.

import numpy as np
import torchvision
import torch
from typing import List, Dict
from nnac.core.onnx_inference import onnxruntime_session_setup
from nnac.accuracy.datasets.coco import coco_ids


def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding


    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.
    """

    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = torch.from_numpy(prediction[0])
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    redundant = True
    multi_label &= nc > 1
    merge = False

    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        if merge and (1 < n < 3e3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]

    return output


def run_postprocessing_subgraph(nn_output: List):
    rt_sess = onnxruntime_session_setup(__file__.replace("postprocess.py", "postprocessing_subgraph.onnx"))
    feed_dict = {
        "/model.24/Sigmoid_output_0": np.expand_dims(nn_output[0], axis=0).astype(np.float32),
        "/model.24/Sigmoid_1_output_0": np.expand_dims(nn_output[1], axis=0).astype(np.float32),
        "/model.24/Sigmoid_2_output_0": np.expand_dims(nn_output[2], axis=0).astype(np.float32)
    }
    pred_onnx = rt_sess.run([], feed_dict)
    return pred_onnx


def postprocess(nn_output: List, input_shape: List) -> List[Dict]:

    transformed_outputs = []
    for output in nn_output:
        if output.shape[0] == 255:
            output = output
        else:
            output = output.transpose(2, 0, 1)
        c, h, w = output.shape
        reshaped = output.reshape(3, c//3, h, w)
        reshaped_per = np.transpose(reshaped, (0, 2, 3, 1))
        # print("\n reshaped_per", reshaped_per.shape)
        transformed_outputs.append(reshaped_per)

    prediction = run_postprocessing_subgraph(transformed_outputs)
    pred = non_max_suppression(
                prediction, 0.001, 0.6, labels=None, multi_label=True, agnostic=False, max_det=300
            )
    predn = pred[0].clone()
    results = list()
    scale_boxes(torch.Size([1088,1088]), predn[:, :4], input_shape)
    # print(predn)
    # exit()
    # for det in predn:
    predn[:, :4] /= 1088
    for *box, score, class_id in predn:
        xmin, ymin, xmax, ymax = box
        entry = {
            "category_id": coco_ids[int(class_id)],
            "score": float(score),
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
        }

        results.append(entry)
    return results
