# Copyright 2023 Synopsys, Inc.
# This Synopsys software and all associated documentation are proprietary
# to Synopsys, Inc. and may only be used pursuant to the terms and conditions
# of a written license agreement with Synopsys, Inc.
# All other use, reproduction, modification, or distribution of the Synopsys
# software or the associated documentation is strictly prohibited.

from pathlib import Path
import numpy as np
import cv2
import math 

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def load_image(img_path, img_size):
    """
    Loads an image by index, returning the image, its original dimensions, and resized dimensions.

    Returns (im, original hw, resized hw)
    """
    im = cv2.imread(str(img_path))
    h0, w0 = im.shape[:2]  # orig hw 
    r = img_size / max(h0, w0)  # ratio
    print("\n preprocess var", max(h0,w0))
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR
        im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

def GetDataLoader(dataset, max_input=None, **kwargs):
    extensions = [".jpg", ".jpeg", ".png"]
    path_to_all_files = sorted(Path(dataset).glob("*"))
    paths_to_images = [image_path for image_path in path_to_all_files if image_path.suffix.lower() in extensions]
    for image_path in paths_to_images[:max_input]:
        ari = []
        shapes = [1,1]
        req_size = 1088
        #426,640
        img, (h0, w0), (h,w) = load_image(image_path, req_size)
        print("\n (h0, w0), (h,w)", (h0, w0), (h,w))
        #725,1088
        # stride = 32
        # pad = 0.5
        # ar = (h0/w0)
        # ari.append(ar)
        # print("\n ar", ar)
        # mini, maxi = min(ari), max(ari)
        # if maxi < 1:
        #     shapes = [maxi, 1]
        # elif mini > 1:
        #     shapes = [1, 1/mini]
        # batch_shapes = np.ceil(np.array(shapes)*req_size / stride + pad).astype(int)*stride
        # print("\n batch shapes", batch_shapes)
        # 768,1120 #1088,1088
        uniform_shapes = [1088, 1088]
        batch_shapes = np.ceil(np.array(uniform_shapes))
        img, ratio, pad = letterbox(img, batch_shapes, auto=False, scaleup=False)
        print("\n ratio, pad",img.shape, ratio, pad)
        # 1, 3, 768, 1120
        print(type(img))
        # (1088, 1088, 3)
        img = preprocess(img)
        print("\n fin img", img)
        #(1, 3, 1088, 1088))
        yield img, [h0,w0]

def preprocess(image):

    # preprocessed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # preprocessed_img = cv2.resize(preprocessed_img, target_size, interpolation=cv2.INTER_LINEAR)
    preprocessed_img = image[np.newaxis, ...].astype(np.float32) / 255.
    preprocessed_img = preprocessed_img.transpose(0, 3, 1, 2)

    return preprocessed_img
 
 