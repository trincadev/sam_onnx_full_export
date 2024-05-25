from matplotlib import pyplot as plt
import onnxruntime as ort
import numpy as np
from PIL import Image
from copy import deepcopy


def preprocess_image(img_input: Image.Image, size_input: int = 1024):
    # global img, img_original_size, orig_width, orig_height, resized_width, resized_height
    # LOAD IMAGE

    print("ok:", img_input.size)
    # 1. PREPROCESS IMAGE FOR ENCODER
    # Resize image preserving aspect ratio using 1024 as a long side
    orig_width, orig_height = img_input.size

    resized_height = size_input
    resized_width = int(size_input / orig_height * orig_width)

    if orig_width > orig_height:
        resized_width = size_input
        resized_height = int(size_input / orig_width * orig_height)

    img_input = img_input.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    return img_input, img_input.size


def padding_tensor(img_input: Image.Image, size_input: int = 1024):
    # Prepare input tensor from image
    input_tensor = np.array(img_input)
    resized_width, resized_height = img_input.size

    # Normalize input tensor numbers
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([[58.395, 57.12, 57.375]])
    input_tensor = (input_tensor - mean) / std

    # Transpose input tensor to shape (Batch,Channels,Height,Width
    input_tensor = input_tensor.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

    # Make image square 1024x1024 by padding short side by zeros
    input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, 0), (0, size_input - resized_width)))
    if resized_height < resized_width:
        input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, size_input - resized_height), (0, 0)))

    return input_tensor


def get_mask(prompt: list[dict], dict_shape_img_input: dict, embeddings):
    input_points, input_labels = get_input_points(prompt)

    # Add a batch index, concatenate a padding point, and transform.
    onnx_coord = np.concatenate(
        [input_points, np.array([[0.0, 0.0]])], axis=0
    )[None, :, :]
    onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[
                 None, :
                 ].astype("float32")

    coords = deepcopy(onnx_coord).astype(float)

    orig_width, orig_height = dict_shape_img_input["original_size"]
    resized_width, resized_height = dict_shape_img_input["resized_size"]

    coords[..., 0] = coords[..., 0] * (resized_width / orig_width)
    coords[..., 1] = coords[..., 1] * (resized_height / orig_height)

    onnx_coord = coords.astype("float32")
    print("ok", onnx_coord)
    # RUN DECODER TO GET MASK
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    decoder = ort.InferenceSession("mobile_sam.decoder.onnx")
    masks, res2, res3 = decoder.run(None, {
        "image_embeddings": embeddings,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array([orig_height, orig_width], dtype=np.float32)
    })
    return masks, res2, res3


def get_input_points(prompt: list[dict]):
    """Get input points"""
    points = []
    labels = []
    for mark in prompt:
        if mark["type"] == "point":
            points.append(mark["data"])
            labels.append(mark["label"])
        elif mark["type"] == "rectangle":
            points.append([mark["data"][0], mark["data"][1]])  # top left
            points.append(
                [mark["data"][2], mark["data"][3]]
            )  # bottom right
            labels.append(2)
            labels.append(3)
    points, labels = np.array(points), np.array(labels)
    return points, labels
