"""Bounding box utilities for object detection."""

import time
import torch
import torchvision


def scale(coords: torch.Tensor, shape1: tuple, gain: tuple,
          pad: tuple) -> torch.Tensor:
    """Scale coordinates from padded image to original image.

    Args:
        coords: Bounding box coordinates (N, 4)
        shape1: Original image shape (h, w)
        gain: Scale gain (gain_x, gain_y)
        pad: Padding (pad_x, pad_y)

    Returns:
        Scaled coordinates
    """
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, [0, 2]] /= gain[0]
    coords[:, [1, 3]] /= gain[1]
    coords[:, 0].clamp_(0, shape1[1])  # x1
    coords[:, 1].clamp_(0, shape1[0])  # y1
    coords[:, 2].clamp_(0, shape1[1])  # x2
    coords[:, 3].clamp_(0, shape1[0])  # y2
    return coords


def make_anchors(x: list, strides: torch.Tensor,
                 offset: float = 0.5) -> tuple:
    """Generate anchors from feature maps.

    Args:
        x: List of feature map tensors
        strides: Stride for each feature level
        offset: Anchor offset (default: 0.5)

    Returns:
        Tuple of (anchor_points, stride_tensor)
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride,
                                        dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        box1: First set of boxes (N, 4) in xyxy format
        box2: Second set of boxes (M, 4) in xyxy format

    Returns:
        IoU matrix (N, M)
    """
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    box1 = box1.T
    box2 = box2.T
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1[:, None] + area2 - intersection)


def wh2xy(x: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        x: Boxes in center format

    Returns:
        Boxes in corner format
    """
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction: torch.Tensor,
                        conf_threshold: float = 0.25,
                        iou_threshold: float = 0.45) -> list:
    """Perform Non-Maximum Suppression on detection predictions.

    Args:
        prediction: Raw model output (batch, num_classes + 4, anchors)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS

    Returns:
        List of detections per image, each (N, 6) [x1, y1, x2, y2, conf, cls]
    """
    nc = prediction.shape[1] - 4  # number of classes
    xc = prediction[:, 4:4 + nc].amax(1) > conf_threshold

    # Settings
    max_wh = 7680  # max box width and height (pixels)
    max_det = 300  # max boxes to keep after NMS
    max_nms = 30000  # max boxes into torchvision.ops.nms()

    start = time.time()
    outputs = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for index, x in enumerate(prediction):
        x = x.transpose(0, -1)[xc[index]]

        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)

        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]

        if not x.shape[0]:
            continue

        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_threshold)
        i = i[:max_det]
        outputs[index] = x[i]

        if (time.time() - start) > 0.5 + 0.05 * prediction.shape[0]:
            print(f'WARNING: NMS time limit exceeded')
            break

    return outputs


class BoxUtils:
    """Static utility class for box operations."""

    @staticmethod
    def scale(coords, shape1, gain, pad):
        return scale(coords, shape1, gain, pad)

    @staticmethod
    def make_anchors(x, strides, offset=0.5):
        return make_anchors(x, strides, offset)

    @staticmethod
    def box_iou(box1, box2):
        return box_iou(box1, box2)

    @staticmethod
    def wh2xy(x):
        return wh2xy(x)

    @staticmethod
    def nms(prediction, conf_threshold=0.25, iou_threshold=0.45):
        return non_max_suppression(prediction, conf_threshold, iou_threshold)

