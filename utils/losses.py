"""Individual loss functions for YOLO training.

Contains:
- ClassificationLoss: BCE loss for class predictions
- BoxLoss: IoU-based loss for bounding box regression
- DFLoss: Distribution Focal Loss for box regression
"""

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from .iou import IoUCalculator


class ClassificationLoss(nn.Module):
    """Binary Cross-Entropy loss for classification.

    Uses BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self, reduction: str = 'none'):
        """Initialize classification loss.

        Args:
            reduction: Loss reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred_scores: torch.Tensor,
                target_scores: torch.Tensor) -> torch.Tensor:
        """Compute classification loss.

        Args:
            pred_scores: Predicted logits (B, num_anchors, num_classes)
            target_scores: Target scores (B, num_anchors, num_classes)

        Returns:
            Classification loss
        """
        return self.bce(pred_scores, target_scores)


class BoxLoss(nn.Module):
    """IoU-based loss for bounding box regression.

    Supports multiple IoU variants through IoUCalculator.
    """

    def __init__(self, iou_type: str = 'ciou'):
        """Initialize box loss.

        Args:
            iou_type: Type of IoU ('iou', 'giou', 'diou', 'ciou')
        """
        super().__init__()
        self.iou_type = iou_type
        self.iou_calculator = IoUCalculator()

    def forward(self, pred_bboxes: torch.Tensor,
                target_bboxes: torch.Tensor) -> torch.Tensor:
        """Compute box regression loss.

        Args:
            pred_bboxes: Predicted boxes (N, 4) in xyxy format
            target_bboxes: Target boxes (N, 4) in xyxy format

        Returns:
            Box loss (1 - IoU for each box)
        """
        if self.iou_type == 'ciou':
            iou = self.iou_calculator.compute_ciou(pred_bboxes, target_bboxes)
        elif self.iou_type == 'diou':
            iou = self.iou_calculator.compute_diou(pred_bboxes, target_bboxes)
        elif self.iou_type == 'giou':
            iou = self.iou_calculator.compute_giou(pred_bboxes, target_bboxes)
        else:
            iou = self.iou_calculator.compute_iou(pred_bboxes, target_bboxes)

        return iou


class DFLoss(nn.Module):
    """Distribution Focal Loss for box regression.

    Predicts a distribution over discrete positions instead of
    direct coordinate regression.

    Reference: https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self):
        """Initialize DFL loss."""
        super().__init__()

    def forward(self, pred_dist: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Compute Distribution Focal Loss.

        Args:
            pred_dist: Predicted distribution (N, num_bins)
            target: Target values (N, 4) - continuous values to be discretized

        Returns:
            DFL loss
        """
        # Get left and right target bins
        dfl_ch = pred_dist.shape[-1]
        target_left = target.long().clamp(0, dfl_ch - 2)
        target_right = (target_left + 1).clamp(max=dfl_ch - 1)

        # Compute weights (how much to weight left vs right)
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left

        # Cross entropy for left and right
        loss_left = cross_entropy(
            pred_dist, target_left.view(-1), reduction="none"
        ).view(target_left.shape)

        loss_right = cross_entropy(
            pred_dist, target_right.view(-1), reduction="none"
        ).view(target_left.shape)

        # Weighted sum
        return (loss_left * weight_left + loss_right * weight_right).mean(-1, keepdim=True)


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss for joint classification-quality prediction.

    Extends Focal Loss to predict IoU quality alongside class.

    Reference: https://arxiv.org/abs/2006.04388
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """Initialize QFL.

        Args:
            gamma: Focusing parameter
            alpha: Balancing parameter
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                quality: torch.Tensor) -> torch.Tensor:
        """Compute Quality Focal Loss.

        Args:
            pred: Predicted logits
            target: Target labels (one-hot)
            quality: IoU quality scores

        Returns:
            QFL loss
        """
        pred_sigmoid = pred.sigmoid()

        # Scale by quality
        scale_factor = pred_sigmoid
        target_scaled = target * quality.unsqueeze(-1)

        # Focal weight
        focal_weight = (target_scaled - pred_sigmoid).abs().pow(self.gamma)

        # BCE loss
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target_scaled, reduction='none'
        )

        return focal_weight * bce


class VarifocalLoss(nn.Module):
    """Varifocal Loss for dense object detection.

    Uses IoU-aware classification score as soft label.

    Reference: https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize VFL.

        Args:
            gamma: Focusing parameter for negative samples
            alpha: Weighting for positive samples
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Varifocal Loss.

        Args:
            pred: Predicted logits
            target: Target scores (IoU-weighted)

        Returns:
            VFL loss
        """
        pred_sigmoid = pred.sigmoid()

        # Separate positive and negative
        positive_mask = target > 0
        negative_mask = ~positive_mask

        # Focal weight for negatives
        focal_weight = torch.zeros_like(pred)
        focal_weight[negative_mask] = pred_sigmoid[negative_mask].pow(self.gamma)
        focal_weight[positive_mask] = self.alpha

        # BCE loss
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        return focal_weight * bce

