"""Task-Aligned Assigner for dynamic label assignment.

Assigns ground truth boxes to anchor points based on alignment metric
that combines classification score and IoU.
"""

import torch
from torch.nn.functional import one_hot
from typing import Tuple

from .iou import IoUCalculator


class TaskAlignedAssigner:
    """Task-Aligned Assigner for YOLO.

    Dynamically assigns ground truth to predictions based on
    alignment metric = score^alpha * IoU^beta

    Reference: TOOD paper - https://arxiv.org/abs/2108.07755
    """

    def __init__(self, top_k: int = 10, alpha: float = 0.5,
                 beta: float = 6.0, eps: float = 1e-9):
        """Initialize assigner.

        Args:
            top_k: Number of top candidates to consider per GT
            alpha: Classification score weight
            beta: IoU weight
            eps: Small value for numerical stability
        """
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.iou_calculator = IoUCalculator()

    @torch.no_grad()
    def assign(self, pred_scores: torch.Tensor, pred_bboxes: torch.Tensor,
               gt_labels: torch.Tensor, gt_bboxes: torch.Tensor,
               gt_mask: torch.Tensor, anchors: torch.Tensor,
               num_classes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign ground truth to predictions.

        Args:
            pred_scores: Predicted class scores (B, num_anchors, num_classes)
            pred_bboxes: Predicted boxes (B, num_anchors, 4)
            gt_labels: Ground truth labels (B, max_gt, 1)
            gt_bboxes: Ground truth boxes (B, max_gt, 4)
            gt_mask: Valid GT mask (B, max_gt, 1)
            anchors: Anchor points (num_anchors, 2)
            num_classes: Number of classes

        Returns:
            Tuple of (target_bboxes, target_scores, foreground_mask)
        """
        batch_size = pred_scores.size(0)
        num_max_boxes = gt_bboxes.size(1)
        device = gt_bboxes.device

        # Handle empty GT case
        if num_max_boxes == 0:
            return self._get_empty_targets(pred_scores, pred_bboxes, device)

        # Compute alignment metric
        overlaps = self._compute_overlaps(gt_bboxes, pred_bboxes)
        align_metric = self._compute_alignment(
            pred_scores, gt_labels, overlaps, batch_size, num_max_boxes
        )

        # Check which anchors are inside GT boxes
        mask_in_gts = self._get_in_gt_mask(gt_bboxes, anchors, batch_size)

        # Combine metrics with GT mask
        metrics = align_metric * mask_in_gts

        # Select top-k candidates
        mask_pos = self._select_topk_candidates(
            metrics, gt_mask, overlaps, batch_size, num_max_boxes
        )

        # Get foreground mask
        fg_mask = mask_pos.sum(-2)

        # Handle multiple GT assignment conflicts
        if fg_mask.max() > 1:
            mask_pos, fg_mask = self._resolve_conflicts(
                mask_pos, fg_mask, overlaps, num_max_boxes
            )

        # Get assigned targets
        target_bboxes, target_scores = self._get_targets(
            mask_pos, fg_mask, gt_labels, gt_bboxes,
            align_metric, overlaps, batch_size, num_max_boxes, num_classes
        )

        return target_bboxes, target_scores, fg_mask.bool()

    def _get_empty_targets(self, pred_scores, pred_bboxes, device):
        """Return empty targets when no GT boxes."""
        return (
            torch.zeros_like(pred_bboxes).to(device),
            torch.zeros_like(pred_scores).to(device),
            torch.zeros_like(pred_scores[..., 0]).bool().to(device)
        )

    def _compute_overlaps(self, gt_bboxes, pred_bboxes):
        """Compute IoU between GT and predictions."""
        overlaps = self.iou_calculator.compute_ciou(
            gt_bboxes.unsqueeze(2),
            pred_bboxes.unsqueeze(1)
        )
        return overlaps.squeeze(3).clamp(0)

    def _compute_alignment(self, pred_scores, gt_labels, overlaps,
                           batch_size, num_max_boxes):
        """Compute alignment metric."""
        # Create index tensor for gathering scores
        i = torch.zeros([2, batch_size, num_max_boxes], dtype=torch.long,
                        device=pred_scores.device)
        i[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_max_boxes)
        i[1] = gt_labels.long().squeeze(-1)

        # Alignment = score^alpha * IoU^beta
        scores_per_gt = pred_scores[i[0], :, i[1]]
        return scores_per_gt.pow(self.alpha) * overlaps.pow(self.beta)

    def _get_in_gt_mask(self, gt_bboxes, anchors, batch_size):
        """Check which anchors fall inside GT boxes."""
        n_boxes = gt_bboxes.size(1)

        # Split GT boxes into left-top and right-bottom
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)

        # Compute deltas from anchors to box edges
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)

        # Anchor is inside if all deltas are positive
        mask_in_gts = bbox_deltas.view(batch_size, n_boxes, anchors.shape[0], -1)
        return mask_in_gts.amin(3).gt_(1e-9)

    def _select_topk_candidates(self, metrics, gt_mask, overlaps,
                                batch_size, num_max_boxes):
        """Select top-k candidates for each GT."""
        top_k_mask = gt_mask.repeat([1, 1, self.top_k]).bool()
        num_anchors = metrics.shape[-1]

        # Get top-k metrics and indices
        top_k_metrics, top_k_indices = torch.topk(
            metrics, self.top_k, dim=-1, largest=True
        )

        # Mask invalid entries
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0)

        # Convert to one-hot and sum
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2)

        # Remove anchors assigned to multiple GTs
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k = is_in_top_k.to(metrics.dtype)

        return mask_top_k * gt_mask

    def _resolve_conflicts(self, mask_pos, fg_mask, overlaps, num_max_boxes):
        """Resolve conflicts when anchor assigned to multiple GTs."""
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, num_max_boxes, 1])

        # Assign to GT with highest IoU
        max_overlaps_idx = overlaps.argmax(1)
        is_max_overlaps = one_hot(max_overlaps_idx, num_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)

        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(-2)

        return mask_pos, fg_mask

    def _get_targets(self, mask_pos, fg_mask, gt_labels, gt_bboxes,
                     align_metric, overlaps, batch_size, num_max_boxes, num_classes):
        """Extract assigned targets."""
        device = gt_labels.device

        # Get target GT index for each anchor
        target_gt_idx = mask_pos.argmax(-2)
        batch_index = torch.arange(end=batch_size, dtype=torch.int64,
                                   device=device)[..., None]
        target_gt_idx = target_gt_idx + batch_index * num_max_boxes

        # Get target labels and boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # Create one-hot target scores
        target_labels.clamp_(0)
        target_scores = one_hot(target_labels, num_classes)

        # Zero out non-foreground
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # Normalize by alignment metric
        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps /
                            (pos_align_metrics + self.eps)).amax(-2)
        norm_align_metric = norm_align_metric.unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores
