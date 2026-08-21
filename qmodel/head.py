"""Quantized detection head with DFL.

Decoupled classification and box regression branches.
"""

import math
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
from typing import List, Tuple

from .repnext import QRepNeXtBlock


class DFL(nn.Module):
    """Distribution Focal Loss layer for box regression."""

    def __init__(self, ch: int = 16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class QRepNeXtHead(nn.Module):
    """Quantized RepNeXt detection head.

    Uses RepNeXt blocks in the detection branches.
    """

    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, num_classes: int = 80, dims: Tuple[int, ...] = (),
                 dfl_ch: int = 16, mlp_ratio: float = 2.0, deploy: bool = False):
        """Initialize detection head.

        Args:
            num_classes: Number of detection classes
            dims: Input channels for each scale
            dfl_ch: DFL channels for box regression
            mlp_ratio: MLP expansion ratio in RepNeXt blocks
            deploy: Inference mode flag
        """
        super().__init__()
        self.num_classes = num_classes
        self.nc = num_classes
        self.dfl_ch = dfl_ch
        self.outputs_per_anchor = num_classes + dfl_ch * 4
        self.no = self.outputs_per_anchor
        self.num_layers = len(dims)
        self.stride = torch.zeros(self.num_layers)

        self.dfl = DFL(dfl_ch)
        self.quant_cat = nnq.FloatFunctional()

        # Classification branches with RepNeXt blocks
        self.cls_branches = nn.ModuleList()
        # Box regression branches with RepNeXt blocks
        self.bbox_branches = nn.ModuleList()

        for dim in dims:
            c1 = max(dim, num_classes)
            c2 = max(dim // 4, dfl_ch * 4)

            # Classification branch
            self.cls_branches.append(nn.Sequential(
                QRepNeXtBlock(dim, kernel_size=3, mlp_ratio=mlp_ratio, deploy=deploy),
                nn.Conv2d(dim, c1, 1),
                nn.GELU(),
                nn.Conv2d(c1, num_classes, 1)
            ))

            # Box regression branch
            self.bbox_branches.append(nn.Sequential(
                QRepNeXtBlock(dim, kernel_size=3, mlp_ratio=mlp_ratio, deploy=deploy),
                nn.Conv2d(dim, c2, 1),
                nn.GELU(),
                nn.Conv2d(c2, 4 * dfl_ch, 1)
            ))

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process multi-scale features.

        Args:
            feats: List of feature maps from neck

        Returns:
            List of detection outputs or fused output for inference
        """
        outputs = []
        for i in range(self.num_layers):
            bbox_out = self.bbox_branches[i](feats[i])
            cls_out = self.cls_branches[i](feats[i])
            outputs.append(self.quant_cat.cat([bbox_out, cls_out], dim=1))

        if self.training:
            return outputs

        # Inference mode
        self.anchors, self.strides = self._make_anchors(outputs, self.stride)

        batch = outputs[0].shape[0]
        concat = torch.cat([f.view(batch, self.outputs_per_anchor, -1) for f in outputs], dim=2)

        box_preds, class_preds = concat.split((self.dfl_ch * 4, self.num_classes), dim=1)
        lt, rb = torch.split(self.dfl(box_preds), 2, dim=1)
        lt = self.anchors.unsqueeze(0) - lt
        rb = self.anchors.unsqueeze(0) + rb
        boxes = torch.cat(((lt + rb) / 2, rb - lt), dim=1)

        return torch.cat((boxes * self.strides, class_preds.sigmoid()), dim=1)

    def _make_anchors(self, feats: List[torch.Tensor],
                      strides: torch.Tensor, offset: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor points."""
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(w, device=feats[i].device) + offset
            sy = torch.arange(h, device=feats[i].device) + offset
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, device=feats[i].device))
        return torch.cat(anchor_points).transpose(0, 1), torch.cat(stride_tensor).transpose(0, 1)

    def initialize_biases(self) -> None:
        """Initialize biases for better convergence."""
        for bbox_mod, cls_mod, s in zip(self.bbox_branches, self.cls_branches, self.stride):
            bbox_mod[-1].bias.data[:] = 1.0
            s = max(float(s), 1.0)
            cls_mod[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (640 / s) ** 2
            )

    def reparameterize(self) -> None:
        """Reparameterize RepNeXt blocks for inference."""
        for branch in list(self.cls_branches) + list(self.bbox_branches):
            for m in branch.modules():
                if hasattr(m, 'reparameterize'):
                    m.reparameterize()
