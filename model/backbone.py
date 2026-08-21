"""RepNeXt Backbone for YOLO.

Based on RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization
Paper: https://arxiv.org/abs/2406.16004

Features:
- Multi-branch reparameterization during training
- Single large-kernel depthwise conv during inference
- Efficient multi-scale feature extraction
"""

import torch
import torch.nn as nn
from typing import Tuple, List

from .repnext import (
    RepNeXtBlock,
    RepNeXtDownsample,
    RepNeXtStem,
    RepNeXtCSPBlock,
    RepNeXtSPP,
    reparameterize_model
)


class Backbone(nn.Module):
    """RepNeXt backbone with CSP-style blocks.

    Combines CSP (Cross Stage Partial) design with RepNeXt blocks
    for improved gradient flow and accuracy.

    Args:
        width: List of channel dimensions [in_ch, stem, s2, s3, s4, s5]
        depth: List of block depths [d1, d2, d3]
        mlp_ratio: MLP expansion ratio in RepNeXt blocks
        deploy: If True, use fused single-branch architecture
    """

    def __init__(self, width: List[int], depth: List[int],
                 mlp_ratio: float = 2.0, deploy: bool = False):
        super().__init__()
        self.deploy = deploy

        # Stem with strided convolutions
        self.stem = nn.Sequential(
            nn.Conv2d(width[0], width[1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(width[1]),
            nn.GELU()
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(width[1], width[2], 3, 2, 1, bias=False),
            nn.BatchNorm2d(width[2]),
            nn.GELU(),
            RepNeXtCSPBlock(width[2], width[2], depth[0], mlp_ratio, deploy)
        )

        # Stage 3 (P3)
        self.stage3 = nn.Sequential(
            nn.Conv2d(width[2], width[3], 3, 2, 1, bias=False),
            nn.BatchNorm2d(width[3]),
            nn.GELU(),
            RepNeXtCSPBlock(width[3], width[3], depth[1], mlp_ratio, deploy)
        )

        # Stage 4 (P4)
        self.stage4 = nn.Sequential(
            nn.Conv2d(width[3], width[4], 3, 2, 1, bias=False),
            nn.BatchNorm2d(width[4]),
            nn.GELU(),
            RepNeXtCSPBlock(width[4], width[4], depth[2], mlp_ratio, deploy)
        )

        # Stage 5 (P5) with SPP
        self.stage5 = nn.Sequential(
            nn.Conv2d(width[4], width[5], 3, 2, 1, bias=False),
            nn.BatchNorm2d(width[5]),
            nn.GELU(),
            RepNeXtCSPBlock(width[5], width[5], depth[0], mlp_ratio, deploy),
            RepNeXtSPP(width[5], width[5])
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass returning multi-scale features (P3, P4, P5)."""
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5

    def reparameterize(self):
        """Convert to inference mode by fusing multi-branch to single."""
        reparameterize_model(self)
        self.deploy = True
        return self


class BackbonePlain(nn.Module):
    """Plain RepNeXt backbone without CSP design.

    Uses pure RepNeXt blocks for a simpler architecture.
    """

    def __init__(self, width: List[int], depth: List[int],
                 mlp_ratio: float = 2.0, deploy: bool = False):
        super().__init__()
        self.deploy = deploy

        self.stem = RepNeXtStem(width[0], width[1])
        self.stage2 = self._make_stage(width[1], width[2], depth[0],
                                       mlp_ratio, deploy)
        self.stage3 = self._make_stage(width[2], width[3], depth[1],
                                       mlp_ratio, deploy)
        self.stage4 = self._make_stage(width[3], width[4], depth[2],
                                       mlp_ratio, deploy)
        self.stage5 = nn.Sequential(
            self._make_stage(width[4], width[5], depth[0], mlp_ratio, deploy),
            RepNeXtSPP(width[5], width[5])
        )

        self._initialize_weights()

    def _make_stage(self, in_dim: int, out_dim: int, depth: int,
                    mlp_ratio: float, deploy: bool) -> nn.Sequential:
        layers = []
        if in_dim != out_dim:
            layers.append(nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.GELU()
            ))
        else:
            layers.append(RepNeXtDownsample(in_dim, mlp_ratio, deploy))

        for _ in range(depth):
            layers.append(RepNeXtBlock(out_dim, kernel_size=7,
                                       mlp_ratio=mlp_ratio, deploy=deploy))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5

    def reparameterize(self):
        reparameterize_model(self)
        self.deploy = True
        return self

