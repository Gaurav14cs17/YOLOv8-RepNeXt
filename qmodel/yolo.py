"""Quantized RepNeXt YOLO model.

Complete YOLOv8 with RepNeXt architecture and quantization support.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional

from .backbone import QRepNeXtBackbone, QRepNeXtBackboneCSP
from .neck import QRepNeXtNeck, QRepNeXtNeckLite, QRepNeXtBiFPN
from .head import QRepNeXtHead
from .pruning import PrunableMixin, PruningConfig


class QYOLORepNeXt(nn.Module, PrunableMixin):
    """Quantized YOLOv8 with RepNeXt architecture."""

    def __init__(self,
                 dims: List[int] = None,
                 depths: List[int] = None,
                 num_classes: int = 80,
                 mlp_ratio: float = 2.0,
                 neck_type: str = 'fpn',
                 deploy: bool = False,
                 pruning_cfg: Optional[Dict] = None):
        """Initialize quantized RepNeXt YOLO.

        Args:
            dims: Channel dimensions for backbone stages
            depths: Number of blocks per backbone stage
            num_classes: Number of detection classes
            mlp_ratio: MLP expansion ratio in RepNeXt blocks
            neck_type: 'fpn', 'lite', or 'bifpn'
            deploy: Inference mode (reparameterized)
            pruning_cfg: Optional pruning configuration
        """
        super().__init__()
        self.deploy = deploy
        self.pruning_cfg = PruningConfig.from_dict(pruning_cfg)

        if dims is None:
            dims = [3, 32, 64, 128, 256, 512]
        if depths is None:
            depths = [1, 2, 2]

        # Backbone
        self.backbone = QRepNeXtBackboneCSP(dims, depths, mlp_ratio, deploy)

        # Neck
        neck_dims = [dims[3], dims[4], dims[5]]
        if neck_type == 'lite':
            self.neck = QRepNeXtNeckLite(neck_dims, mlp_ratio, deploy)
        elif neck_type == 'bifpn':
            self.neck = QRepNeXtBiFPN(neck_dims, 1, mlp_ratio, deploy)
        else:
            self.neck = QRepNeXtNeck(neck_dims, 1, mlp_ratio, deploy)

        # Head
        self.head = QRepNeXtHead(num_classes, tuple(neck_dims), 16, mlp_ratio, deploy)

        # Initialize strides
        with torch.no_grad():
            dummy = torch.zeros(1, dims[0], 640, 640)
            outputs = self._forward_once(dummy)
            self.head.stride = torch.tensor([640 / x.shape[-2] for x in outputs])
            self.stride = self.head.stride
        
        # Initialize biases for better convergence
        self.head.initialize_biases()

        # Copy attributes for compatibility
        self.nc = num_classes
        self.no = self.head.no

    def _forward_once(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Single forward pass for initialization."""
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head([p3, p4, p5])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass."""
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head([p3, p4, p5])

    def reparameterize(self) -> 'QYOLORepNeXt':
        """Reparameterize all RepNeXt blocks for inference."""
        self.backbone.reparameterize()
        self.neck.reparameterize()
        self.head.reparameterize()
        self.deploy = True
        return self

    def fuse(self) -> 'QYOLORepNeXt':
        """Alias for reparameterize for compatibility."""
        return self.reparameterize()

    def prune_model(self, amount: float = 0.2) -> None:
        """Apply global pruning."""
        if self.pruning_cfg and self.pruning_cfg.global_prune:
            self.apply_global_pruning(self, amount)


class QuantizedYOLO(nn.Module):
    """Wrapper for quantization-aware training."""

    def __init__(self, model: nn.Module):
        """Initialize quantized wrapper."""
        super().__init__()
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Copy attributes from wrapped model for compatibility
        self.nc = model.nc
        self.no = getattr(model, 'no', model.head.no)
        self.stride = model.stride
        self.head = model.head  # Expose head for loss computation

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Quantized forward pass."""
        x = self.quant(x)
        outputs = self.model(x)
        if isinstance(outputs, list):
            return [self.dequant(out) for out in outputs]
        return self.dequant(outputs)

    def reparameterize(self) -> 'QuantizedYOLO':
        """Reparameterize the wrapped model."""
        self.model.reparameterize()
        return self


# Model configurations
VARIANTS = {
    'nano': {'dims': [3, 16, 32, 64, 128, 256], 'depths': [1, 2, 2]},
    'tiny': {'dims': [3, 24, 48, 96, 192, 384], 'depths': [1, 2, 2]},
    'small': {'dims': [3, 32, 64, 128, 256, 512], 'depths': [1, 2, 2]},
    'medium': {'dims': [3, 48, 96, 192, 384, 576], 'depths': [2, 4, 4]},
    'large': {'dims': [3, 64, 128, 256, 512, 512], 'depths': [3, 6, 6]},
    'xlarge': {'dims': [3, 80, 160, 320, 640, 640], 'depths': [3, 6, 6]},
}


def create_qyolo_repnext(variant_name: str,
                          num_classes: int = 80,
                          neck_type: str = 'fpn',
                          pruning_cfg: Optional[Dict] = None) -> QYOLORepNeXt:
    """Factory function for creating quantized RepNeXt YOLO variants.

    Args:
        variant_name: Model size ('nano', 'tiny', 'small', 'medium', 'large', 'xlarge')
        num_classes: Number of detection classes
        neck_type: 'fpn', 'lite', or 'bifpn'
        pruning_cfg: Optional pruning configuration

    Returns:
        QYOLORepNeXt model
    """
    config = VARIANTS.get(variant_name.lower())
    if not config:
        raise ValueError(f"Unknown variant: {variant_name}. Choose from {list(VARIANTS.keys())}")

    model = QYOLORepNeXt(
        dims=config['dims'],
        depths=config['depths'],
        num_classes=num_classes,
        neck_type=neck_type,
        pruning_cfg=pruning_cfg
    )

    if pruning_cfg and pruning_cfg.get('prune_on_init', False):
        model.prune_model(pruning_cfg.get('amount', 0.2))

    return model


# Convenience factory functions
def qyolo_repnext_n(num_classes: int = 80, **kwargs) -> QYOLORepNeXt:
    """Create nano variant."""
    return create_qyolo_repnext('nano', num_classes, **kwargs)


def qyolo_repnext_t(num_classes: int = 80, **kwargs) -> QYOLORepNeXt:
    """Create tiny variant."""
    return create_qyolo_repnext('tiny', num_classes, **kwargs)


def qyolo_repnext_s(num_classes: int = 80, **kwargs) -> QYOLORepNeXt:
    """Create small variant."""
    return create_qyolo_repnext('small', num_classes, **kwargs)


def qyolo_repnext_m(num_classes: int = 80, **kwargs) -> QYOLORepNeXt:
    """Create medium variant."""
    return create_qyolo_repnext('medium', num_classes, **kwargs)


def qyolo_repnext_l(num_classes: int = 80, **kwargs) -> QYOLORepNeXt:
    """Create large variant."""
    return create_qyolo_repnext('large', num_classes, **kwargs)


def qyolo_repnext_x(num_classes: int = 80, **kwargs) -> QYOLORepNeXt:
    """Create xlarge variant."""
    return create_qyolo_repnext('xlarge', num_classes, **kwargs)


# Convenience for lite and bifpn variants
def qyolo_repnext_n_lite(num_classes: int = 80, **kwargs) -> QYOLORepNeXt:
    return create_qyolo_repnext('nano', num_classes, neck_type='lite', **kwargs)


def qyolo_repnext_s_lite(num_classes: int = 80, **kwargs) -> QYOLORepNeXt:
    return create_qyolo_repnext('small', num_classes, neck_type='lite', **kwargs)


def qyolo_repnext_s_bifpn(num_classes: int = 80, **kwargs) -> QYOLORepNeXt:
    return create_qyolo_repnext('small', num_classes, neck_type='bifpn', **kwargs)
