import math
import torch
from model.blocks import ConvBlock
from utils.boxes import make_anchors

class DFL(torch.nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = x

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)

class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, num_classes=80, filters=()):
        super().__init__()
        self.dfl_channels = 16
        self.num_classes = num_classes
        
        # Add YOLOv8-style attribute names
        self.nc = num_classes
        self.outputs_per_anchor = num_classes + self.dfl_channels * 4
        self.no = self.outputs_per_anchor  # matches YOLOv8 expected API
        
        self.num_layers = len(filters)
        self.stride = torch.zeros(self.num_layers)

        c1 = max(filters[0], self.num_classes)
        c2 = max(filters[0] // 4, self.dfl_channels * 4)

        self.dfl = DFL(self.dfl_channels)

        self.cls_branch = torch.nn.ModuleList([
            torch.nn.Sequential(
                ConvBlock(f, c1, 3),
                ConvBlock(c1, c1, 3),
                torch.nn.Conv2d(c1, self.num_classes, 1)
            ) for f in filters
        ])

        self.bbox_branch = torch.nn.ModuleList([
            torch.nn.Sequential(
                ConvBlock(f, c2, 3),
                ConvBlock(c2, c2, 3),
                torch.nn.Conv2d(c2, 4 * self.dfl_channels, 1)
            ) for f in filters
        ])


    def forward(self, feats):
        for i in range(self.num_layers):
            feats[i] = torch.cat((self.bbox_branch[i](feats[i]), self.cls_branch[i](feats[i])), dim=1)
        if self.training:
            return feats

        self.anchors, self.strides = [t.transpose(0, 1) for t in make_anchors(feats, self.stride, 0.5)]

        batch = feats[0].shape[0]
        concat = torch.cat([f.view(batch, self.outputs_per_anchor, -1) for f in feats], dim=2)

        box_preds, class_preds = concat.split((self.dfl_channels * 4, self.num_classes), dim=1)
        lt, rb = torch.split(self.dfl(box_preds), 2, dim=1)
        lt = self.anchors.unsqueeze(0) - lt
        rb = self.anchors.unsqueeze(0) + rb
        boxes = torch.cat(((lt + rb) / 2, rb - lt), dim=1)
        return torch.cat((boxes * self.strides, class_preds.sigmoid()), dim=1)

    def initialize_biases(self):
        for bbox_module, cls_module, s in zip(self.bbox_branch, self.cls_branch, self.stride):
            bbox_module[-1].bias.data[:] = 1.0
            s = max(float(s), 1.0)
            cls_module[-1].bias.data[:self.num_classes] = math.log(5 / self.num_classes / (640 / s) ** 2)
