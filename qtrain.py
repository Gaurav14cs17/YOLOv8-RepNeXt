# qtrain.py - Quantization-Aware Training

import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
import tqdm
import yaml
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from qmodel import QYOLORepNeXt, QuantizedYOLO, create_qyolo_repnext
from utils import (
    setup_seed, setup_multi_processes,
    AverageMeter, clip_gradients,
    ComputeLoss, compute_ap, non_max_suppression,
    scale, box_iou
)
from dataloader import Dataset

warnings.filterwarnings("ignore")


class Trainer:
    """Quantization-Aware Training for YOLOv8."""

    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.init_model()

        # Setup distributed training if needed
        self.setup_distributed()

        # Initialize data loaders
        self.init_data_loaders()

        # Initialize optimizer and scheduler
        self.init_optimizer()

        # Loss function
        self.criterion = ComputeLoss(self.model, params)

        # Metrics
        self.best_ap = 0.0
        self.writer = None
        if self.is_main_process:
            self.writer = SummaryWriter(log_dir='runs/train')

    def init_model(self):
        """Initialize model with QAT preparation using RepNeXt."""
        # Get model variant from params or default to 'nano'
        model_variant = self.params.get('variant', 'nano')
        neck_type = self.params.get('neck_type', 'fpn')
        names = self.params['names']
        num_classes = len(names.values()) if isinstance(names, dict) else len(names)
        self.model = create_qyolo_repnext(model_variant, num_classes, neck_type)

        # Load pretrained weights if available
        weight_file = f'./weights/v8_{model_variant[0]}.pth'
        if os.path.exists(weight_file):
            try:
                state = torch.load(weight_file, map_location='cpu', weights_only=False)
                pretrained = state['model'] if isinstance(state, dict) and 'model' in state else state
                self.model.load_state_dict(pretrained.float().state_dict(), strict=False)
            except (KeyError, RuntimeError) as exc:
                print(f'Warning: could not load pretrained weights from {weight_file}: {exc}')

        # Prepare for QAT
        self.model.train()
        self.model = QuantizedYOLO(self.model)
        self.model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        torch.quantization.prepare_qat(self.model, inplace=True)
        self.model.to(self.device)

    def setup_distributed(self):
        """Setup distributed training."""
        self.is_main_process = self.args.local_rank == 0

        if self.args.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                module=self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank
            )

    def init_data_loaders(self):
        """Initialize train and validation data loaders."""
        # Train dataset
        data_cfg = self.params.get('data', {})
        train_dir = data_cfg.get('train', 'dataset/images/train')
        val_dir = data_cfg.get('val', 'dataset/images/val')

        # Collect training images
        train_files = self._collect_images(train_dir)
        train_dataset = Dataset(train_files, self.args.input_size, self.params, True)
        train_sampler = None
        if self.args.distributed:
            train_sampler = data.distributed.DistributedSampler(train_dataset)

        self.train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
            collate_fn=Dataset.collate_fn
        )

        # Validation dataset
        val_files = self._collect_images(val_dir)
        self.val_loader = data.DataLoader(
            Dataset(val_files, self.args.input_size, self.params, False),
            batch_size=self.args.batch_size // 2,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=Dataset.collate_fn
        )

    def _collect_images(self, img_dir):
        """Collect image paths that have matching label files."""
        from pathlib import Path

        FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')
        root = Path(img_dir)
        label_root = Path(str(root).replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'))
        images = []
        for ext in FORMATS:
            for pattern in (f'**/*.{ext}', f'**/*.{ext.upper()}'):
                for img_path in root.glob(pattern):
                    label_path = label_root / f'{img_path.stem}.txt'
                    if label_path.exists():
                        images.append(str(img_path))
        return images

    def init_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.accumulate = max(round(64 / (self.args.batch_size * self.args.world_size)), 1)
        self.params['weight_decay'] *= self.args.batch_size * self.args.world_size * self.accumulate / 64

        # Parameter groups
        biases, bn_weights, weights = [], [], []
        for module in self.model.modules():
            if hasattr(module, 'bias') and isinstance(module.bias, torch.nn.Parameter):
                biases.append(module.bias)
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_weights.append(module.weight)
            elif hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                weights.append(module.weight)

        self.optimizer = torch.optim.SGD(
            biases, lr=self.params['lr0'],
            momentum=self.params['momentum'], nesterov=True
        )
        self.optimizer.add_param_group({'params': weights, 'weight_decay': self.params['weight_decay']})
        self.optimizer.add_param_group({'params': bn_weights})

        # Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: (1 - x / self.args.epochs) * (1.0 - self.params['lrf']) + self.params['lrf']
        )

        # Warmup
        self.num_warmup = max(round(self.params['warmup_epochs'] * len(self.train_loader)), 1000)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()

        if self.args.distributed:
            self.train_loader.sampler.set_epoch(epoch)

        # Disable mosaic for last 10 epochs
        if self.args.epochs - epoch <= 10:
            self.train_loader.dataset.mosaic = False

        pbar = enumerate(self.train_loader)
        if self.is_main_process:
            print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
            pbar = tqdm.tqdm(pbar, total=len(self.train_loader))

        avg_loss = AverageMeter()

        for i, (images, targets, _) in pbar:
            images = images.to(self.device, non_blocking=True).float() / 255.0
            targets = targets.to(self.device, non_blocking=True)

            # Warmup
            self.warmup(i + len(self.train_loader) * epoch)

            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Optimize
            if (i + len(self.train_loader) * epoch) % self.accumulate == 0:
                clip_gradients(self.model)
                self.optimizer.step()
                self.optimizer.zero_grad()

            avg_loss.update(loss.item(), images.size(0))

            if self.is_main_process:
                memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                pbar.set_description(('%10s' * 2 + '%10.3g') % (
                    f'{epoch + 1}/{self.args.epochs}', memory, avg_loss.avg
                ))

        self.lr_scheduler.step()
        return avg_loss

    def warmup(self, step):
        """Learning rate warmup."""
        if step <= self.num_warmup:
            xp = [0, self.num_warmup]
            fp = [1, 64 / (self.args.batch_size * self.args.world_size)]
            self.accumulate = max(1, np.interp(step, xp, fp).round())

            for j, param_group in enumerate(self.optimizer.param_groups):
                if j == 0:
                    fp = [self.params['warmup_bias_lr'],
                          param_group['initial_lr'] * self.lr_scheduler.get_last_lr()[0]]
                else:
                    fp = [0.0, param_group['initial_lr'] * self.lr_scheduler.get_last_lr()[0]]

                param_group['lr'] = np.interp(step, xp, fp)

                if 'momentum' in param_group:
                    fp = [self.params['warmup_momentum'], self.params['momentum']]
                    param_group['momentum'] = np.interp(step, xp, fp)

    @torch.no_grad()
    def validate(self, model=None):
        """Validate the model."""
        if model is None:
            model = self.model.module if self.args.distributed else self.model

        model.eval()
        device = next(model.parameters()).device

        iou_thresholds = torch.linspace(0.5, 0.95, 10, device=device)
        n_iou = iou_thresholds.numel()
        metrics = []

        pbar = tqdm.tqdm(self.val_loader, desc='Validating')
        for images, targets, shapes in pbar:
            images = images.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device, non_blocking=True)
            batch_size, _, height, width = images.shape

            outputs = model(images)
            targets[:, 2:] *= torch.tensor([width, height, width, height], device=device)
            # Use very low threshold to capture predictions during early training
            outputs = non_max_suppression(outputs, conf_threshold=0.0001, iou_threshold=0.65)

            for i, output in enumerate(outputs):
                labels = targets[targets[:, 0] == i, 1:]
                correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=device)

                if output.shape[0] == 0:
                    if labels.shape[0]:
                        metrics.append((correct, torch.tensor([]).to(device),
                                       torch.tensor([]).to(device), labels[:, 0]))
                    continue

                detections = output.clone()
                if shapes[i] is not None:
                    # shapes[i] = (orig_shape, ((ratio_h, ratio_w), (pad_h, pad_w)))
                    orig_shape = shapes[i][0]
                    gain = shapes[i][1][0]  # (ratio_h, ratio_w)
                    pad = shapes[i][1][1]   # (pad_h, pad_w)
                    scale(detections[:, :4], orig_shape, gain, pad)

                if labels.shape[0]:
                    tbox = labels[:, 1:5].clone()
                    tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2
                    tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2
                    tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2
                    tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2
                    if shapes[i] is not None:
                        scale(tbox, orig_shape, gain, pad)

                    correct_np = np.zeros((detections.shape[0], n_iou), dtype=bool)
                    t_tensor = torch.cat((labels[:, 0:1], tbox), dim=1)
                    iou = box_iou(t_tensor[:, 1:], detections[:, :4])
                    correct_class = t_tensor[:, 0:1] == detections[:, 5]

                    for j in range(n_iou):
                        matches = torch.where((iou >= iou_thresholds[j]) & correct_class)
                        if matches[0].shape[0]:
                            matched = torch.cat((torch.stack(matches, dim=1),
                                                iou[matches[0], matches[1]].unsqueeze(1)), dim=1).cpu().numpy()
                            if matched.shape[0] > 1:
                                matched = matched[matched[:, 2].argsort()[::-1]]
                                matched = matched[np.unique(matched[:, 1], return_index=True)[1]]
                                matched = matched[np.unique(matched[:, 0], return_index=True)[1]]
                            correct_np[matched[:, 1].astype(int), j] = True

                    correct = torch.tensor(correct_np, dtype=torch.bool, device=device)

                metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

        if metrics:
            metrics_np = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
            if len(metrics_np) and metrics_np[0].any():
                tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics_np)
            else:
                m_pre = m_rec = map50 = mean_ap = 0.0
        else:
            m_pre = m_rec = map50 = mean_ap = 0.0

        model.train()
        return m_pre, m_rec, map50, mean_ap

    def save_model(self, model, metrics, epoch):
        """Save quantized model checkpoint."""
        if not self.is_main_process:
            return

        # metrics = [precision, recall, map50, mean_ap, loss]
        mean_ap = metrics[3]
        
        # Save state_dict (avoids pickling issues with QAT qconfig)
        save_model = model.module if hasattr(model, 'module') else model
        
        names = self.params['names']
        num_classes = len(names.values()) if isinstance(names, dict) else len(names)

        # Get model config for reconstruction
        config = {
            'variant': self.params.get('variant', 'nano'),
            'num_classes': num_classes,
            'neck_type': self.params.get('neck_type', 'fpn')
        }
        
        ckpt = {
            'state_dict': save_model.state_dict(),
            'config': config,
            'epoch': epoch,
            'metrics': metrics
        }
        
        torch.save(ckpt, './weights_quant/last.pt')
        if mean_ap >= self.best_ap:
            self.best_ap = mean_ap
            torch.save(ckpt, './weights_quant/best.pt')

        # Log metrics
        # metrics = [precision, recall, map50, mean_ap, loss]
        with open('weights_quant/step.csv', 'a') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'Loss', 'Precision', 'Recall', 'mAP@50', 'mAP@50-95'])
            if epoch == 0:
                writer.writeheader()
            writer.writerow({
                'epoch': str(epoch + 1).zfill(3),
                'Loss': f'{metrics[4]:.4f}',
                'Precision': f'{metrics[0]:.4f}',
                'Recall': f'{metrics[1]:.4f}',
                'mAP@50': f'{metrics[2]:.4f}',
                'mAP@50-95': f'{metrics[3]:.4f}'
            })

    def train(self):
        """Main training loop."""
        for epoch in range(self.args.epochs):
            train_metrics = self.train_epoch(epoch)

            if self.is_main_process:
                precision, recall, map50, mean_ap = self.validate()
                
                # Display metrics table
                print(f'\n{"="*60}')
                print(f'  Epoch {epoch+1}/{self.args.epochs} | Loss: {train_metrics.avg:.4f}')
                print(f'  {"─"*56}')
                print(f'  Precision: {precision:.4f} | Recall: {recall:.4f}')
                print(f'  mAP@50: {map50:.4f} | mAP@50-95: {mean_ap:.4f}')
                print(f'{"="*60}\n')
                
                self.save_model(
                    self.model.module if self.args.distributed else self.model,
                    [precision, recall, map50, mean_ap, train_metrics.avg],
                    epoch
                )

        if self.is_main_process and self.writer:
            self.writer.close()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def setup_environment(args):
    """Setup distributed training environment."""
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = args.world_size > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0 and not os.path.exists('weights_quant'):
        os.makedirs('weights_quant')

    setup_seed()
    setup_multi_processes()


def main():
    args = parse_args()
    setup_environment(args)

    with open(os.path.join('config', 'config.yml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.train:
        trainer = Trainer(args, params)
        trainer.train()

    if args.test:
        trainer = Trainer(args, params)
        ckpt = torch.load('./weights_quant/best.pt', map_location=trainer.device, weights_only=False)
        model = trainer.model.module if args.distributed else trainer.model
        model.load_state_dict(ckpt['state_dict'])
        precision, recall, map50, mean_ap = trainer.validate(model)
        print(f'\n{"="*50}')
        print(f'  Precision: {precision:.4f} | Recall: {recall:.4f}')
        print(f'  mAP@50: {map50:.4f} | mAP@50-95: {mean_ap:.4f}')
        print(f'{"="*50}')


if __name__ == "__main__":
    main()
