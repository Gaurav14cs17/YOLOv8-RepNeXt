"""
Inference script for Quantized YOLOv8-RepNeXt model.

Supports:
- Standard inference
- INT8 quantized inference
- Reparameterized inference

Usage:
    python qinference.py --weights weights/qmodel_best.pt --source image.jpg
    python qinference.py --weights weights/qmodel_best.pt --source video.mp4 --quantize
    python qinference.py --weights weights/qmodel_best.pt --source 0  # webcam
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import os
from pathlib import Path
import yaml
from typing import Optional, List

from qmodel.yolo import QYOLORepNeXt, QuantizedYOLO, create_qyolo_repnext
from utils.boxes import non_max_suppression, scale


class QYOLOInference:
    """Quantized YOLOv8-RepNeXt inference class."""

    def __init__(self, weights_path: str, device: str = None,
                 conf_thres: float = 0.25, iou_thres: float = 0.45,
                 img_size: int = 640, reparameterize: bool = True,
                 quantize: bool = False, backend: str = 'fbgemm'):
        """Initialize quantized inference engine.
        
        Args:
            weights_path: Path to model weights (.pt file)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            img_size: Input image size
            reparameterize: Whether to reparameterize for faster inference
            quantize: Whether to apply INT8 quantization
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        """
        # Quantization only works on CPU
        if quantize:
            self.device = 'cpu'
            torch.backends.quantized.engine = backend
        else:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.quantize = quantize
        
        # Load model
        self.model = self._load_model(weights_path, reparameterize, quantize)
        
        # Generate colors for visualization
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    def _load_model(self, weights_path: str, reparameterize: bool,
                    quantize: bool) -> nn.Module:
        """Load model from checkpoint."""
        print(f"Loading quantized model from {weights_path}...")
        
        # Add safe globals
        torch.serialization.add_safe_globals([QYOLORepNeXt, QuantizedYOLO])
        
        # Load checkpoint
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        # Extract model
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                model = ckpt['model']
            elif 'state_dict' in ckpt:
                # Create model from config and load state dict
                config = ckpt.get('config', {})
                model = create_qyolo_repnext(
                    config.get('variant', 'nano'),
                    config.get('num_classes', 80),
                    config.get('neck_type', 'fpn')
                )
                
                # Clean up state_dict from QAT-specific keys and handle prefix
                state_dict = ckpt['state_dict']
                cleaned_state_dict = {}
                QAT_SKIP_KEYS = (
                    'activation_post_process', 'weight_fake_quant', 'fake_quant',
                    'observer', 'histogram', 'min_val', 'max_val', 'zero_point',
                )
                for k, v in state_dict.items():
                    if any(qat_key in k for qat_key in QAT_SKIP_KEYS):
                        continue
                    # Strip 'model.' prefix if present
                    if k.startswith('model.'):
                        k = k[6:]
                    cleaned_state_dict[k] = v
                
                # Load with strict=False to ignore missing QAT keys
                model.load_state_dict(cleaned_state_dict, strict=False)
            else:
                model = ckpt
        else:
            model = ckpt
        
        # Convert to float and eval mode
        model = model.float().eval()
        
        # Reparameterize for faster inference
        if reparameterize and hasattr(model, 'reparameterize'):
            print("Reparameterizing RepNeXt blocks...")
            model = model.reparameterize()
        
        # Apply INT8 quantization
        if quantize:
            print("Applying INT8 quantization...")
            model = self._quantize_model(model)
        
        model.to(self.device)
        
        # Get class info
        self.num_classes = model.nc if hasattr(model, 'nc') else 80
        self.stride = model.stride if hasattr(model, 'stride') else torch.tensor([8, 16, 32])
        
        print(f"Model loaded: {self.num_classes} classes, device={self.device}")
        if quantize:
            print("Running in INT8 quantized mode")
        return model

    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to model."""
        # Dynamic quantization for linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        return quantized_model

    def preprocess(self, img: np.ndarray) -> tuple:
        """Preprocess image for inference.
        
        Args:
            img: BGR image from cv2.imread()
            
        Returns:
            tensor: Preprocessed tensor [1, 3, H, W]
            orig_shape: Original image shape (H, W)
            ratio_pad: ((ratio_h, ratio_w), (pad_h, pad_w))
        """
        orig_shape = img.shape[:2]  # (H, W)
        
        # Calculate resize ratio
        r = min(self.img_size / orig_shape[0], self.img_size / orig_shape[1])
        new_size = (int(round(orig_shape[1] * r)), int(round(orig_shape[0] * r)))
        
        # Resize
        img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        dw = (self.img_size - new_size[0]) / 2
        dh = (self.img_size - new_size[1]) / 2
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add padding
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # BGR to RGB, HWC to CHW, normalize
        img_padded = img_padded[:, :, ::-1].transpose(2, 0, 1)
        img_padded = np.ascontiguousarray(img_padded, dtype=np.float32) / 255.0
        
        # To tensor with batch dimension
        tensor = torch.from_numpy(img_padded).unsqueeze(0).to(self.device)
        
        ratio_pad = ((r, r), (dw, dh))
        return tensor, orig_shape, ratio_pad

    def postprocess(self, preds: torch.Tensor, orig_shape: tuple,
                    ratio_pad: tuple) -> np.ndarray:
        """Postprocess model predictions.
        
        Args:
            preds: Raw model predictions
            orig_shape: Original image shape (H, W)
            ratio_pad: ((ratio_h, ratio_w), (pad_h, pad_w))
            
        Returns:
            detections: Array of [x1, y1, x2, y2, conf, class_id]
        """
        # Handle list output from quantized model
        if isinstance(preds, list):
            preds = preds[0] if len(preds) == 1 else torch.cat(preds, dim=1)
        
        # Apply NMS
        detections = non_max_suppression(preds, self.conf_thres, self.iou_thres)
        
        if not detections or len(detections[0]) == 0:
            return np.array([])
        
        dets = detections[0]
        
        # Scale boxes to original image size
        dets[:, :4] = scale(dets[:, :4], orig_shape, ratio_pad[0], ratio_pad[1])
        
        return dets.cpu().numpy()

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> tuple:
        """Run inference on an image.
        
        Args:
            img: BGR image from cv2.imread()
            
        Returns:
            detections: Array of [x1, y1, x2, y2, conf, class_id]
            inference_time: Time in milliseconds
        """
        # Preprocess
        tensor, orig_shape, ratio_pad = self.preprocess(img)
        
        # Inference
        start = time.perf_counter()
        preds = self.model(tensor)
        inference_time = (time.perf_counter() - start) * 1000
        
        # Postprocess
        detections = self.postprocess(preds, orig_shape, ratio_pad)
        
        return detections, inference_time

    def draw_detections(self, img: np.ndarray, detections: np.ndarray,
                        class_names: list = None) -> np.ndarray:
        """Draw detections on image."""
        if class_names is None:
            class_names = [f'class_{i}' for i in range(self.num_classes)]
            
        img = img.copy()
        
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
            
            color = self.colors[cls_id % len(self.colors)].tolist()
            label = f'{class_names[cls_id]} {conf:.2f}'
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            
            # Draw label text
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return img

    def benchmark(self, num_runs: int = 100, warmup: int = 10) -> dict:
        """Benchmark inference speed.
        
        Args:
            num_runs: Number of inference runs
            warmup: Number of warmup runs
            
        Returns:
            dict with timing statistics
        """
        dummy_img = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Warmup
        print(f"Warming up ({warmup} runs)...")
        for _ in range(warmup):
            self.predict(dummy_img)
        
        # Benchmark
        print(f"Benchmarking ({num_runs} runs)...")
        times = []
        for i in range(num_runs):
            _, inf_time = self.predict(dummy_img)
            times.append(inf_time)
            if (i + 1) % 20 == 0:
                print(f"  Run {i+1}/{num_runs}: {inf_time:.2f}ms")
        
        times = np.array(times)
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'fps': 1000 / np.mean(times)
        }


def load_class_names(config_path: str) -> Optional[list]:
    """Load class names from config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if 'names' in config:
            if isinstance(config['names'], dict):
                return list(config['names'].values())
            return config['names']
    except Exception as e:
        print(f"Warning: Could not load class names: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description='Quantized YOLOv8-RepNeXt Inference')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID')
    parser.add_argument('--config', type=str, default='config/config.yml', help='Config file')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--no-reparam', action='store_true', help='Disable reparameterization')
    parser.add_argument('--quantize', action='store_true', help='Enable INT8 quantization')
    parser.add_argument('--backend', type=str, default='fbgemm', 
                       choices=['fbgemm', 'qnnpack'], help='Quantization backend')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark only')
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = QYOLOInference(
        weights_path=args.weights,
        device=args.device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        img_size=args.img_size,
        reparameterize=not args.no_reparam,
        quantize=args.quantize,
        backend=args.backend
    )
    
    # Run benchmark if requested
    if args.benchmark:
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        results = engine.benchmark()
        print(f"Mean inference time: {results['mean']:.2f}ms ± {results['std']:.2f}ms")
        print(f"Min/Max: {results['min']:.2f}ms / {results['max']:.2f}ms")
        print(f"FPS: {results['fps']:.1f}")
        print("=" * 50)
        return
    
    # Load class names
    class_names = load_class_names(args.config)
    
    # Create output directory
    if args.save:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Determine source type
    is_webcam = args.source.isdigit()
    source = int(args.source) if is_webcam else args.source
    
    # Open video/image source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open source: {args.source}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) if not is_webcam else 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_video = total_frames > 1 or is_webcam
    
    # Video writer
    writer = None
    if args.save and is_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = Path(args.output) / f"qresult_{Path(args.source).stem}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    frame_idx = 0
    total_time = 0
    
    mode = "INT8 Quantized" if args.quantize else "FP32"
    print(f"\nRunning {mode} inference on {args.source}...")
    print("-" * 50)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        detections, inf_time = engine.predict(frame)
        total_time += inf_time
        frame_idx += 1
        
        # Draw results
        result = engine.draw_detections(frame, detections, class_names)
        
        # Add info
        info_text = f"FPS: {1000/inf_time:.1f} | {mode} | Objects: {len(detections)}"
        cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show
        if args.show:
            cv2.imshow('QYOLOv8-RepNeXt Inference', result)
            key = cv2.waitKey(1 if is_video else 0) & 0xFF
            if key == ord('q'):
                break
        
        # Save
        if args.save:
            if is_video and writer:
                writer.write(result)
            else:
                out_path = Path(args.output) / f"qresult_{Path(args.source).name}"
                cv2.imwrite(str(out_path), result)
                print(f"Saved: {out_path}")
        
        # Print progress
        if frame_idx % 30 == 0 or not is_video:
            print(f"Frame {frame_idx}: {len(detections)} objects, {inf_time:.1f}ms")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("-" * 50)
    print(f"Processed {frame_idx} frames ({mode} mode)")
    if frame_idx > 0:
        print(f"Average inference time: {total_time / frame_idx:.1f}ms")
        print(f"Average FPS: {1000 * frame_idx / total_time:.1f}")


if __name__ == '__main__':
    main()

