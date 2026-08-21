"""
Inference script for YOLOv8-RepNeXt model.

Usage:
    python inference.py --weights weights/best.pt --source image.jpg
    python inference.py --weights weights/best.pt --source video.mp4
    python inference.py --weights weights/best.pt --source 0  # webcam
"""

import argparse
import cv2
import numpy as np
import torch
import time
import os
from pathlib import Path
import yaml

from model.yolo import YOLO
from utils.boxes import non_max_suppression, scale


class YOLOInference:
    """YOLOv8-RepNeXt inference class."""

    def __init__(self, weights_path: str, device: str = None, 
                 conf_thres: float = 0.25, iou_thres: float = 0.45,
                 img_size: int = 640, reparameterize: bool = True):
        """Initialize inference engine.
        
        Args:
            weights_path: Path to model weights (.pt file)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            img_size: Input image size
            reparameterize: Whether to reparameterize for faster inference
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        
        # Load model
        self.model = self._load_model(weights_path, reparameterize)
        
        # Generate colors for visualization
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
        
    def _load_model(self, weights_path: str, reparameterize: bool) -> YOLO:
        """Load model from checkpoint."""
        print(f"Loading model from {weights_path}...")
        
        # Add safe globals for loading
        torch.serialization.add_safe_globals([YOLO])
        
        # Load checkpoint
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Extract model
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model = ckpt['model']
        else:
            model = ckpt
        
        # Ensure model is on CPU and in float32 before any operations
        model = model.to('cpu')
        
        # Convert each parameter to float32
        for param in model.parameters():
            param.data = param.data.float()
        for buffer_name, buffer in model.named_buffers():
            if buffer.is_floating_point():
                buffer.data = buffer.data.float()
        
        model.eval()
        
        # Fuse Conv+BN layers if method exists
        if hasattr(model, 'fuse'):
            model = model.fuse()
        
        # Reparameterize RepNeXt blocks for faster inference
        if reparameterize and hasattr(model, 'reparameterize'):
            print("Reparameterizing RepNeXt blocks for faster inference...")
            model = model.reparameterize()
        
        model.to(self.device)
        
        # Get class info
        self.num_classes = model.nc if hasattr(model, 'nc') else 80
        self.stride = model.stride if hasattr(model, 'stride') else [8, 16, 32]
        
        print(f"Model loaded: {self.num_classes} classes, device={self.device}")
        return model

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
        """Draw detections on image.
        
        Args:
            img: BGR image
            detections: Array of [x1, y1, x2, y2, conf, class_id]
            class_names: List of class names
            
        Returns:
            img: Image with drawn detections
        """
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


def load_class_names(config_path: str) -> list:
    """Load class names from config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if 'names' in config:
            if isinstance(config['names'], dict):
                return list(config['names'].values())
            return config['names']
    except Exception as e:
        print(f"Warning: Could not load class names from {config_path}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description='YOLOv8-RepNeXt Inference')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    parser.add_argument('--config', type=str, default='config/config.yml', help='Config file for class names')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--no-reparam', action='store_true', help='Disable reparameterization')
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = YOLOInference(
        weights_path=args.weights,
        device=args.device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        img_size=args.img_size,
        reparameterize=not args.no_reparam
    )
    
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
    
    # Video writer for saving
    writer = None
    if args.save and is_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = Path(args.output) / f"result_{Path(args.source).stem}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    frame_idx = 0
    total_time = 0
    
    print(f"\nRunning inference on {args.source}...")
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
        
        # Add FPS info
        fps_text = f"FPS: {1000/inf_time:.1f} | Objects: {len(detections)}"
        cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show
        if args.show:
            cv2.imshow('YOLOv8-RepNeXt Inference', result)
            key = cv2.waitKey(1 if is_video else 0) & 0xFF
            if key == ord('q'):
                break
        
        # Save
        if args.save:
            if is_video and writer:
                writer.write(result)
            else:
                out_path = Path(args.output) / f"result_{Path(args.source).name}"
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
    print(f"Processed {frame_idx} frames")
    if frame_idx > 0:
        print(f"Average inference time: {total_time / frame_idx:.1f}ms")
        print(f"Average FPS: {1000 * frame_idx / total_time:.1f}")


if __name__ == '__main__':
    main()

