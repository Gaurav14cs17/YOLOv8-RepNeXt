"""YOLO Dataset for object detection training.

Handles image loading, caching, and augmentation pipeline.
"""

import hashlib
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from .transforms import wh2xy, xy2wh, Resizer, GeometricTransform, resample
from .augmentations import HSVAugmentation, Albumentations, MixUpAugmentation
from .mosaic import MosaicLoader


FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')


class Dataset(TorchDataset):
    """YOLO dataset with mosaic and augmentation support.

    Supports:
    - Mosaic augmentation (4-image combination)
    - MixUp augmentation
    - HSV color augmentation
    - Random perspective transforms
    - Flip augmentations
    - Albumentations (if installed)
    """

    def __init__(self, filenames: list, input_size: int,
                 params: dict, augment: bool = True):
        """Initialize dataset.

        Args:
            filenames: List of image file paths
            input_size: Target image size
            params: Training parameters
            augment: Whether to apply augmentation
        """
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size

        # Load and cache labels
        cache = self._load_labels(filenames)
        labels, shapes = zip(*cache.values())

        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.filenames = list(cache.keys())
        self.n = len(shapes)
        self.indices = range(self.n)

        # Initialize augmentation components
        self.albumentations = Albumentations()
        self.mosaic_loader = MosaicLoader(input_size)
        self.resizer = Resizer()
        self.hsv_aug = HSVAugmentation()
        self.mixup_aug = MixUpAugmentation()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple:
        """Get a training sample.

        Args:
            index: Sample index

        Returns:
            Tuple of (image_tensor, target_tensor, shapes)
        """
        index = self.indices[index]
        params = self.params
        use_mosaic = self.mosaic and random.random() < params['mosaic']

        if use_mosaic:
            shapes = None
            image, label = self._load_mosaic(index)

            # MixUp augmentation
            if random.random() < self.params.get('mix_up', self.params.get('mixup', 0.0)):
                mix_idx = random.choice(self.indices)
                image2, label2 = self._load_mosaic(mix_idx)
                image, label = self.mixup_aug.mix_up(image, label, image2, label2)
        else:
            image, shape = self._load_image(index)
            h, w = image.shape[:2]

            # Resize with letterbox
            image, ratio, pad = self.resizer.resize(image, self.input_size, self.augment)
            pre_ratio = (h / shape[0], w / shape[1])
            gain = (ratio[0] * pre_ratio[0], ratio[1] * pre_ratio[1])
            shapes = shape, (gain, pad)

            # Transform labels
            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])

            if self.augment:
                image, label = GeometricTransform.random_perspective(image, label, params)

        # Convert labels to normalized format
        nl = len(label)
        if nl:
            label[:, 1:5] = xy2wh(label[:, 1:5], image.shape[1], image.shape[0])

        # Apply augmentations
        if self.augment:
            image, label = self._apply_augmentations(image, label)
            nl = len(label)

        # Create target tensor
        target = torch.zeros((nl, 6))
        if nl:
            target[:, 1:] = torch.from_numpy(label)

        # Convert HWC BGR to CHW RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)

        return torch.from_numpy(sample), target, shapes

    def _load_image(self, i: int) -> tuple:
        """Load and optionally resize image.

        Args:
            i: Image index

        Returns:
            Tuple of (image, original_shape)
        """
        image = cv2.imread(self.filenames[i])
        if image is None:
            raise FileNotFoundError(f'Cannot read image: {self.filenames[i]}')
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)

        if r != 1:
            interp = resample() if self.augment else cv2.INTER_LINEAR
            image = cv2.resize(image, dsize=(int(w * r), int(h * r)),
                              interpolation=interp)

        return image, (h, w)

    def _load_mosaic(self, index: int) -> tuple:
        """Load 4-image mosaic.

        Args:
            index: Primary image index

        Returns:
            Tuple of (mosaic_image, mosaic_labels)
        """
        indices = [index] + random.choices(self.indices, k=3)
        return self.mosaic_loader.create_mosaic(
            indices, self._load_image, self.labels, self.params
        )

    def _apply_augmentations(self, image: np.ndarray, label: np.ndarray) -> tuple:
        """Apply augmentation pipeline.

        Args:
            image: Input image
            label: Labels

        Returns:
            Tuple of (augmented_image, augmented_labels)
        """
        # Albumentations
        image, label = self.albumentations(image, label)

        # HSV augmentation
        self.hsv_aug.augment(image, self.params)

        nl = len(label)

        # Flip up-down
        if random.random() < self.params['flip_ud']:
            image = np.flipud(image)
            if nl:
                label[:, 2] = 1 - label[:, 2]

        # Flip left-right
        if random.random() < self.params['flip_lr']:
            image = np.fliplr(image)
            if nl:
                label[:, 1] = 1 - label[:, 1]

        return image, label

    def _load_labels(self, filenames: list) -> dict:
        """Load and cache labels for all images.

        Args:
            filenames: List of image paths

        Returns:
            Dict mapping filename to (labels, shape)
        """
        cache_id = hashlib.md5(''.join(sorted(filenames)).encode()).hexdigest()[:16]
        cache_path = f'{os.path.dirname(filenames[0])}.{cache_id}.cache'

        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=False)

        cache = {}
        for filename in filenames:
            try:
                # Verify image
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()
                shape = image.size

                assert shape[0] > 9 and shape[1] > 9, f'Image too small: {shape}'
                assert image.format.lower() in FORMATS, f'Invalid format: {image.format}'

                # Load labels
                label_path = filename.replace(
                    f'{os.sep}images{os.sep}',
                    f'{os.sep}labels{os.sep}'
                ).rsplit('.', 1)[0] + '.txt'

                if os.path.isfile(label_path):
                    with open(label_path) as f:
                        label = [x.split() for x in f.read().strip().splitlines() if x]
                        label = np.array(label, dtype=np.float32)

                    if len(label):
                        assert label.shape[1] == 5, 'Labels need 5 columns'
                        assert (label >= 0).all(), 'Negative values found'
                        assert (label[:, 1:] <= 1).all(), 'Values > 1 found'

                        # Remove duplicates
                        _, unique_idx = np.unique(label, axis=0, return_index=True)
                        label = label[unique_idx]
                    else:
                        label = np.zeros((0, 5), dtype=np.float32)
                else:
                    label = np.zeros((0, 5), dtype=np.float32)

                cache[filename] = [label, shape]

            except (FileNotFoundError, AssertionError):
                continue

        torch.save(cache, cache_path)
        return cache

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        """Custom collate function for DataLoader.

        Args:
            batch: List of (sample, target, shapes) tuples

        Returns:
            Tuple of (stacked_samples, concatenated_targets, shapes)
        """
        samples, targets, shapes = zip(*batch)

        for i, item in enumerate(targets):
            item[:, 0] = i  # Add batch index

        return torch.stack(samples, 0), torch.cat(targets, 0), shapes

