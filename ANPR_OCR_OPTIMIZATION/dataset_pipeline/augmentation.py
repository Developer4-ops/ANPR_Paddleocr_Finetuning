"""
ANPR OCR Optimization Pipeline
dataset_pipeline/augmentation.py

Domain-specific image augmentations tailored for Indian license plates:
- Mild Gaussian blur (simulates motion/unfocused cameras)
- Brightness / Contrast jitter (simulates headlights, glare, shadows)
- Mild perspective warp (simulates angled camera views)
- Gaussian / Salt & Pepper noise (simulates dirty plates / sensor noise)
- JPEG compression artifacts (simulates low bandwidth streaming)
"""

import cv2
import random
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class PlateAugmentor:
    """
    Applies realistic physical distortions to license plate crops.
    """

    def __init__(
        self,
        blur_prob: float = 0.5,
        brightness_contrast_prob: float = 0.6,
        perspective_prob: float = 0.4,
        noise_prob: float = 0.4,
        jpeg_prob: float = 0.3
    ):
        self.blur_prob = blur_prob
        self.bc_prob = brightness_contrast_prob
        self.persp_prob = perspective_prob
        self.noise_prob = noise_prob
        self.jpeg_prob = jpeg_prob

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a random combination of augmentations to an image array (BGR).
        """
        augmented = image.copy()

        # 1. Blur
        if random.random() < self.blur_prob:
            k_size = random.choice([3, 5])
            augmented = cv2.GaussianBlur(augmented, (k_size, k_size), 0)

        # 2. Brightness & Contrast
        if random.random() < self.bc_prob:
            alpha = random.uniform(0.7, 1.3)  # Contrast control
            beta = random.randint(-35, 35)    # Brightness control
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

        # 3. Perspective Distortion
        if random.random() < self.persp_prob:
            h, w = augmented.shape[:2]
            dx = int(w * random.uniform(0.02, 0.07))
            dy = int(h * random.uniform(0.02, 0.07))

            src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
            dst_pts = np.float32([
                [random.randint(0, dx), random.randint(0, dy)],
                [w - 1 - random.randint(0, dx), random.randint(0, dy)],
                [random.randint(0, dx), h - 1 - random.randint(0, dy)],
                [w - 1 - random.randint(0, dx), h - 1 - random.randint(0, dy)]
            ])

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            augmented = cv2.warpPerspective(augmented, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 4. Noise
        if random.random() < self.noise_prob:
            gauss = np.random.normal(0, random.uniform(5, 15), augmented.shape).astype(np.float32)
            noisy = augmented.astype(np.float32) + gauss
            augmented = np.clip(noisy, 0, 255).astype(np.uint8)

        # 5. JPEG Compression Artifacts
        if random.random() < self.jpeg_prob:
            quality = random.randint(50, 80)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc = cv2.imencode('.jpg', augmented, encode_param)
            augmented = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        return augmented

    def augment_file(self, src_path: Path, dst_path: Path) -> bool:
        """Reads image, augments, and saves to destination path."""
        img = cv2.imread(str(src_path))
        if img is None:
            logger.warning(f"Could not read image for augmentation: {src_path}")
            return False
        aug_img = self.augment(img)
        return cv2.imwrite(str(dst_path), aug_img)
