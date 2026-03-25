from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


ContrastMethod = Literal["clahe", "hist_equal"]


@dataclass(frozen=True)
class ContrastConfig:
    method: ContrastMethod = "clahe"
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ContrastEnhancer:
    """
    Apply contrast enhancement to a PIL image.

    - Input: PIL image (any mode)
    - Output: PIL RGB image
    """

    def __init__(self, cfg: ContrastConfig):
        self.cfg = cfg
        self._clahe = cv2.createCLAHE(clipLimit=cfg.clip_limit, tileGridSize=cfg.tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        # MRI data is often grayscale. Convert to 8-bit grayscale for contrast ops.
        img_rgb = img.convert("RGB")
        img_np = np.array(img_rgb)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        if self.cfg.method == "clahe":
            enhanced = self._clahe.apply(gray)
        elif self.cfg.method == "hist_equal":
            enhanced = cv2.equalizeHist(gray)
        else:
            raise ValueError(f"Unknown contrast method: {self.cfg.method}")

        # Convert back to RGB by replicating the enhanced grayscale channel.
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)


def build_transforms(
    *,
    train: bool,
    image_size: int,
    enable_augmentation: bool,
    contrast_cfg: ContrastConfig,
    imagenet_normalize: bool = True,
) -> transforms.Compose:
    """
    Returns a torchvision transform pipeline.

    Notes:
    - `TF.to_tensor` converts pixels to float32 and scales them to [0, 1].
    - If `imagenet_normalize=True`, we further apply ImageNet mean/std
      to match pretrained backbones' expected input distribution.
    """

    contrast = ContrastEnhancer(contrast_cfg)

    common: List = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        contrast,
    ]

    if train and enable_augmentation:
        # Lightweight yet MRI-friendly augmentation; works on RGB after contrast enhancement.
        aug: List = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.9, 1.1),  # "zoom"
                shear=0,
                fill=0,
            ),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
        ]
        # Optional vertical flip; many brain tumor datasets tolerate it, but you can disable if needed.
        aug.append(transforms.RandomVerticalFlip(p=0.1))
        common.extend(aug)

    common.extend(
        [
            transforms.ToTensor(),  # scales to [0, 1]
            transforms.Lambda(
                lambda t: TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD) if imagenet_normalize else t
            ),
        ]
    )
    return transforms.Compose(common)

