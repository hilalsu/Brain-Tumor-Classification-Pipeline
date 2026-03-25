from __future__ import annotations

from typing import List, Optional

import numpy as np


class DeepFeatureExtractor:
    """
    Extract deep features from a pretrained backbone (without classification head).

    Implementation detail:
    - With timm, setting `num_classes=0` removes the head and returns the pooled representation.
    - Output is converted to a 2D array (N, D).
    """

    def __init__(self, backbone_name: str, device: str, *, use_amp: bool = True):
        self.backbone_name = backbone_name
        self.device = device
        self.use_amp = use_amp

        import torch

        import timm

        self.torch = torch
        self.timm = timm

        self.model = self._load_model()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.feature_dim: Optional[int] = None

    def _load_model(self):
        # `num_classes=0` returns the representation before the classifier head.
        model = self.timm.create_model(
            self.backbone_name,
            pretrained=True,
            num_classes=0,
            in_chans=3,
            global_pool="avg",
        )
        model.to(self.device)
        return model

    def extract(self, dataloader) -> np.ndarray:
        feats: List[np.ndarray] = []
        with self.torch.inference_mode():
            for x, _y in dataloader:
                x = x.to(self.device, non_blocking=True)

                if self.use_amp and self.device.startswith("cuda"):
                    with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                        out = self.model(x)
                else:
                    out = self.model(x)

                # out: (B, D) or (B, ..., D). Flatten to vectors.
                if out.ndim > 2:
                    out = out.view(out.shape[0], -1)

                out_np = out.detach().float().cpu().numpy()
                feats.append(out_np)

        X = np.concatenate(feats, axis=0)
        self.feature_dim = int(X.shape[1])
        return X

