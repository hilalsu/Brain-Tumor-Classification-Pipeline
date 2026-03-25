from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def infer_class_names(training_dir: str, *, explicit: Optional[List[str]] = None) -> List[str]:
    if explicit is not None:
        return list(explicit)

    p = Path(training_dir)
    if not p.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")

    class_names = [x.name for x in p.iterdir() if x.is_dir()]
    class_names.sort()
    if not class_names:
        raise ValueError(f"No class subdirectories found under: {training_dir}")
    return class_names


def collect_image_paths(
    split_dir: str,
    class_names: List[str],
    *,
    max_images_per_class: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    Returns (paths, labels) using folder structure:
      split_dir/<label>/*.jpg
    """

    paths: List[str] = []
    labels: List[str] = []
    split_path = Path(split_dir)
    for cls in class_names:
        cls_dir = split_path / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")

        candidates = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        candidates.sort()
        if max_images_per_class is not None:
            candidates = candidates[:max_images_per_class]

        paths.extend([str(p) for p in candidates])
        labels.extend([cls] * len(candidates))

    if not paths:
        raise ValueError(f"No images found in split_dir={split_dir} for classes={class_names}")
    return paths, labels


class MRIDataset(Dataset):
    def __init__(self, paths: Sequence[str], labels: Sequence[int], transform):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img) if self.transform is not None else img
        y = self.labels[idx]
        return x, y


class IndexMappedDataset(Dataset):
    """Dataset wrapper that remaps indices (useful for oversampling)."""

    def __init__(self, base_dataset: Dataset, mapped_indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.mapped_indices = list(mapped_indices)

    def __len__(self) -> int:
        return len(self.mapped_indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.mapped_indices[idx]]


def oversample_indices(labels: np.ndarray, *, random_state: int) -> np.ndarray:
    """
    Oversample so each class has the same number of samples as the majority class.
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) <= 1:
        return np.arange(len(labels), dtype=np.int64)

    max_count = int(counts.max())
    rng = np.random.default_rng(random_state)

    all_indices: List[int] = []
    for c in classes:
        cls_indices = np.where(labels == c)[0]
        if len(cls_indices) == 0:
            continue
        if len(cls_indices) < max_count:
            extra = rng.choice(cls_indices, size=max_count - len(cls_indices), replace=True)
            cls_indices = np.concatenate([cls_indices, extra])
        all_indices.extend(cls_indices.tolist())

    all_indices = np.array(all_indices, dtype=np.int64)
    rng.shuffle(all_indices)
    return all_indices


def make_train_val_splits(
    paths: List[str],
    labels_str: List[str],
    class_to_idx: Dict[str, int],
    *,
    val_fraction: float,
    random_state: int,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    y = np.array([class_to_idx[s] for s in labels_str], dtype=np.int64)
    idxs = np.arange(len(paths))

    train_idxs, val_idxs = train_test_split(
        idxs, test_size=val_fraction, random_state=random_state, stratify=y
    )

    train_paths = [paths[i] for i in train_idxs]
    val_paths = [paths[i] for i in val_idxs]
    train_labels = y[train_idxs].tolist()
    val_labels = y[val_idxs].tolist()
    return train_paths, train_labels, val_paths, val_labels


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

