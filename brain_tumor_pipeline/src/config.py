from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional


@dataclass(frozen=True)
class DataConfig:
    """
    Dataset layout expected by default:
      dataset_root/
        Training/<class_name>/*.jpg|*.png
        Testing/<class_name>/*.jpg|*.png
    """

    dataset_root: str = "dataset"
    training_dirname: str = "Training"
    testing_dirname: str = "Testing"
    class_names: Optional[List[str]] = None  # If None, infer from Training subfolders.

    image_size: int = 224
    val_fraction: float = 0.2

    # Data augmentation is applied during deep feature extraction for the training split.
    enable_augmentation: bool = True

    # Class balance is enforced by oversampling the training split (with augmentation during feature extraction).
    enable_class_balance: bool = True

    # Limit the number of images per class for quicker experiments (None = no limit).
    max_images_per_class: Optional[int] = None


@dataclass(frozen=True)
class FeatureExtractorConfig:
    backbones: List[str]
    batch_size: int = 16
    num_workers: int = 0

    # Use mixed precision during feature extraction when on CUDA.
    use_amp: bool = True


@dataclass(frozen=True)
class FeatureFusionConfig:
    # Whether to standardize concatenated features with StandardScaler.
    use_standard_scaler: bool = True


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """
    Grey Wolf Optimization (GWO) for binary feature subsets.
    """

    # GWO hyperparameters
    population_size: int = 30
    max_iterations: int = 80
    top_k_history: int = 8

    # "Adaptive convergence control" (non-linear decay of the coefficient "a").
    # Higher -> faster decay (more exploitation later).
    a_decay_power: float = 1.5

    # Reinitialization when stagnating.
    stagnation_patience: int = 12
    reinitialize_every: int = 0  # 0 disables periodic reinit; stagnation can still trigger it.
    mutation_flip_prob: float = 0.01

    # Evaluate each wolf by training an SVM (RBF) on a subset.
    svm_C: float = 1.0
    svm_gamma: str = "scale"
    cv_folds: int = 3

    # Subset size constraints (keeps selection feasible).
    min_features: int = 20
    max_features: int = 250

    # Fast candidate reduction before GWO (keeps the GWO search feasible).
    # GWO still determines the optimal final subset size automatically.
    enable_preselection: bool = True
    preselect_k: int = 800
    preselect_method: Literal["f_classif", "mutual_info_classif"] = "f_classif"

    # Multi-objective tradeoff: fitness = accuracy - lambda_size * (k / d)
    lambda_size_start: float = 0.05
    lambda_size_end: float = 0.25

    # Ensure reproducibility inside selection.
    seed: int = 42

    # To avoid pathological subsets.
    min_accuracy_gain_tol: float = 1e-4


@dataclass(frozen=True)
class ModelTrainingConfig:
    random_state: int = 42

    # Optional: try a small hyperparameter search. Turn on if you want better results.
    tune_hyperparameters: bool = False

    # For XGBoost (if installed)
    xgb_max_estimators: int = 400


@dataclass(frozen=True)
class PipelineConfig:
    output_base_dir: str = "brain_tumor_pipeline/outputs"
    seed: int = 42

    data: DataConfig = DataConfig()
    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig(
        backbones=[
            "efficientnet_b0",
            "resnet50",
        ]
    )
    fusion: FeatureFusionConfig = FeatureFusionConfig()
    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()
    models: ModelTrainingConfig = ModelTrainingConfig()

    # GPU / device selection
    device: Optional[str] = None  # None => auto


def default_config() -> PipelineConfig:
    return PipelineConfig()


def to_dict(config: PipelineConfig) -> Dict:
    return asdict(config)

