from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from typing import List
import csv
import time
import traceback
from pathlib import Path
import sys
import importlib

import numpy as np

# Ensure `brain_tumor_pipeline` imports work when running:
#   python brain_tumor_pipeline/run_pipeline.py
_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from brain_tumor_pipeline.src.config import PipelineConfig
from brain_tumor_pipeline.src.data import (
    MRIDataset,
    build_dataloader,
    collect_image_paths,
    infer_class_names,
    make_train_val_splits,
    oversample_indices,
    IndexMappedDataset,
)
from brain_tumor_pipeline.src.features import DeepFeatureExtractor
from brain_tumor_pipeline.src.fusion import FeatureFusionScaler
from brain_tumor_pipeline.src.gwo_feature_selector import GreyWolfFeatureSelector
from brain_tumor_pipeline.src.preprocess import ContrastConfig, build_transforms
from brain_tumor_pipeline.src.utils import get_run_output_dir, save_json, set_global_seed, setup_logger
from brain_tumor_pipeline.src.visualize import plot_confusion_matrix, plot_gwo_convergence, plot_roc_curves_multiclass
from brain_tumor_pipeline.src.models import evaluate_classifiers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Brain tumor classification pipeline (deep features + GWO + classifiers).")
    p.add_argument("--dataset-root", type=str, default="dataset")
    p.add_argument("--output-base-dir", type=str, default="brain_tumor_pipeline/outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="e.g. cuda, cpu, or 'cuda:0'")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--no-class-balance", action="store_true")
    p.add_argument("--imagenet-normalize", action="store_true", help="Apply ImageNet mean/std after scaling to [0,1].")
    p.add_argument("--contrast-method", type=str, default="clahe", choices=["clahe", "hist_equal"])
    p.add_argument("--max-images-per-class", type=int, default=None, help="Limit images per class for faster runs.")
    p.add_argument("--min-features", type=int, default=None)
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--population-size", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    dataset_root_path = Path(args.dataset_root)
    if not dataset_root_path.is_absolute():
        dataset_root_path = workspace_root / dataset_root_path

    output_base_dir_path = Path(args.output_base_dir)
    if not output_base_dir_path.is_absolute():
        output_base_dir_path = workspace_root / output_base_dir_path

    args.dataset_root = str(dataset_root_path)
    args.output_base_dir = str(output_base_dir_path)

    cfg = PipelineConfig()

    fs_overrides = {}
    if args.min_features is not None:
        fs_overrides["min_features"] = int(args.min_features)
    if args.max_features is not None:
        fs_overrides["max_features"] = int(args.max_features)
    if args.population_size is not None:
        fs_overrides["population_size"] = int(args.population_size)
    if args.max_iterations is not None:
        fs_overrides["max_iterations"] = int(args.max_iterations)

    cfg = replace(
        cfg,
        output_base_dir=args.output_base_dir,
        seed=int(args.seed),
        device=args.device,
        data=replace(
            cfg.data,
            dataset_root=args.dataset_root,
            image_size=int(args.image_size),
            val_fraction=float(args.val_fraction),
            enable_augmentation=not args.no_augmentation,
            enable_class_balance=not args.no_class_balance,
            class_names=None,
            max_images_per_class=args.max_images_per_class,
        ),
        feature_selection=replace(cfg.feature_selection, seed=int(args.seed), **fs_overrides),
        models=replace(cfg.models, random_state=int(args.seed)),
    )

    set_global_seed(cfg.seed)

    out_dir = get_run_output_dir(cfg.output_base_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(out_dir, name="brain_tumor")
    logger.info("Starting pipeline")
    logger.info("Config: %s", cfg)
    logger.info("Resolved dataset_root: %s", cfg.data.dataset_root)
    logger.info("Resolved output_base_dir: %s", cfg.output_base_dir)
    logger.info("Resolved out_dir: %s", out_dir)

    # Early dependency checks
    missing: List[str] = []
    for mod_name in ["timm", "cv2", "torch", "torchvision", "sklearn", "PIL"]:
        try:
            importlib.import_module(mod_name)
        except Exception:
            missing.append(mod_name)

    if missing:
        err_payload = {
            "error": "Missing required dependencies",
            "missing_modules": missing,
            "hint": "Run: pip install -r brain_tumor_pipeline\\requirements.txt",
        }
        save_json(os.path.join(out_dir, "dependency_error.json"), err_payload)
        logger.error("Dependency check failed: %s", missing)
        print(f"Missing dependencies: {missing}")
        print("Run: pip install -r brain_tumor_pipeline\\requirements.txt")
        return

    contrast_cfg = ContrastConfig(method=args.contrast_method)

    training_dir = os.path.join(cfg.data.dataset_root, cfg.data.training_dirname)
    testing_dir = os.path.join(cfg.data.dataset_root, cfg.data.testing_dirname)

    class_names = infer_class_names(training_dir, explicit=cfg.data.class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    logger.info("Classes: %s", class_names)

    train_paths_all, train_labels_all_str = collect_image_paths(
        training_dir,
        class_names,
        max_images_per_class=cfg.data.max_images_per_class,
    )

    train_paths, train_labels, val_paths, val_labels = make_train_val_splits(
        train_paths_all,
        train_labels_all_str,
        class_to_idx,
        val_fraction=cfg.data.val_fraction,
        random_state=cfg.seed,
    )

    test_paths, test_labels_str = collect_image_paths(
        testing_dir,
        class_names,
        max_images_per_class=cfg.data.max_images_per_class,
    )
    test_labels = [class_to_idx[s] for s in test_labels_str]

    train_transform = build_transforms(
        train=True,
        image_size=cfg.data.image_size,
        enable_augmentation=cfg.data.enable_augmentation,
        contrast_cfg=contrast_cfg,
        imagenet_normalize=args.imagenet_normalize,
    )
    eval_transform = build_transforms(
        train=False,
        image_size=cfg.data.image_size,
        enable_augmentation=False,
        contrast_cfg=contrast_cfg,
        imagenet_normalize=args.imagenet_normalize,
    )

    train_base_ds = MRIDataset(train_paths, train_labels, train_transform)
    val_ds = MRIDataset(val_paths, val_labels, eval_transform)
    test_ds = MRIDataset(test_paths, test_labels, eval_transform)

    if cfg.data.enable_class_balance and len(set(train_labels)) > 1:
        y_train = np.array(train_labels, dtype=np.int64)
        mapped = oversample_indices(y_train, random_state=cfg.seed)
        train_ds = IndexMappedDataset(train_base_ds, mapped)
        logger.info("Class balance enabled: oversampled train size=%d", len(train_ds))
    else:
        train_ds = train_base_ds

    device = cfg.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    logger.info("Using device: %s", device)

    train_loader = build_dataloader(train_ds, batch_size=cfg.feature_extractor.batch_size, num_workers=cfg.feature_extractor.num_workers, shuffle=False, seed=cfg.seed)
    val_loader = build_dataloader(val_ds, batch_size=cfg.feature_extractor.batch_size, num_workers=cfg.feature_extractor.num_workers, shuffle=False, seed=cfg.seed)
    test_loader = build_dataloader(test_ds, batch_size=cfg.feature_extractor.batch_size, num_workers=cfg.feature_extractor.num_workers, shuffle=False, seed=cfg.seed)

    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    logger.info("Extracting features for backbones: %s", cfg.feature_extractor.backbones)

    for backbone in cfg.feature_extractor.backbones:
        logger.info("Extracting backbone: %s", backbone)
        extractor = DeepFeatureExtractor(backbone, device=device, use_amp=cfg.feature_extractor.use_amp)
        X_train = extractor.extract(train_loader)
        X_val = extractor.extract(val_loader)
        X_test = extractor.extract(test_loader)
        train_parts.append(X_train)
        val_parts.append(X_val)
        test_parts.append(X_test)
        logger.info("Backbone=%s, feature_dim=%d", backbone, X_train.shape[1])

    fuser = FeatureFusionScaler(use_standard_scaler=cfg.fusion.use_standard_scaler)
    X_train_fused = fuser.fit_transform(train_parts)
    X_val_fused = fuser.transform(val_parts)
    X_test_fused = fuser.transform(test_parts)
    d_total = X_train_fused.shape[1]
    logger.info("Fused feature dimension: %d", d_total)

    y_train_np = np.array(train_labels, dtype=np.int64)
    if cfg.data.enable_class_balance and len(train_ds) != len(train_labels):
        mapped_labels: List[int] = [int(train_base_ds.labels[i]) for i in train_ds.mapped_indices]  # type: ignore[attr-defined]
        y_train_np = np.array(mapped_labels, dtype=np.int64)

    candidate_indices = np.arange(d_total, dtype=np.int64)
    X_train_gwo = X_train_fused
    if cfg.feature_selection.enable_preselection and d_total > cfg.feature_selection.preselect_k:
        logger.info(
            "Preselecting candidates for GWO: method=%s, k=%d (from d_total=%d)",
            cfg.feature_selection.preselect_method,
            cfg.feature_selection.preselect_k,
            d_total,
        )
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

        if cfg.feature_selection.preselect_method == "f_classif":
            selector_k = SelectKBest(score_func=f_classif, k=cfg.feature_selection.preselect_k)
        else:
            selector_k = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=cfg.seed), k=cfg.feature_selection.preselect_k)

        selector_k.fit(X_train_fused, y_train_np)
        candidate_indices = selector_k.get_support(indices=True).astype(np.int64)
        X_train_gwo = X_train_fused[:, candidate_indices]
        logger.info("GWO candidate count: %d", int(candidate_indices.shape[0]))

    logger.info("Running GWO feature selection (may take time)...")
    selector = GreyWolfFeatureSelector(
        cfg=cfg.feature_selection,
        total_features=int(candidate_indices.shape[0]),
        random_state=cfg.seed,
    )
    t0 = time.time()
    gwo_result = selector.fit(X_train_gwo, y_train_np)
    logger.info("GWO finished in %.2fs", time.time() - t0)

    selected_rel_indices = np.array(gwo_result["best_feature_indices"], dtype=np.int64)
    selected_indices = candidate_indices[selected_rel_indices]
    logger.info("Selected %d features (CV acc=%.4f)", int(selected_indices.shape[0]), gwo_result["best_accuracy_cv"])

    X_train_sel = X_train_fused[:, selected_indices]
    X_val_sel = X_val_fused[:, selected_indices]
    X_test_sel = X_test_fused[:, selected_indices]

    try:
        save_json(os.path.join(out_dir, "gwo_result.json"), gwo_result)
        plot_gwo_convergence(gwo_result.get("best_by_iter", []), save_path=os.path.join(out_dir, "gwo_convergence.png"))
        logger.info("Saved GWO artifacts.")
    except Exception as e:
        logger.error("Failed to save GWO artifacts: %s\n%s", str(e), traceback.format_exc())

    logger.info("Training classifiers...")
    t1 = time.time()
    try:
        results = evaluate_classifiers(
            X_train_sel, y_train_np,
            X_test_sel, np.array(test_labels, dtype=np.int64),
            random_state=cfg.seed,
            tune_hyperparameters=cfg.models.tune_hyperparameters,
            xgb_max_estimators=cfg.models.xgb_max_estimators,
        )
    except Exception as e:
        results = []
        logger.error("evaluate_classifiers failed: %s\n%s", str(e), traceback.format_exc())
    logger.info("Classifier evaluation finished in %.2fs", time.time() - t1)
    logger.info("Number of evaluated models: %d", len(results))

    best_res = None
    if results:
        best_res = sorted(results, key=lambda r: r.metrics.get("accuracy", -1.0), reverse=True)[0]
        logger.info("Best model=%s, metrics=%s", best_res.model_name, best_res.metrics)
    else:
        logger.warning("No models were evaluated")

    colormaps = ["Blues", "Greens", "Oranges", "Purples", "Reds"]
    for ri, res in enumerate(results):
        try:
            plot_confusion_matrix(
                res.confusion, class_names,
                save_path=os.path.join(out_dir, f"confusion_matrix_{res.model_name}.png"),
                cmap=colormaps[ri % len(colormaps)],
                title=f"{res.model_name} confusion matrix",
            )
        except Exception as e:
            logger.error("Failed to plot confusion matrix for %s: %s", res.model_name, str(e))

        try:
            if res.y_score is not None and getattr(res.y_score, "ndim", 0) == 2:
                plot_roc_curves_multiclass(
                    np.array(test_labels, dtype=np.int64), res.y_score, class_names,
                    save_path=os.path.join(out_dir, f"roc_{res.model_name}.png"),
                    title=f"{res.model_name} ROC curves (OvR)",
                )
        except Exception as e:
            logger.error("Failed to plot ROC for %s: %s", res.model_name, str(e))

    csv_path = os.path.join(out_dir, "all_results.csv")
    fieldnames = ["model_name", "feature_subset_size", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                writer.writerow({
                    "model_name": res.model_name,
                    "feature_subset_size": int(selected_indices.shape[0]),
                    "accuracy": res.metrics.get("accuracy", float("nan")),
                    "precision": res.metrics.get("precision", float("nan")),
                    "recall": res.metrics.get("recall", float("nan")),
                    "f1_score": res.metrics.get("f1_score", float("nan")),
                    "roc_auc": res.metrics.get("roc_auc", float("nan")),
                })
        logger.info("Saved CSV metrics to: %s", csv_path)
    except Exception as e:
        logger.error("Failed to save CSV: %s", str(e))

    summary = {
        "class_names": class_names,
        "best_feature_subset_size": int(selected_indices.shape[0]),
        "best_model": None if best_res is None else best_res.model_name,
        "best_model_metrics": {} if best_res is None else best_res.metrics,
        "warning": "No models were evaluated" if best_res is None else None,
        "results": [{"model_name": r.model_name, "metrics": r.metrics} for r in results],
    }
    try:
        save_json(os.path.join(out_dir, "results_summary.json"), summary)
    except Exception as e:
        logger.error("Failed to save results_summary.json: %s", str(e))

    print(f"Best feature subset size: {selected_indices.shape[0]}")
    if best_res is None:
        print("No models were evaluated")
    else:
        print(f"Best model: {best_res.model_name}")
        print(f"Best model metrics: {json.dumps(best_res.metrics, indent=2)}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
