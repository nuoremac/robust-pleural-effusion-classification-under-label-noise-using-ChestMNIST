"""Loading, subsampling and preprocessing of ChestMNIST for the effusion task.

This module implements the common protocol agreed by the team (Phase I interface).
It is kept free of any model or metric code so that Phase II and Phase III import
exactly the same feature matrices.

Two rules are enforced here because every downstream result depends on them:

1. the official MedMNIST validation and test splits are used unchanged;
2. the scaler and the PCA basis are fitted on the training subsample only, so no
   information from validation or test leaks into the representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR, IMAGE_SIDE, LABEL_NAMES, N_FEATURES, PCA_COMPONENTS,
    TARGET_INDEX, TARGET_NAME, TRAIN_SUBSET_SIZE,
)

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- raw
def load_split(split: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Return the raw uint8 images and the full 14-column label matrix of a split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    images = np.load(data_dir / f"{split}_images.npy")
    labels = np.load(data_dir / f"{split}_labels.npy")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"{split}: {images.shape[0]} images for {labels.shape[0]} labels")
    if images.shape[1:] != (IMAGE_SIDE, IMAGE_SIDE):
        raise ValueError(f"{split}: expected {IMAGE_SIDE}x{IMAGE_SIDE} images, got {images.shape[1:]}")
    if labels.shape[1] != len(LABEL_NAMES):
        raise ValueError(f"{split}: expected {len(LABEL_NAMES)} labels, got {labels.shape[1]}")
    return images, labels


def describe_dataset(data_dir: Path = DATA_DIR) -> dict:
    """Collect the descriptive statistics quoted in the article (shapes, prevalence, missing values)."""
    report: dict = {"target": TARGET_NAME, "target_index": TARGET_INDEX, "splits": {}}
    for split in SPLITS:
        images, labels = load_split(split, data_dir)
        y = labels[:, TARGET_INDEX].astype(np.int8)
        report["splits"][split] = {
            "n_samples": int(images.shape[0]),
            "image_shape": tuple(int(v) for v in images.shape[1:]),
            "intensity_min": int(images.min()),
            "intensity_max": int(images.max()),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "prevalence": float(y.mean()),
            # ChestMNIST is distributed as dense uint8 arrays: there is no
            # missing-value sentinel. The check is kept so the claim is verified
            # rather than assumed.
            "n_missing_pixels": int(np.isnan(images.astype(np.float32)).sum()),
            "n_missing_labels": int(np.isnan(labels.astype(np.float32)).sum()),
            "label_values": sorted(int(v) for v in np.unique(labels)),
        }
    _, train_labels = load_split("train", data_dir)
    report["train_prevalence_all_labels"] = {
        name: float(train_labels[:, i].mean()) for i, name in enumerate(LABEL_NAMES)
    }
    report["mean_findings_per_image_train"] = float(train_labels.sum(axis=1).mean())
    return report


# --------------------------------------------------------------------- subsample
def stratified_subset_indices(y: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Indices of a class-stratified subsample of ``y`` of the requested size.

    Stratification preserves the 11.8 percent effusion prevalence of the full
    training split, so the imbalance studied in Task 3.1 is a property of the
    data and not an artefact of the subsampling.
    """
    n = y.shape[0]
    if size >= n:
        return np.arange(n)
    keep, _ = train_test_split(
        np.arange(n), train_size=size, stratify=y, random_state=seed, shuffle=True
    )
    return np.sort(keep)


# ------------------------------------------------------------------- container
@dataclass
class Dataset:
    """Preprocessed matrices for one seed, in a single feature space."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    feature_space: str
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    def summary(self) -> dict:
        return {
            "feature_space": self.feature_space,
            "seed": self.seed,
            "n_features": self.n_features,
            "n_train": int(self.X_train.shape[0]),
            "n_val": int(self.X_val.shape[0]),
            "n_test": int(self.X_test.shape[0]),
            "train_prevalence": float(self.y_train.mean()),
            "val_prevalence": float(self.y_val.mean()),
            "test_prevalence": float(self.y_test.mean()),
        }


def _flatten(images: np.ndarray) -> np.ndarray:
    """28x28 uint8 images to a float32 matrix of 784 intensities in [0, 1]."""
    return images.reshape(images.shape[0], N_FEATURES).astype(np.float32) / 255.0


def build_dataset(
    seed: int,
    feature_space: str = "standardized",
    subset_size: int = TRAIN_SUBSET_SIZE,
    data_dir: Path = DATA_DIR,
) -> Dataset:
    """Assemble the matrices used by every model of the study.

    Parameters
    ----------
    feature_space:
        ``"standardized"`` keeps the 784 pixel features and applies per-feature
        standardization. ``"pca"`` applies the same standardization and then
        projects on the leading ``PCA_COMPONENTS`` principal directions; it is
        used only for the representation ablation of distance- and
        kernel-based models.
    """
    if feature_space not in {"standardized", "pca"}:
        raise ValueError(f"unknown feature_space {feature_space!r}")

    train_images, train_labels = load_split("train", data_dir)
    val_images, val_labels = load_split("val", data_dir)
    test_images, test_labels = load_split("test", data_dir)

    y_train_full = train_labels[:, TARGET_INDEX].astype(np.int8)
    keep = stratified_subset_indices(y_train_full, subset_size, seed)

    X_train = _flatten(train_images[keep])
    y_train = y_train_full[keep]
    X_val = _flatten(val_images)
    y_val = val_labels[:, TARGET_INDEX].astype(np.int8)
    X_test = _flatten(test_images)
    y_test = test_labels[:, TARGET_INDEX].astype(np.int8)

    scaler = StandardScaler().fit(X_train)              # fitted on train only
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    meta: dict = {"scaler": "StandardScaler", "subset_size": int(len(keep))}
    if feature_space == "pca":
        pca = PCA(n_components=PCA_COMPONENTS, random_state=seed).fit(X_train)
        X_train = pca.transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        meta["pca_components"] = PCA_COMPONENTS
        meta["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())

    return Dataset(
        X_train=np.ascontiguousarray(X_train, dtype=np.float32),
        y_train=y_train,
        X_val=np.ascontiguousarray(X_val, dtype=np.float32),
        y_val=y_val,
        X_test=np.ascontiguousarray(X_test, dtype=np.float32),
        y_test=y_test,
        train_indices=keep,
        feature_space=feature_space,
        seed=seed,
        meta=meta,
    )
