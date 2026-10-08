"""Task 3.1: evaluate the seven baseline families and select three of them.

Procedure, run on the primary seed:

1. build the common training subsample and the standardized 784 dimensional
   feature space (``data.build_dataset``);
2. for every family, fit each candidate of its standard grid on the clean
   training labels and score it on the untouched validation split;
3. keep, per family, the candidate with the highest validation average
   precision; this is the baseline reported in Table 2;
4. evaluate that candidate once on the test split, which is touched only here
   and only after the choice has been made;
5. rank the seven families by validation average precision and record the top
   three, which Task 3.2 then re-trains under label noise;
6. as a representation check, re-run the two families whose behaviour depends
   directly on the ambient dimension (KNN and the RBF kernel machine) on a
   50 component PCA projection fitted on the training subsample only.

Outputs: ``results/results_baselines.csv``, ``results/results_representation.csv``,
``results/selection.json``, the score vectors needed by the figures under
``results/scores/`` and one MLflow run per candidate.

Usage: python src/run_baselines.py [--seed 42] [--no-mlflow]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data
import evaluate
import models
import tracking
from config import (
    ARTIFACTS_DIR, MLFLOW_TRACKING_URI, N_SELECTED_MODELS, PCA_COMPONENTS,
    PRIMARY_SEED, RESULTS_DIR, SELECTION_METRIC, TARGET_NAME, TRAIN_SUBSET_SIZE,
)

SCORES_DIR = RESULTS_DIR / "scores"


def _rows_for(
    family: str,
    params: dict,
    feature_space: str,
    seed: int,
    results: dict[str, evaluate.EvalResult],
    val_threshold: float,
) -> dict:
    """Flatten one candidate into a single CSV row."""
    val, test = results["val"], results["test"]
    row: dict = {
        "family": family,
        "model": models.DISPLAY_NAMES[family],
        "params": models.params_label(params),
        "feature_space": feature_space,
        "n_features": None,
        "seed": seed,
        "score_source": val.score_source,
        "fit_seconds": round(val.fit_seconds, 3),
        "val_predict_seconds": round(val.predict_seconds, 3),
        "test_predict_seconds": round(test.predict_seconds, 3),
    }
    row.update({f"val_{k}": v for k, v in val.metrics.items()})
    row.update({f"test_{k}": v for k, v in test.metrics.items()})
    # Operating point transferred from validation to test, reported next to the
    # default threshold so the two can be told apart in the discussion.
    tuned_val = evaluate.metrics_at_threshold(val.y_true, val.y_score, val_threshold)
    tuned_test = evaluate.metrics_at_threshold(test.y_true, test.y_score, val_threshold)
    row["val_selected_threshold"] = round(float(val_threshold), 6)
    row["val_f1_at_val_threshold"] = tuned_val["f1"]
    row["test_f1_at_val_threshold"] = tuned_test["f1"]
    row["test_precision_at_val_threshold"] = tuned_test["precision"]
    row["test_recall_at_val_threshold"] = tuned_test["recall"]
    return row


def run(seed: int = PRIMARY_SEED, use_mlflow: bool = True) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if use_mlflow:
        if tracking.init_tracking():
            print(f"MLflow tracking enabled, backend store: {MLFLOW_TRACKING_URI}")
        else:
            print("MLflow is not installed: metrics are written to CSV only.")

    # ---------------------------------------------------------------- data
    print(f"\nDataset description for target '{TARGET_NAME}'")
    description = data.describe_dataset()
    (RESULTS_DIR / "dataset_description.json").write_text(
        json.dumps(description, indent=2), encoding="utf-8"
    )
    for split, info in description["splits"].items():
        print(
            f"  {split:5s} n={info['n_samples']:6d} positives={info['n_positive']:5d} "
            f"prevalence={info['prevalence']:.4f} missing_labels={info['n_missing_labels']}"
        )

    primary = data.build_dataset(seed=seed, feature_space="standardized")
    print(f"\nTraining subsample: {primary.summary()}")
    np.save(RESULTS_DIR / f"train_indices_seed{seed}.npy", primary.train_indices)

    splits = {"val": (primary.X_val, primary.y_val), "test": (primary.X_test, primary.y_test)}

    # ------------------------------------------------------ candidate sweep
    rows: list[dict] = []
    best_per_family: dict[str, dict] = {}

    for family, grid in models.STANDARD_GRIDS.items():
        print(f"\n[{family}] {len(grid)} candidate(s)")
        family_rows: list[dict] = []
        for params in grid:
            label = models.params_label(params)
            started = time.perf_counter()
            model = models.build_model(family, params, seed)
            model, results = evaluate.fit_and_evaluate(
                model, primary.X_train, primary.y_train, splits
            )
            threshold, _ = evaluate.best_f1_threshold(
                results["val"].y_true, results["val"].y_score
            )
            row = _rows_for(family, params, "standardized", seed, results, threshold)
            row["n_features"] = primary.n_features
            family_rows.append(row)
            rows.append(row)

            np.savez_compressed(
                SCORES_DIR / f"baseline_{family}_{label.replace(' ', '')}_seed{seed}.npz",
                val_score=results["val"].y_score,
                val_true=results["val"].y_true,
                test_score=results["test"].y_score,
                test_true=results["test"].y_true,
            )

            print(
                f"  {label:22s} val AP={row['val_average_precision']:.4f} "
                f"val AUC={row['val_roc_auc']:.4f} val F1={row['val_f1']:.4f} "
                f"test AP={row['test_average_precision']:.4f} "
                f"({time.perf_counter() - started:.1f}s)"
            )

            if use_mlflow:
                with tracking.run(
                    f"baseline-{family}-{label}-seed{seed}",
                    tags={
                        "phase": "III",
                        "task": "3.1",
                        "label_condition": "clean",
                        "configuration": "standard",
                        "family": family,
                    },
                ):
                    tracking.log_params(
                        {
                            "family": family,
                            **params,
                            "seed": seed,
                            "feature_space": "standardized",
                            "n_features": primary.n_features,
                            "train_subset_size": TRAIN_SUBSET_SIZE,
                            "noise_rate": 0.0,
                            "score_source": results["val"].score_source,
                            "selection_metric": SELECTION_METRIC,
                        }
                    )
                    tracking.log_metrics(
                        {
                            **results["val"].flat("val"),
                            **results["test"].flat("test"),
                            "fit_seconds": results["val"].fit_seconds,
                            "val_selected_threshold": threshold,
                            "test_f1_at_val_threshold": row["test_f1_at_val_threshold"],
                        }
                    )
                    tracking.log_model(model)

        best = max(family_rows, key=lambda r: r[f"val_{SELECTION_METRIC}"])
        best_per_family[family] = best
        print(f"  -> retained: {best['params']} (val AP={best[f'val_{SELECTION_METRIC}']:.4f})")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(RESULTS_DIR / "results_candidates.csv", index=False)

    baselines = pd.DataFrame(best_per_family.values()).sort_values(
        f"val_{SELECTION_METRIC}", ascending=False
    )
    baselines.to_csv(RESULTS_DIR / "results_baselines.csv", index=False)

    # ------------------------------------------------------------ selection
    selected = baselines["family"].head(N_SELECTED_MODELS).tolist()
    selection = {
        "seed": seed,
        "selection_metric": f"val_{SELECTION_METRIC}",
        "selected_families": selected,
        "selected_params": {f: best_per_family[f]["params"] for f in selected},
        "standard_params": {
            f: next(
                p for p in models.STANDARD_GRIDS[f]
                if models.params_label(p) == best_per_family[f]["params"]
            )
            for f in selected
        },
        "ranking": [
            {"family": r["family"], "val_average_precision": r["val_average_precision"]}
            for _, r in baselines.iterrows()
        ],
        "val_prevalence": float(primary.y_val.mean()),
        "test_prevalence": float(primary.y_test.mean()),
        "train_prevalence": float(primary.y_train.mean()),
    }
    (RESULTS_DIR / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"\nSelected by val {SELECTION_METRIC}: {selected}")

    # ------------------------------------------------- representation check
    print(f"\nRepresentation check on a {PCA_COMPONENTS} component PCA projection")
    reduced = data.build_dataset(seed=seed, feature_space="pca")
    print(f"  explained variance: {reduced.meta['pca_explained_variance']:.4f}")
    reduced_splits = {
        "val": (reduced.X_val, reduced.y_val),
        "test": (reduced.X_test, reduced.y_test),
    }
    ablation: list[dict] = []
    for family, params in [("knn", {"n_neighbors": 5}), ("svm_rbf", {"C": 1.0})]:
        model = models.build_model(family, params, seed)
        model, results = evaluate.fit_and_evaluate(
            model, reduced.X_train, reduced.y_train, reduced_splits
        )
        threshold, _ = evaluate.best_f1_threshold(results["val"].y_true, results["val"].y_score)
        row = _rows_for(family, params, f"pca{PCA_COMPONENTS}", seed, results, threshold)
        row["n_features"] = reduced.n_features
        row["pca_explained_variance"] = reduced.meta["pca_explained_variance"]
        ablation.append(row)
        print(
            f"  {family:10s} {models.params_label(params):16s} "
            f"val AP={row['val_average_precision']:.4f} test AP={row['test_average_precision']:.4f}"
        )
        if use_mlflow:
            with tracking.run(
                f"representation-{family}-pca{PCA_COMPONENTS}-seed{seed}",
                tags={"phase": "III", "task": "3.1", "label_condition": "clean",
                      "configuration": "standard", "family": family},
            ):
                tracking.log_params(
                    {"family": family, **params, "seed": seed,
                     "feature_space": f"pca{PCA_COMPONENTS}", "n_features": reduced.n_features,
                     "pca_explained_variance": reduced.meta["pca_explained_variance"]}
                )
                tracking.log_metrics(
                    {**results["val"].flat("val"), **results["test"].flat("test")}
                )

    representation_path = RESULTS_DIR / "results_representation.csv"
    pd.DataFrame(ablation).to_csv(representation_path, index=False)
    print(f"\nWritten: {RESULTS_DIR / 'results_baselines.csv'} and {representation_path}")
    return baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    run(seed=args.seed, use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
